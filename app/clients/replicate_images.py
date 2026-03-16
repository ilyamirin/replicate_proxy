from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from app.clients.errors import (
    ReplicateError,
    UserFacingExecutionError,
    classify_replicate_error_message,
)
from app.clients.replicate_files import ReplicateFilesClient
from app.clients.retry import (
    is_transient_prediction_error,
    retry_replicate_error,
    retry_transport,
)
from app.config import ReplicateModel, Settings
from app.tool_schemas import ImageGenerationRequest, ImageGenerationResponse

TERMINAL_STATUSES = {"failed", "canceled", "aborted", "succeeded"}


class ReplicateImageClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        files_client: ReplicateFilesClient | None = None,
        download_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_http_client = http_client is None
        self._owns_files_client = files_client is None
        self._owns_download_client = download_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.replicate_base_url,
            timeout=settings.replicate_http_timeout_seconds,
        )
        self._files_client = files_client or ReplicateFilesClient(settings)
        self._download_client = download_client or httpx.AsyncClient(
            timeout=settings.replicate_http_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "replicate-proxy/0.1"},
        )

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()
        if self._owns_files_client:
            await self._files_client.aclose()
        if self._owns_download_client:
            await self._download_client.aclose()

    async def generate_image(
        self,
        model: ReplicateModel,
        payload: ImageGenerationRequest,
        *,
        tool_name: str,
    ) -> ImageGenerationResponse:
        try:
            resolved_prediction = await retry_replicate_error(
                lambda: self._create_and_resolve_prediction(model, payload),
                retries=self.settings.replicate_transport_retries,
                backoff_seconds=self.settings.replicate_transport_retry_backoff_seconds,
                should_retry=is_transient_prediction_error,
            )
            output_url = self._output_url(resolved_prediction)
            local_path = None
            if self.settings.replicate_image_download_output:
                local_path = await self._download_output(
                    output_url,
                    resolved_prediction.get("id"),
                )

            return ImageGenerationResponse(
                tool_name=tool_name,
                model=model.public_id,
                prompt=payload.prompt,
                output_url=output_url,
                local_path=local_path,
                output_format=payload.output_format,
                prediction_id=resolved_prediction.get("id"),
            )
        except ReplicateError as exc:
            category, retryable = classify_replicate_error_message(str(exc))
            raise UserFacingExecutionError(
                stage="image_generation",
                category=category,
                retryable=retryable,
                provider="replicate",
                model=model.public_id,
                technical_message=str(exc),
            ) from exc

    async def _create_and_resolve_prediction(
        self,
        model: ReplicateModel,
        payload: ImageGenerationRequest,
    ) -> dict:
        prediction = await self._create_prediction(model, payload)
        return await self._resolve_prediction(prediction)

    async def _create_prediction(
        self,
        model: ReplicateModel,
        payload: ImageGenerationRequest,
    ) -> dict:
        input_payload = {"prompt": payload.prompt}
        if payload.image_input:
            input_payload["image_input"] = [
                await self._files_client.prepare_image_url(source)
                for source in payload.image_input
            ]
        if payload.aspect_ratio is not None:
            input_payload["aspect_ratio"] = payload.aspect_ratio
        if payload.resolution is not None:
            input_payload["resolution"] = payload.resolution
        if payload.output_format is not None:
            input_payload["output_format"] = payload.output_format
        if payload.google_search:
            input_payload["google_search"] = True
        if payload.image_search:
            input_payload["image_search"] = True

        response = await self._request(
            "POST",
            self._prediction_url(model),
            headers={"Prefer": f"wait={self.settings.replicate_sync_wait_seconds}"},
            json={"input": input_payload},
        )
        return response.json()

    async def _resolve_prediction(self, prediction: dict) -> dict:
        if self._output_url(prediction, required=False):
            return prediction

        status = prediction.get("status")
        if status in TERMINAL_STATUSES:
            raise ReplicateError(self._error_message(prediction))

        get_url = prediction.get("urls", {}).get("get")
        if not get_url:
            raise ReplicateError("Replicate prediction did not include a polling URL.")

        deadline = monotonic() + self.settings.replicate_poll_timeout_seconds
        while monotonic() < deadline:
            await asyncio.sleep(self.settings.replicate_poll_interval_seconds)
            response = await self._request("GET", get_url)
            prediction = response.json()
            if self._output_url(prediction, required=False):
                return prediction
            if prediction.get("status") in TERMINAL_STATUSES:
                raise ReplicateError(self._error_message(prediction))

        raise ReplicateError("Replicate image prediction timed out before completion.")

    async def _download_output(self, output_url: str, prediction_id: str | None) -> str:
        response = await retry_transport(
            lambda: self._download_client.get(output_url),
            retries=self.settings.replicate_transport_retries,
            backoff_seconds=self.settings.replicate_transport_retry_backoff_seconds,
            error_prefix="Replicate output download transport error",
        )
        if response.is_error:
            raise ReplicateError(
                "Replicate output download error "
                f"{response.status_code}: {response.text}"
            )

        output_dir = Path(self.settings.replicate_image_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix(output_url, response.headers.get("Content-Type"))
        filename = prediction_id or uuid4().hex
        path = output_dir / f"{filename}{suffix}"
        path.write_bytes(response.content)
        return str(path.resolve())

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        if not self.settings.replicate_api_token:
            raise ReplicateError("REPLICATE_API_TOKEN is not set.")

        response = await retry_transport(
            lambda: self._http_client.request(
                method,
                url,
                headers=self._headers(kwargs.pop("headers", None)),
                **kwargs,
            ),
            retries=self.settings.replicate_transport_retries,
            backoff_seconds=self.settings.replicate_transport_retry_backoff_seconds,
            error_prefix="Replicate transport error",
        )
        if response.is_error:
            raise ReplicateError(
                f"Replicate API error {response.status_code}: {response.text}"
            )
        return response

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.settings.replicate_api_token}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    @staticmethod
    def _prediction_url(model: ReplicateModel) -> str:
        return f"/models/{model.owner}/{model.name}/predictions"

    @staticmethod
    def _output_url(prediction: dict, *, required: bool = True) -> str | None:
        output = prediction.get("output")
        if isinstance(output, list) and output:
            return str(output[0])
        if isinstance(output, str) and output:
            return output
        if required:
            raise ReplicateError("Replicate prediction did not return an output URL.")
        return None

    @staticmethod
    def _error_message(prediction: dict) -> str:
        error = prediction.get("error")
        if error:
            return f"Replicate prediction failed: {error}"
        status = prediction.get("status", "unknown")
        return f"Replicate prediction ended with status: {status}"

    @staticmethod
    def _suffix(output_url: str, content_type: str | None) -> str:
        if content_type == "image/png":
            return ".png"
        if content_type == "image/jpeg":
            return ".jpg"
        suffix = Path(urlparse(output_url).path).suffix
        return suffix or ".bin"
