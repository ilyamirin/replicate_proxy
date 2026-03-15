from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
from urllib.parse import urlparse
from uuid import uuid4

import httpx

from app.clients.errors import ReplicateError
from app.clients.replicate_files import ReplicateFilesClient
from app.clients.retry import retry_transport
from app.config import ReplicateModel, Settings
from app.tool_schemas import QwenImageEditRequest, QwenImageEditResponse

TERMINAL_STATUSES = {"failed", "canceled", "aborted", "succeeded"}


class ReplicateQwenEditClient:
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

    async def edit_image(
        self,
        model: ReplicateModel,
        payload: QwenImageEditRequest,
        *,
        tool_name: str,
    ) -> QwenImageEditResponse:
        prediction = await self._create_prediction(model, payload)
        resolved_prediction = await self._resolve_prediction(prediction)
        output_urls = self._output_urls(resolved_prediction)
        local_paths: list[str] = []
        if self.settings.replicate_qwen_edit_download_output:
            for index, output_url in enumerate(output_urls):
                local_paths.append(
                    await self._download_output(
                        output_url,
                        resolved_prediction.get("id"),
                        index,
                    )
                )

        return QwenImageEditResponse(
            tool_name=tool_name,
            model=model.public_id,
            prompt=payload.prompt,
            output_urls=output_urls,
            local_paths=local_paths,
            output_format=payload.output_format,
            prediction_id=resolved_prediction.get("id"),
        )

    async def _create_prediction(
        self,
        model: ReplicateModel,
        payload: QwenImageEditRequest,
    ) -> dict:
        input_payload = {
            "prompt": payload.prompt,
            "image": [
                await self._files_client.prepare_image_url(source)
                for source in payload.image_input
            ],
            "go_fast": payload.go_fast,
            "disable_safety_checker": (
                self.settings.replicate_qwen_edit_force_disable_safety_checker
            ),
        }
        if payload.aspect_ratio is not None:
            input_payload["aspect_ratio"] = payload.aspect_ratio
        if payload.seed is not None:
            input_payload["seed"] = payload.seed
        if payload.output_format is not None:
            input_payload["output_format"] = payload.output_format
        if payload.output_quality is not None:
            input_payload["output_quality"] = payload.output_quality

        response = await self._request(
            "POST",
            self._prediction_url(model),
            headers={"Prefer": f"wait={self.settings.replicate_sync_wait_seconds}"},
            json={"input": input_payload},
        )
        return response.json()

    async def _resolve_prediction(self, prediction: dict) -> dict:
        if self._output_urls(prediction, required=False):
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
            if self._output_urls(prediction, required=False):
                return prediction
            if prediction.get("status") in TERMINAL_STATUSES:
                raise ReplicateError(self._error_message(prediction))

        raise ReplicateError(
            "Replicate image edit prediction timed out before completion."
        )

    async def _download_output(
        self,
        output_url: str,
        prediction_id: str | None,
        index: int,
    ) -> str:
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

        output_dir = Path(self.settings.replicate_qwen_edit_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = self._suffix(output_url, response.headers.get("Content-Type"))
        filename = prediction_id or uuid4().hex
        path = output_dir / f"{filename}-{index}{suffix}"
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
    def _output_urls(prediction: dict, *, required: bool = True) -> list[str]:
        output = prediction.get("output")
        if isinstance(output, list) and output:
            return [str(item) for item in output]
        if isinstance(output, str) and output:
            return [output]
        if required:
            raise ReplicateError("Replicate prediction did not return output URLs.")
        return []

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
        if content_type == "image/webp":
            return ".webp"
        suffix = Path(urlparse(output_url).path).suffix
        return suffix or ".bin"
