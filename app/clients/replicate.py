from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import monotonic

import httpx

from app.config import ReplicateModel, Settings
from app.schemas import ChatCompletionRequest

TERMINAL_STATUSES = {"failed", "canceled", "aborted", "succeeded"}


class ReplicateError(RuntimeError):
    """Replicate API request failed."""


@dataclass
class ReplicateReplyStream:
    prediction: dict
    events: AsyncIterator[tuple[str, str]] | None = None

    async def iter_output(self) -> AsyncIterator[str]:
        if self.events is None:
            output = ReplicateClient._coerce_output(self.prediction)
            if output:
                yield output
            return

        emitted = False
        async for event, data in self.events:
            if event == "output":
                emitted = True
                yield data
            elif event == "error":
                raise ReplicateError(ReplicateClient._coerce_error(data))
            elif event == "done":
                return

        if not emitted:
            output = ReplicateClient._coerce_output(self.prediction)
            if output:
                yield output


class ReplicateClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.replicate_base_url,
            timeout=settings.replicate_http_timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()

    async def create_reply(
        self,
        model: ReplicateModel,
        payload: ChatCompletionRequest,
    ) -> str:
        prediction = await self._create_prediction(model, payload, stream=False)
        return await self._resolve_output(prediction)

    async def create_reply_stream(
        self,
        model: ReplicateModel,
        payload: ChatCompletionRequest,
    ) -> ReplicateReplyStream:
        prediction = await self._create_prediction(model, payload, stream=True)
        stream_url = prediction.get("urls", {}).get("stream")
        if not stream_url:
            resolved_prediction = prediction
            if self._coerce_output(prediction) is None:
                output = await self._resolve_output(prediction)
                resolved_prediction = {**prediction, "output": output}
            return ReplicateReplyStream(prediction=resolved_prediction)

        return ReplicateReplyStream(
            prediction=prediction,
            events=self._iter_sse(stream_url),
        )

    async def _create_prediction(
        self,
        model: ReplicateModel,
        payload: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict:
        response = await self._request(
            "POST",
            self._prediction_url(model),
            headers={"Prefer": f"wait={self.settings.replicate_sync_wait_seconds}"},
            json={
                "stream": stream,
                "input": self._build_input(payload),
            },
        )
        return response.json()

    async def _resolve_output(self, prediction: dict) -> str:
        output = self._coerce_output(prediction)
        if output is not None:
            return output

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
            output = self._coerce_output(prediction)
            if output is not None:
                return output
            if prediction.get("status") in TERMINAL_STATUSES:
                raise ReplicateError(self._error_message(prediction))

        raise ReplicateError("Replicate prediction timed out before completion.")

    async def _iter_sse(self, url: str) -> AsyncIterator[tuple[str, str]]:
        async with self._http_client.stream(
            "GET",
            url,
            headers=self._headers({"Accept": "text/event-stream"}),
        ) as stream:
            if stream.is_error:
                stream_body = await stream.aread()
                raise ReplicateError(
                    f"Replicate stream error {stream.status_code}: {stream_body}"
                )

            event = "message"
            data_lines: list[str] = []
            async for line in stream.aiter_lines():
                if not line:
                    if data_lines:
                        yield event, "\n".join(data_lines)
                    event = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event = line.removeprefix("event:").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.removeprefix("data:").lstrip())

            if data_lines:
                yield event, "\n".join(data_lines)

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        if not self.settings.replicate_api_token:
            raise ReplicateError("REPLICATE_API_TOKEN is not set.")

        response = await self._http_client.request(
            method,
            url,
            headers=self._headers(kwargs.pop("headers", None)),
            **kwargs,
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

    def _build_input(self, payload: ChatCompletionRequest) -> dict:
        input_payload = {
            "messages": [
                message.model_dump(exclude_none=True) for message in payload.messages
            ]
        }
        reasoning_effort = self._pick_option(
            payload.reasoning_effort,
            self.settings.replicate_default_reasoning_effort,
        )
        verbosity = self._pick_option(
            payload.verbosity,
            self.settings.replicate_default_verbosity,
        )
        max_completion_tokens = self._pick_option(
            payload.max_completion_tokens,
            self.settings.replicate_default_max_completion_tokens,
        )
        if reasoning_effort is not None:
            input_payload["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            input_payload["verbosity"] = verbosity
        if max_completion_tokens is not None:
            input_payload["max_completion_tokens"] = max_completion_tokens
        return input_payload

    @staticmethod
    def _prediction_url(model: ReplicateModel) -> str:
        return f"/models/{model.owner}/{model.name}/predictions"

    @staticmethod
    def _pick_option(request_value, default_value):
        return request_value if request_value is not None else default_value

    @staticmethod
    def _coerce_output(prediction: dict) -> str | None:
        output = prediction.get("output")
        if output is None:
            return None
        if isinstance(output, list):
            return "".join(str(item) for item in output)
        return str(output)

    @staticmethod
    def _error_message(prediction: dict) -> str:
        error = prediction.get("error")
        if error:
            return f"Replicate prediction failed: {error}"
        status = prediction.get("status", "unknown")
        return f"Replicate prediction ended with status: {status}"

    @staticmethod
    def _coerce_error(data: str) -> str:
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return data
        if isinstance(payload, dict) and payload.get("error"):
            return str(payload["error"])
        return data
