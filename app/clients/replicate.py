from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import monotonic

import httpx

from app.clients.errors import ReplicateError
from app.clients.replicate_files import ReplicateFilesClient
from app.config import ReplicateModel, Settings
from app.schemas import ChatCompletionRequest

TERMINAL_STATUSES = {"failed", "canceled", "aborted", "succeeded"}


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
        files_client: ReplicateFilesClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._owns_files_client = files_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.replicate_base_url,
            timeout=settings.replicate_http_timeout_seconds,
        )
        self._files_client = files_client or ReplicateFilesClient(settings)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http_client.aclose()
        if self._owns_files_client:
            await self._files_client.aclose()

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
        input_payload = await self._build_input(payload)
        response = await self._request(
            "POST",
            self._prediction_url(model),
            headers={"Prefer": f"wait={self.settings.replicate_sync_wait_seconds}"},
            json={
                "stream": stream,
                "input": input_payload,
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

    async def _build_input(self, payload: ChatCompletionRequest) -> dict:
        input_payload: dict = {}
        if payload.messages:
            if self._messages_include_images(payload):
                input_payload.update(await self._build_native_vision_input(payload))
            else:
                input_payload["messages"] = await self._prepare_messages(payload)
        else:
            if payload.prompt is not None:
                input_payload["prompt"] = payload.prompt
            if payload.system_prompt is not None:
                input_payload["system_prompt"] = payload.system_prompt
            if payload.image_input:
                input_payload["image_input"] = [
                    await self._files_client.prepare_image_url(image_url)
                    for image_url in payload.image_input
                ]
        reasoning_effort = self._pick_option(
            payload.reasoning_effort,
            None,
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

    async def _build_native_vision_input(self, payload: ChatCompletionRequest) -> dict:
        system_parts: list[str] = []
        prompt_lines: list[str] = []
        image_input: list[str] = []

        for message in payload.messages:
            text_parts: list[str] = []
            content = message.content
            if isinstance(content, str):
                text_parts.append(content)
            else:
                for part in content:
                    if part.type == "text":
                        text_parts.append(part.text)
                    elif part.type == "image_url":
                        uploaded_url = await self._files_client.prepare_image_url(
                            part.image_url.url
                        )
                        image_input.append(uploaded_url)

            text = "\n".join(text_parts).strip()
            if not text:
                continue
            if message.role == "system":
                system_parts.append(text)
            elif message.role == "user":
                prompt_lines.append(text)
            else:
                prompt_lines.append(f"{message.role}: {text}")

        input_payload: dict = {}
        if system_parts:
            input_payload["system_prompt"] = "\n\n".join(system_parts)
        if prompt_lines:
            input_payload["prompt"] = "\n\n".join(prompt_lines)
        if image_input:
            input_payload["image_input"] = image_input
        return input_payload

    async def _prepare_messages(self, payload: ChatCompletionRequest) -> list[dict]:
        prepared_messages: list[dict] = []
        for message in payload.messages:
            dump = message.model_dump(exclude_none=True)
            content = dump.get("content")
            if isinstance(content, list):
                normalized_parts: list[dict] = []
                for part in content:
                    if part.get("type") == "image_url":
                        image_url = part["image_url"]["url"]
                        part = {
                            **part,
                            "image_url": {
                                **part["image_url"],
                                "url": await self._files_client.prepare_image_url(
                                    image_url
                                ),
                            },
                        }
                    normalized_parts.append(part)
                dump["content"] = normalized_parts
            prepared_messages.append(dump)
        return prepared_messages

    @staticmethod
    def _messages_include_images(payload: ChatCompletionRequest) -> bool:
        for message in payload.messages:
            content = message.content
            if isinstance(content, str):
                continue
            if any(part.type == "image_url" for part in content):
                return True
        return False

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
            if not output:
                return None
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
