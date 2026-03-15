import asyncio
import json

import httpx

from app.clients.errors import InputValidationError
from app.clients.replicate import ReplicateClient, ReplicateError
from app.config import AssistantModel, EchoModel, ReplicateModel, Settings
from app.schemas import ChatCompletionRequest, ChatMessage


class FakeFilesClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def prepare_image_url(self, source: str) -> str:
        self.calls.append(source)
        return f"https://api.replicate.com/v1/files/{len(self.calls)}"

    async def aclose(self) -> None:
        return None


def make_settings() -> Settings:
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        app_log_level="INFO",
        api_prefix="/v1",
        health_path="/health",
        public_base_url=None,
        media_path="/media",
        media_root="artifacts",
        echo_empty_response="",
        echo_model=EchoModel(public_id="echo"),
        assistant_model=AssistantModel(public_id="assistant"),
        assistant_router_model_id="gpt-5-nano",
        assistant_full_model_id="gpt-5.4",
        assistant_sqlite_path="data/test-langgraph.sqlite",
        replicate_api_token="token",
        replicate_base_url="https://api.replicate.com/v1",
        replicate_model_map={
            "gpt-5.4": ReplicateModel(
                public_id="gpt-5.4",
                owner="openai",
                name="gpt-5.4",
            )
        },
        replicate_image_tool_id="generate_image",
        replicate_image_model=ReplicateModel(
            public_id="nano-banana-2",
            owner="google",
            name="nano-banana-2",
        ),
        replicate_image_output_dir="artifacts/test-images",
        replicate_image_download_output=True,
        replicate_qwen_edit_tool_id="edit_image_uncensored",
        replicate_qwen_edit_model=ReplicateModel(
            public_id="qwen-image-edit-plus",
            owner="qwen",
            name="qwen-image-edit-plus",
        ),
        replicate_qwen_edit_output_dir="artifacts/test-qwen-edit",
        replicate_qwen_edit_download_output=True,
        replicate_qwen_edit_force_disable_safety_checker=True,
        replicate_local_image_input_roots=("/tmp/test-images",),
        replicate_default_verbosity=None,
        replicate_default_max_completion_tokens=None,
        replicate_sync_wait_seconds=60,
        replicate_poll_interval_seconds=0.0,
        replicate_poll_timeout_seconds=1.0,
        replicate_http_timeout_seconds=5.0,
        replicate_transport_retries=2,
        replicate_transport_retry_backoff_seconds=0.0,
    )


def make_payload(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "gpt-5.4",
        "messages": [ChatMessage(role="user", content="hello")],
    }
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def test_replicate_client_returns_sync_output() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={"status": "succeeded", "output": ["hello", " world"]},
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            reply = await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(messages=[ChatMessage(role="user", content="say hello")]),
            )

        assert reply == "hello world"
        assert calls[0].headers["Authorization"] == "Bearer token"
        assert calls[0].headers["Prefer"] == "wait=60"
        assert calls[0].url.path == "/v1/models/openai/gpt-5.4/predictions"
        body = json.loads(calls[0].content)
        assert body["stream"] is False
        assert body["input"]["messages"] == [{"role": "user", "content": "say hello"}]
        assert "reasoning_effort" not in body["input"]
        assert "verbosity" not in body["input"]
        assert "max_completion_tokens" not in body["input"]

    asyncio.run(run())


def test_replicate_client_raises_on_failed_prediction_with_empty_output() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": "failed", "output": [], "error": "bad"}
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            try:
                await client.create_reply(
                    make_settings().replicate_model_map["gpt-5.4"],
                    make_payload(messages=[ChatMessage(role="user", content="fail")]),
                )
            except ReplicateError as exc:
                assert "bad" in str(exc)
            else:
                raise AssertionError("ReplicateError was not raised")

    asyncio.run(run())


def test_replicate_client_streams_output_events() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "status": "starting",
                    "output": None,
                    "urls": {"stream": "https://api.replicate.com/v1/stream/1"},
                },
            ),
            httpx.Response(
                200,
                content=(
                    b"event: output\n"
                    b"data: hel\n\n"
                    b"event: output\n"
                    b"data: lo\n\n"
                    b"event: done\n"
                    b"data: {}\n\n"
                ),
            ),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            reply_stream = await client.create_reply_stream(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(
                    messages=[ChatMessage(role="user", content="stream hello")]
                ),
            )
            chunks = [chunk async for chunk in reply_stream.iter_output()]

        assert chunks == ["hel", "lo"]

    asyncio.run(run())


def test_replicate_client_stream_preflight_falls_back_to_ready_output() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "succeeded", "output": ["ready"], "urls": {}},
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            reply_stream = await client.create_reply_stream(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(messages=[ChatMessage(role="user", content="ready")]),
            )
            chunks = [chunk async for chunk in reply_stream.iter_output()]

        assert chunks == ["ready"]

    asyncio.run(run())


def test_replicate_client_polls_until_output_ready() -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "status": "processing",
                    "output": None,
                    "urls": {
                        "get": "https://api.replicate.com/v1/predictions/prediction-1"
                    },
                },
            ),
            httpx.Response(200, json={"status": "succeeded", "output": ["done"]}),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            reply = await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(messages=[ChatMessage(role="user", content="say done")]),
            )

        assert reply == "done"

    asyncio.run(run())


def test_replicate_client_raises_on_failed_prediction() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "failed", "error": "boom"})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            try:
                await client.create_reply(
                    make_settings().replicate_model_map["gpt-5.4"],
                    make_payload(messages=[ChatMessage(role="user", content="fail")]),
                )
            except ReplicateError as exc:
                assert "boom" in str(exc)
            else:
                raise AssertionError("ReplicateError was not raised")

    asyncio.run(run())


def test_replicate_client_retries_transport_errors_then_succeeds() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.RemoteProtocolError("upstream disconnected")
        return httpx.Response(
            200,
            json={"status": "succeeded", "output": ["hello after retry"]},
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            reply = await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(messages=[ChatMessage(role="user", content="retry me")]),
            )

        assert reply == "hello after retry"
        assert attempts == 3

    asyncio.run(run())


def test_replicate_client_raises_after_exhausting_transport_retries() -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.RemoteProtocolError("upstream disconnected")

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            try:
                await client.create_reply(
                    make_settings().replicate_model_map["gpt-5.4"],
                    make_payload(
                        messages=[ChatMessage(role="user", content="retry me")]
                    ),
                )
            except ReplicateError as exc:
                assert "Replicate transport error" in str(exc)
            else:
                raise AssertionError("ReplicateError was not raised")

        assert attempts == 3

    asyncio.run(run())


def test_replicate_client_passes_request_options() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "succeeded", "output": ["ok"]})

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(make_settings(), http_client=http_client)
            await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                make_payload(
                    reasoning_effort="high",
                    verbosity="medium",
                    max_completion_tokens=321,
                ),
            )

        body = json.loads(calls[0].content)
        assert body["input"]["reasoning_effort"] == "high"
        assert body["input"]["verbosity"] == "medium"
        assert body["input"]["max_completion_tokens"] == 321

    asyncio.run(run())


def test_replicate_client_uses_native_prompt_fields_when_messages_are_omitted() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "succeeded", "output": ["ok"]})

    async def run() -> None:
        files_client = FakeFilesClient()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(
                make_settings(),
                http_client=http_client,
                files_client=files_client,
            )
            await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                ChatCompletionRequest(
                    model="gpt-5.4",
                    prompt="Describe this image",
                    system_prompt="Reply briefly.",
                    image_input=["https://example.com/cat.png"],
                ),
            )

        body = json.loads(calls[0].content)
        assert files_client.calls == ["https://example.com/cat.png"]
        assert "messages" not in body["input"]
        assert body["input"]["prompt"] == "Describe this image"
        assert body["input"]["system_prompt"] == "Reply briefly."
        assert body["input"]["image_input"] == ["https://api.replicate.com/v1/files/1"]

    asyncio.run(run())


def test_replicate_client_rejects_disallowed_local_image_paths() -> None:
    async def run() -> None:
        client = ReplicateClient(make_settings())
        try:
            await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                ChatCompletionRequest(
                    model="gpt-5.4",
                    reasoning_effort="low",
                    prompt="Describe this image",
                    image_input=["/etc/passwd"],
                ),
            )
        except InputValidationError as exc:
            assert "Allowed roots" in str(exc)
        else:
            raise AssertionError("InputValidationError was not raised")
        finally:
            await client.aclose()

    asyncio.run(run())


def test_replicate_client_uploads_native_image_input_urls() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "succeeded", "output": ["ok"]})

    async def run() -> None:
        files_client = FakeFilesClient()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(
                make_settings(),
                http_client=http_client,
                files_client=files_client,
            )
            await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                ChatCompletionRequest(
                    model="gpt-5.4",
                    prompt="Describe this image",
                    image_input=["https://example.com/cat.png"],
                    reasoning_effort="none",
                ),
            )

        body = json.loads(calls[0].content)
        assert files_client.calls == ["https://example.com/cat.png"]
        assert body["input"]["image_input"] == ["https://api.replicate.com/v1/files/1"]

    asyncio.run(run())


def test_replicate_client_converts_message_images_to_native_input() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "succeeded", "output": ["ok"]})

    async def run() -> None:
        files_client = FakeFilesClient()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url=make_settings().replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateClient(
                make_settings(),
                http_client=http_client,
                files_client=files_client,
            )
            await client.create_reply(
                make_settings().replicate_model_map["gpt-5.4"],
                ChatCompletionRequest(
                    model="gpt-5.4",
                    reasoning_effort="none",
                    messages=[
                        ChatMessage(
                            role="user",
                            content=[
                                {"type": "text", "text": "What animal is shown?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/cat.png"},
                                },
                            ],
                        )
                    ],
                ),
            )

        body = json.loads(calls[0].content)
        assert files_client.calls == ["https://example.com/cat.png"]
        assert "messages" not in body["input"]
        assert body["input"]["prompt"] == "What animal is shown?"
        assert body["input"]["image_input"] == ["https://api.replicate.com/v1/files/1"]

    asyncio.run(run())
