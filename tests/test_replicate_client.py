import asyncio
import json

import httpx

from app.clients.replicate import ReplicateClient, ReplicateError
from app.config import EchoModel, ReplicateModel, Settings
from app.schemas import ChatMessage


def make_settings() -> Settings:
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        api_prefix="/v1",
        health_path="/health",
        echo_empty_response="",
        echo_model=EchoModel(public_id="echo"),
        replicate_api_token="token",
        replicate_base_url="https://api.replicate.com/v1",
        replicate_model_map={
            "gpt-5.4": ReplicateModel(
                public_id="gpt-5.4",
                owner="openai",
                name="gpt-5.4",
            )
        },
        replicate_reasoning_effort="none",
        replicate_verbosity="low",
        replicate_max_completion_tokens=4096,
        replicate_sync_wait_seconds=60,
        replicate_poll_interval_seconds=0.0,
        replicate_poll_timeout_seconds=1.0,
        replicate_http_timeout_seconds=5.0,
    )


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
                [ChatMessage(role="user", content="say hello")],
            )

        assert reply == "hello world"
        assert calls[0].headers["Authorization"] == "Bearer token"
        assert calls[0].headers["Prefer"] == "wait=60"
        assert calls[0].url.path == "/v1/models/openai/gpt-5.4/predictions"
        body = json.loads(calls[0].content)
        assert body["stream"] is False
        assert body["input"]["messages"] == [{"role": "user", "content": "say hello"}]

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
                [ChatMessage(role="user", content="stream hello")],
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
                [ChatMessage(role="user", content="ready")],
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
                [ChatMessage(role="user", content="say done")],
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
                    [ChatMessage(role="user", content="fail")],
                )
            except ReplicateError as exc:
                assert "boom" in str(exc)
            else:
                raise AssertionError("ReplicateError was not raised")

    asyncio.run(run())
