import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.clients.replicate import ReplicateError
from app.config import EchoModel, ReplicateModel, Settings, load_settings
from app.main import create_app


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


def make_client(settings: Settings | None = None) -> TestClient:
    return TestClient(create_app(settings or make_settings()))


def parse_sse(body: str) -> list[dict | str]:
    events: list[dict | str] = []
    for chunk in body.strip().split("\n\n"):
        if not chunk.startswith("data: "):
            continue
        payload = chunk.removeprefix("data: ")
        if payload == "[DONE]":
            events.append(payload)
        else:
            events.append(json.loads(payload))
    return events


def test_healthcheck() -> None:
    with make_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completions_echoes_last_user_message() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ignored"},
                    {"role": "user", "content": "echo this"},
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "echo"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "echo this",
    }
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_chat_completions_returns_empty_string_without_user_message() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [{"role": "system", "content": "No user input here."}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ""


def test_models_lists_available_models() -> None:
    with make_client() as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "echo",
                "object": "model",
                "created": 0,
                "owned_by": "local",
            },
            {
                "id": "gpt-5.4",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
        ],
    }


def test_chat_completions_returns_400_for_unknown_model() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown model: unknown-model"


def test_chat_completions_uses_requested_replicate_model() -> None:
    settings = make_settings()
    settings.replicate_model_map["gpt-5.4-alt"] = ReplicateModel(
        public_id="gpt-5.4-alt",
        owner="openai",
        name="gpt-5.4",
    )

    with make_client(settings) as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            return_value="backend reply"
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4-alt",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    called_model = (
        client.app.state.services.replicate_client.create_reply.await_args.args[0]
    )
    assert called_model.public_id == "gpt-5.4-alt"
    assert called_model.owner == "openai"
    assert called_model.name == "gpt-5.4"


def test_streaming_echo_returns_sse_and_usage() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "echo stream"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[1]["choices"][0]["delta"] == {"content": "echo stream"}
    assert events[2]["usage"]["total_tokens"] > 0
    assert events[3]["choices"][0]["finish_reason"] == "stop"
    assert events[4] == "[DONE]"


def test_streaming_replicate_preflight_error_returns_502() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply_stream = AsyncMock(
            side_effect=ReplicateError("boom")
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "stream": True,
                "messages": [{"role": "user", "content": "fail"}],
            },
        )

    assert response.status_code == 502


def test_load_settings_reads_env() -> None:
    settings = load_settings()

    assert settings.echo_model.public_id
    assert settings.replicate_model_map


def test_app_uses_startup_settings_snapshot() -> None:
    settings = make_settings()
    app = create_app(settings)
    settings.replicate_model_map["late-model"] = ReplicateModel(
        public_id="late-model",
        owner="openai",
        name="gpt-5.4",
    )

    with TestClient(app) as client:
        response = client.get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "late-model" not in ids
