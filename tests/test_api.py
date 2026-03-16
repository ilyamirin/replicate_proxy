import json
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.clients.errors import UserFacingExecutionError
from app.clients.replicate import ReplicateError
from app.config import (
    AssistantModel,
    EchoModel,
    ReplicateModel,
    Settings,
    load_settings,
)
from app.main import create_app
from app.tool_schemas import ImageGenerationResponse, QwenImageEditResponse


def make_settings() -> Settings:
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        app_reload=False,
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
            ),
            "gpt-5-nano": ReplicateModel(
                public_id="gpt-5-nano",
                owner="openai",
                name="gpt-5-nano",
            ),
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


class EmptyReplyStream:
    async def iter_output(self):
        if False:
            yield ""


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


def test_chat_completions_echo_accepts_multimodal_content() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe image"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/cat.png"},
                            },
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"].startswith(
        "describe image"
    )


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
                "id": "assistant",
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
            {
                "id": "gpt-5-nano",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            },
        ],
    }


def test_tools_lists_generate_image_tool() -> None:
    with make_client() as client:
        response = client.get("/v1/tools")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "generate_image"
    assert body["data"][0]["object"] == "tool"
    assert "prompt" in body["data"][0]["input_schema"]["properties"]
    assert body["data"][1]["id"] == "edit_image_uncensored"
    assert "image_input" in body["data"][1]["input_schema"]["properties"]


def test_generate_image_tool_uses_replicate_image_client() -> None:
    with make_client() as client:
        client.app.state.services.replicate_image_client.generate_image = AsyncMock(
            return_value=ImageGenerationResponse(
                tool_name="generate_image",
                model="nano-banana-2",
                prompt="draw a fox",
                output_url="https://example.com/result.png",
                local_path="/tmp/result.png",
                output_format="png",
                prediction_id="pred-1",
            )
        )
        response = client.post(
            "/v1/tools/generate_image",
            json={
                "prompt": "draw a fox",
                "aspect_ratio": "3:4",
                "resolution": "1K",
                "output_format": "png",
            },
        )

    assert response.status_code == 200
    called_model = (
        client.app.state.services.replicate_image_client.generate_image.await_args.args[
            0
        ]
    )
    called_payload = (
        client.app.state.services.replicate_image_client.generate_image.await_args.args[
            1
        ]
    )
    assert called_model.public_id == "nano-banana-2"
    assert called_payload.prompt == "draw a fox"
    assert called_payload.aspect_ratio == "3:4"


def test_generate_image_tool_rejects_conflicting_search_flags() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/tools/generate_image",
            json={
                "prompt": "draw a fox",
                "google_search": True,
                "image_search": True,
            },
        )

    assert response.status_code == 422


def test_generate_image_tool_rejects_local_paths_outside_allowed_roots() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/tools/generate_image",
            json={
                "prompt": "draw a fox",
                "image_input": ["/etc/passwd"],
            },
        )

    assert response.status_code == 422
    assert "Allowed roots" in response.json()["detail"]


def test_edit_image_tool_uses_qwen_client() -> None:
    with make_client() as client:
        client.app.state.services.replicate_qwen_edit_client.edit_image = AsyncMock(
            return_value=QwenImageEditResponse(
                tool_name="edit_image_uncensored",
                model="qwen-image-edit-plus",
                prompt="edit this image",
                output_urls=["https://example.com/result.webp"],
                local_paths=["/tmp/result.webp"],
                output_format="webp",
                prediction_id="pred-2",
            )
        )
        response = client.post(
            "/v1/tools/edit_image_uncensored",
            json={
                "prompt": "edit this image",
                "image_input": ["tests/fixtures/vision-comic.jpeg"],
                "output_format": "webp",
            },
        )

    assert response.status_code == 200
    called_model = (
        client.app.state.services.replicate_qwen_edit_client.edit_image.await_args.args[
            0
        ]
    )
    called_payload = (
        client.app.state.services.replicate_qwen_edit_client.edit_image.await_args.args[
            1
        ]
    )
    assert called_model.public_id == "qwen-image-edit-plus"
    assert called_payload.image_input == ["tests/fixtures/vision-comic.jpeg"]


def test_edit_image_tool_requires_image_input() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/tools/edit_image_uncensored",
            json={
                "prompt": "edit this image",
            },
        )

    assert response.status_code == 422


def test_edit_image_tool_accepts_output_quality_zero() -> None:
    with make_client() as client:
        client.app.state.services.replicate_qwen_edit_client.edit_image = AsyncMock(
            return_value=QwenImageEditResponse(
                tool_name="edit_image_uncensored",
                model="qwen-image-edit-plus",
                prompt="edit this image",
                output_urls=["https://example.com/result.png"],
                local_paths=["/tmp/result.png"],
                output_format="png",
                prediction_id="pred-3",
            )
        )
        response = client.post(
            "/v1/tools/edit_image_uncensored",
            json={
                "prompt": "edit this image",
                "image_input": ["tests/fixtures/vision-comic.jpeg"],
                "output_format": "png",
                "output_quality": 0,
            },
        )

    assert response.status_code == 200


def test_edit_image_tool_rejects_invalid_enum_values() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/tools/edit_image_uncensored",
            json={
                "prompt": "edit this image",
                "image_input": ["tests/fixtures/vision-comic.jpeg"],
                "aspect_ratio": "2:1",
                "output_format": "tiff",
            },
        )

    assert response.status_code == 422


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


def test_assistant_model_uses_graph_service() -> None:
    with make_client() as client:
        client.app.state.services.assistant_graph_service.create_reply = AsyncMock(
            return_value="assistant reply"
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "user": "user-1",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "assistant reply"


def test_assistant_model_reads_openwebui_headers() -> None:
    with make_client() as client:
        client.app.state.services.assistant_graph_service.create_reply = AsyncMock(
            return_value="assistant via headers"
        )
        response = client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "chat-123",
                "X-OpenWebUI-User-Id": "user-456",
                "X-OpenWebUI-User-Name": "Ilya",
            },
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    called_payload = (
        client.app.state.services.assistant_graph_service.create_reply.await_args.args[
            0
        ]
    )
    assert called_payload.metadata["conversation_id"] == "chat-123"
    assert called_payload.metadata["openwebui_user_name"] == "Ilya"
    assert called_payload.user == "user-456"
    assert (
        response.json()["choices"][0]["message"]["content"] == "assistant via headers"
    )


def test_replicate_model_marks_openwebui_meta_request() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            return_value='{"title":"meta"}'
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "reasoning_effort": "low",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "### Task:\nGenerate a concise, 3-5 word title with an "
                            "emoji summarizing the chat history.\n### Output:\n"
                            'JSON format: { "title": "your concise title here" }\n'
                            "### Chat History:\n<chat_history>\nUSER: hello\n"
                            "</chat_history>"
                        ),
                    }
                ],
            },
        )

    assert response.status_code == 200
    called_payload = (
        client.app.state.services.replicate_client.create_reply.await_args.args[1]
    )
    assert called_payload.metadata["openwebui_meta_request"] is True


def test_assistant_model_requires_conversation_id_or_user() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "assistant",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 422


def test_assistant_stream_reads_openwebui_headers() -> None:
    with make_client() as client:
        client.app.state.services.assistant_graph_service.create_reply = AsyncMock(
            return_value="stream ok"
        )
        response = client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "chat-stream-1",
                "X-OpenWebUI-User-Id": "user-stream-1",
            },
            json={
                "model": "assistant",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[1]["choices"][0]["delta"]["content"] == "stream ok"
    called_payload = (
        client.app.state.services.assistant_graph_service.create_reply.await_args.args[
            0
        ]
    )
    assert called_payload.metadata["conversation_id"] == "chat-stream-1"
    assert called_payload.user == "user-stream-1"


def test_assistant_stream_preflight_error_returns_503_error_payload() -> None:
    with make_client() as client:
        client.app.state.services.assistant_graph_service.create_reply = AsyncMock(
            side_effect=ReplicateError("router failed")
        )
        response = client.post(
            "/v1/chat/completions",
            headers={
                "X-OpenWebUI-Chat-Id": "chat-stream-err",
                "X-OpenWebUI-User-Id": "user-stream-err",
            },
            json={
                "model": "assistant",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "execution_error"
    assert "Что-то пошло не так" in response.json()["error"]["message"]


def test_chat_completions_uses_requested_replicate_model() -> None:
    settings = make_settings()
    settings.replicate_model_map["gpt-5-nano-alt"] = ReplicateModel(
        public_id="gpt-5-nano-alt",
        owner="openai",
        name="gpt-5-nano",
    )

    with make_client(settings) as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            return_value="backend reply"
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-nano-alt",
                "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    called_model = (
        client.app.state.services.replicate_client.create_reply.await_args.args[0]
    )
    assert called_model.public_id == "gpt-5-nano-alt"
    assert called_model.owner == "openai"
    assert called_model.name == "gpt-5-nano"


def test_chat_completions_passes_request_model_options() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            return_value="backend reply"
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "reasoning_effort": "high",
                "verbosity": "medium",
                "max_completion_tokens": 321,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    called_payload = (
        client.app.state.services.replicate_client.create_reply.await_args.args[1]
    )
    assert called_payload.reasoning_effort == "high"
    assert called_payload.verbosity == "medium"
    assert called_payload.max_completion_tokens == 321


def test_gpt_5_nano_rejects_invalid_reasoning_effort_for_model() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-nano",
                "reasoning_effort": "xhigh",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 422
    assert "not valid for model gpt-5-nano" in response.json()["detail"]


def test_replicate_models_require_reasoning_effort() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "reasoning_effort is required for model gpt-5.4"


def test_chat_completions_accepts_prompt_system_prompt_and_image_input() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            return_value="backend reply"
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5-nano",
                "reasoning_effort": "low",
                "prompt": "Describe this image",
                "system_prompt": "Reply briefly.",
                "image_input": ["https://example.com/cat.png"],
            },
        )

    assert response.status_code == 200
    called_payload = (
        client.app.state.services.replicate_client.create_reply.await_args.args[1]
    )
    assert called_payload.prompt == "Describe this image"
    assert called_payload.system_prompt == "Reply briefly."
    assert called_payload.image_input == ["https://example.com/cat.png"]


def test_echo_uses_prompt_when_messages_are_omitted() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "echo",
                "prompt": "echo via prompt",
                "system_prompt": "ignored by echo output",
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "echo via prompt"


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


def test_streaming_replicate_preflight_error_returns_503_error_payload() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply_stream = AsyncMock(
            side_effect=ReplicateError("boom")
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "stream": True,
                "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": "fail"}],
            },
        )

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "execution_error"
    assert "Что-то пошло не так" in response.json()["error"]["message"]


def test_generate_image_tool_returns_failed_payload_on_execution_error() -> None:
    with make_client() as client:
        client.app.state.services.replicate_image_client.generate_image = AsyncMock(
            side_effect=UserFacingExecutionError(
                stage="image_generation",
                category="provider_overloaded",
                provider="replicate",
                model="nano-banana-2",
                technical_message="provider failed",
            )
        )
        response = client.post(
            "/v1/tools/generate_image",
            json={"prompt": "draw a fox"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "tool_name": "generate_image",
        "status": "failed",
        "error_type": "execution_error",
        "category": "provider_overloaded",
        "message": (
            "Что-то пошло не так на этапе генерации изображения. "
            "Попробуй повторить запрос."
        ),
        "retryable": True,
        "provider": "replicate",
        "model": "nano-banana-2",
        "stage": "image_generation",
    }


def test_direct_model_returns_structured_error_payload_on_runtime_failure() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply = AsyncMock(
            side_effect=RuntimeError("unexpected failure")
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "reasoning_effort": "medium",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "type": "execution_error",
            "category": "internal_failure",
            "message": "Что-то пошло не так. Попробуй повторить запрос.",
            "retryable": False,
            "provider": None,
            "model": None,
            "stage": None,
        }
    }


def test_streaming_chat_completions_passes_request_model_options() -> None:
    with make_client() as client:
        client.app.state.services.replicate_client.create_reply_stream = AsyncMock(
            return_value=EmptyReplyStream()
        )
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "stream": True,
                "reasoning_effort": "high",
                "verbosity": "medium",
                "max_completion_tokens": 222,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    called_payload = (
        client.app.state.services.replicate_client.create_reply_stream.await_args.args[
            1
        ]
    )
    assert called_payload.reasoning_effort == "high"
    assert called_payload.verbosity == "medium"
    assert called_payload.max_completion_tokens == 222


def test_chat_completions_rejects_invalid_model_options() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "reasoning_effort": "invalid",
                "verbosity": "loud",
                "max_completion_tokens": 0,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 422


def test_chat_completions_requires_some_input() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-5.4"},
        )

    assert response.status_code == 422


def test_chat_completions_reject_local_image_paths_outside_allowed_roots() -> None:
    with make_client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4",
                "reasoning_effort": "low",
                "prompt": "Describe this image.",
                "image_input": ["/etc/passwd"],
            },
        )

    assert response.status_code == 422
    assert "Allowed roots" in response.json()["detail"]


def test_load_settings_reads_env() -> None:
    settings = load_settings()

    assert settings.echo_model.public_id
    assert settings.replicate_model_map
    assert (
        settings.replicate_image_model.public_id == settings.replicate_image_model.name
    )
    assert settings.replicate_local_image_input_roots


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
