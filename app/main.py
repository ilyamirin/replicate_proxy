from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.backends import AppServices, build_services
from app.clients.errors import InputValidationError
from app.config import (
    AssistantModel,
    EchoModel,
    ReplicateModel,
    Settings,
    load_settings,
    snapshot_settings,
)
from app.model_options import allowed_reasoning_efforts, completion_token_bounds
from app.openwebui_meta import mark_openwebui_meta_request
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelListResponse,
    ResponseMessage,
    build_messages_from_request,
)
from app.tool_schemas import (
    ImageGenerationRequest,
    QwenImageEditRequest,
    ToolCard,
    ToolListResponse,
)
from app.user_facing_errors import (
    api_error_payload,
    tool_error_payload,
    user_facing_message,
)

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = snapshot_settings(settings or load_settings())
    services = build_services(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await services.aclose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.services = services
    app.mount(
        settings.media_path,
        StaticFiles(directory=settings.media_root, check_dir=False),
        name="media",
    )

    @app.get(settings.health_path)
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(f"{settings.api_prefix}/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelCard(id=settings.echo_model.public_id, owned_by="local"),
                ModelCard(id=settings.assistant_model.public_id, owned_by="local"),
            ]
            + [
                ModelCard(id=model.public_id, owned_by=model.owner)
                for model in settings.replicate_model_map.values()
            ]
        )

    @app.get(f"{settings.api_prefix}/tools", response_model=ToolListResponse)
    async def list_tools() -> ToolListResponse:
        return ToolListResponse(
            data=[
                ToolCard(
                    id=settings.replicate_image_tool_id,
                    description=(
                        "Generate or edit images with google/nano-banana-2 via "
                        "Replicate."
                    ),
                    input_schema=ImageGenerationRequest.model_json_schema(),
                ),
                ToolCard(
                    id=settings.replicate_qwen_edit_tool_id,
                    description=(
                        "Edit images with qwen/qwen-image-edit-plus and force "
                        "disable_safety_checker=true."
                    ),
                    input_schema=QwenImageEditRequest.model_json_schema(),
                ),
            ]
        )

    @app.post(f"{settings.api_prefix}/tools/{settings.replicate_image_tool_id}")
    async def generate_image_tool(
        request: Request,
        payload: ImageGenerationRequest,
    ):
        services = request.app.state.services
        try:
            result = await services.replicate_image_client.generate_image(
                services.settings.replicate_image_model,
                payload,
                tool_name=services.settings.replicate_image_tool_id,
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("generate_image tool failed")
            return tool_error_payload(
                services.settings.replicate_image_tool_id,
                exc,
            )
        return result.model_dump()

    @app.post(f"{settings.api_prefix}/tools/{settings.replicate_qwen_edit_tool_id}")
    async def edit_image_tool(
        request: Request,
        payload: QwenImageEditRequest,
    ):
        services = request.app.state.services
        try:
            result = await services.replicate_qwen_edit_client.edit_image(
                services.settings.replicate_qwen_edit_model,
                payload,
                tool_name=services.settings.replicate_qwen_edit_tool_id,
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("edit_image_uncensored tool failed")
            return tool_error_payload(
                services.settings.replicate_qwen_edit_tool_id, exc
            )
        return result.model_dump()

    @app.post(f"{settings.api_prefix}/chat/completions")
    async def create_chat_completion(
        request: Request,
        payload: ChatCompletionRequest,
    ):
        services = request.app.state.services
        model = resolve_model(services.settings, payload.model)
        if model is None:
            raise HTTPException(
                status_code=400, detail=f"Unknown model: {payload.model}"
            )
        mark_openwebui_meta_request(payload)
        if isinstance(model, AssistantModel):
            enrich_assistant_payload_from_headers(request, payload)
        validate_model_payload(services.settings, model, payload)
        if isinstance(model, AssistantModel):
            try:
                services.assistant_graph_service.require_conversation_id(payload)
            except InputValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        if payload.stream:
            try:
                prepared_reply, prepared_stream = await prepare_model_stream(
                    services,
                    model,
                    payload,
                    request_base_url=str(request.base_url),
                )
            except InputValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("stream preflight failed model=%s", payload.model)
                return JSONResponse(status_code=503, content=api_error_payload(exc))
            return StreamingResponse(
                stream_chat_completion(
                    services,
                    payload,
                    model,
                    prepared_reply,
                    prepared_stream,
                    request_base_url=str(request.base_url),
                ),
                media_type="text/event-stream",
            )

        try:
            reply = await create_model_reply(
                services,
                model,
                payload,
                request_base_url=str(request.base_url),
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("chat completion failed model=%s", payload.model)
            return JSONResponse(status_code=503, content=api_error_payload(exc))

        usage = services.token_counter.build_usage(
            build_messages_from_request(payload),
            reply,
        )
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time()),
            model=payload.model,
            choices=[Choice(message=ResponseMessage(content=reply))],
            usage=usage,
        )
        return response.model_dump()

    return app


def resolve_model(
    settings: Settings,
    public_id: str,
) -> AssistantModel | EchoModel | ReplicateModel | None:
    if public_id == settings.echo_model.public_id:
        return settings.echo_model
    if public_id == settings.assistant_model.public_id:
        return settings.assistant_model
    return settings.replicate_model_map.get(public_id)


async def create_model_reply(
    services: AppServices,
    model: AssistantModel | EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
    *,
    request_base_url: str,
) -> str:
    messages = build_messages_from_request(payload)
    if isinstance(model, EchoModel):
        return await services.echo_service.create_reply(messages)
    if isinstance(model, AssistantModel):
        return await services.assistant_graph_service.create_reply(
            payload,
            request_base_url=request_base_url,
        )
    return await services.replicate_client.create_reply(model, payload)


async def stream_chat_completion(
    services: AppServices,
    payload: ChatCompletionRequest,
    model: AssistantModel | EchoModel | ReplicateModel,
    prepared_reply: str | None,
    prepared_stream,
    *,
    request_base_url: str,
) -> AsyncIterator[bytes]:
    chat_id = f"chatcmpl-{uuid4().hex}"
    created = int(time())
    text_parts: list[str] = []
    messages = build_messages_from_request(payload)

    yield sse_chunk(
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )

    try:
        if isinstance(model, EchoModel):
            reply = prepared_reply or await services.echo_service.create_reply(messages)
            if reply:
                text_parts.append(reply)
                yield sse_chunk(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": payload.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": reply},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        elif isinstance(model, AssistantModel):
            reply = (
                prepared_reply
                or await services.assistant_graph_service.create_reply(
                    payload,
                    request_base_url=request_base_url,
                )
            )
            if reply:
                text_parts.append(reply)
                yield sse_chunk(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": payload.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": reply},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        else:
            async for piece in prepared_stream.iter_output():
                if not piece:
                    continue
                text_parts.append(piece)
                yield sse_chunk(
                    {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": payload.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
    except InputValidationError as exc:
        yield sse_chunk(
            {
                "error": {
                    "message": str(exc),
                    "type": "input_validation_error",
                }
            }
        )
        yield b"data: [DONE]\n\n"
        return
    except Exception as exc:
        logger.exception("streaming chat completion failed model=%s", payload.model)
        yield sse_chunk(
            {
                "error": {
                    "message": user_facing_message(exc),
                    "type": "execution_error",
                }
            }
        )
        yield b"data: [DONE]\n\n"
        return

    completion_text = "".join(text_parts)
    usage = services.token_counter.build_usage(messages, completion_text)
    if payload.stream_options and payload.stream_options.include_usage:
        yield sse_chunk(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": payload.model,
                "choices": [],
                "usage": usage.model_dump(),
            }
        )

    yield sse_chunk(
        {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": payload.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield b"data: [DONE]\n\n"


def sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


async def prepare_model_stream(
    services: AppServices,
    model: AssistantModel | EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
    *,
    request_base_url: str,
):
    if isinstance(model, EchoModel):
        reply = await services.echo_service.create_reply(
            build_messages_from_request(payload)
        )
        return reply, None
    if isinstance(model, AssistantModel):
        reply = await services.assistant_graph_service.create_reply(
            payload,
            request_base_url=request_base_url,
        )
        return reply, None
    return None, await services.replicate_client.create_reply_stream(model, payload)


def validate_model_payload(
    settings: Settings,
    model: AssistantModel | EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
) -> None:
    if isinstance(model, (EchoModel, AssistantModel)):
        return

    image_count = count_request_images(payload)
    if model.name == "claude-4.5-sonnet" and image_count > 1:
        raise HTTPException(
            status_code=422,
            detail=f"model {model.public_id} accepts at most one image input",
        )

    bounds = completion_token_bounds(model)
    effective_max_completion_tokens = (
        payload.max_completion_tokens
        if payload.max_completion_tokens is not None
        else settings.replicate_default_max_completion_tokens
    )
    if bounds is not None and effective_max_completion_tokens is not None:
        min_tokens, max_tokens = bounds
        if not min_tokens <= effective_max_completion_tokens <= max_tokens:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"max_completion_tokens for model {model.public_id} must be "
                    f"between {min_tokens} and {max_tokens}"
                ),
            )

    allowed = allowed_reasoning_efforts(model)
    if allowed is None:
        return

    if payload.reasoning_effort is None:
        raise HTTPException(
            status_code=422,
            detail=f"reasoning_effort is required for model {model.public_id}",
        )

    if payload.reasoning_effort not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=422,
            detail=(
                f"reasoning_effort={payload.reasoning_effort!r} is not valid "
                f"for model {model.public_id}; allowed: {allowed_text}"
            ),
        )


def count_request_images(payload: ChatCompletionRequest) -> int:
    if payload.image_input:
        return len(payload.image_input)

    count = 0
    for message in payload.messages:
        content = message.content
        if isinstance(content, str):
            continue
        count += sum(1 for part in content if part.type == "image_url")
    return count


def enrich_assistant_payload_from_headers(
    request: Request,
    payload: ChatCompletionRequest,
) -> None:
    chat_id = request.headers.get("X-OpenWebUI-Chat-Id", "").strip()
    user_id = request.headers.get("X-OpenWebUI-User-Id", "").strip()
    message_id = request.headers.get("X-OpenWebUI-Message-Id", "").strip()
    user_name = request.headers.get("X-OpenWebUI-User-Name", "").strip()
    user_email = request.headers.get("X-OpenWebUI-User-Email", "").strip()
    user_role = request.headers.get("X-OpenWebUI-User-Role", "").strip()

    if chat_id and not str(payload.metadata.get("conversation_id", "")).strip():
        payload.metadata["conversation_id"] = chat_id
    if message_id and not str(payload.metadata.get("openwebui_message_id", "")).strip():
        payload.metadata["openwebui_message_id"] = message_id
    if user_id and not payload.user:
        payload.user = user_id
    if user_name and not str(payload.metadata.get("openwebui_user_name", "")).strip():
        payload.metadata["openwebui_user_name"] = user_name
    if user_email and not str(payload.metadata.get("openwebui_user_email", "")).strip():
        payload.metadata["openwebui_user_email"] = user_email
    if user_role and not str(payload.metadata.get("openwebui_user_role", "")).strip():
        payload.metadata["openwebui_user_role"] = user_role


app = create_app()
