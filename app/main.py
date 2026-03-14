from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.backends import AppServices, build_services
from app.clients.replicate import ReplicateError
from app.config import (
    EchoModel,
    ReplicateModel,
    Settings,
    load_settings,
    snapshot_settings,
)
from app.model_options import allowed_reasoning_efforts
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ModelCard,
    ModelListResponse,
    ResponseMessage,
    build_messages_from_request,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = snapshot_settings(settings or load_settings())
    services = build_services(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await services.aclose()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.services = services

    @app.get(settings.health_path)
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(f"{settings.api_prefix}/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[ModelCard(id=settings.echo_model.public_id, owned_by="local")]
            + [
                ModelCard(id=model.public_id, owned_by=model.owner)
                for model in settings.replicate_model_map.values()
            ]
        )

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
        validate_model_payload(services.settings, model, payload)

        if payload.stream:
            try:
                prepared_stream = await prepare_model_stream(services, model, payload)
            except ReplicateError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            return StreamingResponse(
                stream_chat_completion(services, payload, model, prepared_stream),
                media_type="text/event-stream",
            )

        try:
            reply = await create_model_reply(services, model, payload)
        except ReplicateError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
) -> EchoModel | ReplicateModel | None:
    if public_id == settings.echo_model.public_id:
        return settings.echo_model
    return settings.replicate_model_map.get(public_id)


async def create_model_reply(
    services: AppServices,
    model: EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
) -> str:
    messages = build_messages_from_request(payload)
    if isinstance(model, EchoModel):
        return await services.echo_service.create_reply(messages)
    return await services.replicate_client.create_reply(model, payload)


async def stream_chat_completion(
    services: AppServices,
    payload: ChatCompletionRequest,
    model: EchoModel | ReplicateModel,
    prepared_stream,
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
            reply = await services.echo_service.create_reply(messages)
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
    except ReplicateError as exc:
        yield sse_chunk({"error": {"message": str(exc), "type": "replicate_error"}})
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
    model: EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
):
    if isinstance(model, EchoModel):
        return None
    return await services.replicate_client.create_reply_stream(model, payload)


def validate_model_payload(
    _: Settings,
    model: EchoModel | ReplicateModel,
    payload: ChatCompletionRequest,
) -> None:
    if isinstance(model, EchoModel):
        return

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


app = create_app()
