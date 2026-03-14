from time import time
from uuid import uuid4

from fastapi import FastAPI

from app.config import get_settings
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ResponseMessage,
    Usage,
)
from app.services import echo_service

settings = get_settings()

app = FastAPI(title=settings.app_name)


def estimate_tokens(text: str) -> int:
    return len(text.split())


@app.get(settings.health_path)
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    f"{settings.api_prefix}/chat/completions",
    response_model=ChatCompletionResponse,
)
async def create_chat_completion(
    payload: ChatCompletionRequest,
) -> ChatCompletionResponse:
    reply = await echo_service.create_reply(payload.messages)
    prompt_text = " ".join(message.content for message in payload.messages)
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(reply)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex}",
        created=int(time()),
        model=payload.model,
        choices=[Choice(message=ResponseMessage(content=reply))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
