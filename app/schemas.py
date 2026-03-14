import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    detail: Literal["auto", "low", "high"] | None = None


class InputTextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str


class InputImagePart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["image_url"]
    image_url: ImageURL


ChatContentPart = Annotated[InputTextPart | InputImagePart, Field(discriminator="type")]
MessageContent = str | list[ChatContentPart]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: "StreamOptions | None" = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    verbosity: Literal["low", "medium", "high"] | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)


class StreamOptions(BaseModel):
    include_usage: bool = False


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: Literal["stop"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard] = Field(default_factory=list)


def content_to_string(content: MessageContent) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if isinstance(part, InputTextPart):
            parts.append(part.text)
        else:
            parts.append(
                json.dumps(
                    part.model_dump(exclude_none=True),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    return "\n".join(parts)
