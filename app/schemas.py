import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    stream_options: "StreamOptions | None" = None
    prompt: str | None = None
    system_prompt: str | None = None
    image_input: list[str] = Field(default_factory=list)
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None
    ) = None
    verbosity: Literal["low", "medium", "high"] | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_input_shape(self) -> "ChatCompletionRequest":
        if self.messages or self.prompt or self.system_prompt or self.image_input:
            return self
        raise ValueError(
            "One of messages, prompt, system_prompt, or image_input must be provided."
        )


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


def build_messages_from_request(payload: ChatCompletionRequest) -> list[ChatMessage]:
    if payload.messages:
        return payload.messages

    messages: list[ChatMessage] = []
    if payload.system_prompt:
        messages.append(ChatMessage(role="system", content=payload.system_prompt))

    user_parts: list[ChatContentPart] = []
    if payload.prompt:
        user_parts.append(InputTextPart(type="text", text=payload.prompt))
    user_parts.extend(
        InputImagePart(type="image_url", image_url=ImageURL(url=url))
        for url in payload.image_input
    )
    if user_parts:
        if len(user_parts) == 1 and isinstance(user_parts[0], InputTextPart):
            user_content: MessageContent = user_parts[0].text
        else:
            user_content = user_parts
        messages.append(ChatMessage(role="user", content=user_content))

    return messages
