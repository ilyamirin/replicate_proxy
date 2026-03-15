from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    image_input: list[str] = Field(default_factory=list, max_length=14)
    aspect_ratio: (
        Literal[
            "match_input_image",
            "1:1",
            "3:2",
            "2:3",
            "4:3",
            "3:4",
            "16:9",
            "9:16",
            "1:4",
            "4:1",
            "1:8",
            "8:1",
        ]
        | None
    ) = None
    resolution: Literal["512", "768", "1K", "2K"] | None = None
    google_search: bool = False
    image_search: bool = False
    output_format: Literal["jpg", "png"] | None = None

    @model_validator(mode="after")
    def validate_search_mode(self) -> "ImageGenerationRequest":
        if self.google_search and self.image_search:
            raise ValueError("google_search and image_search cannot both be true.")
        return self


class QwenImageEditRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(min_length=1)
    image_input: list[str] = Field(min_length=1)
    aspect_ratio: (
        Literal[
            "match_input_image",
            "1:1",
            "16:9",
            "9:16",
            "4:3",
            "3:4",
            "3:2",
            "2:3",
        ]
        | None
    ) = None
    go_fast: bool = True
    seed: int | None = None
    output_format: Literal["webp", "jpg", "png"] | None = None
    output_quality: int | None = Field(default=None, ge=0, le=100)


class ToolCard(BaseModel):
    id: str
    object: Literal["tool"] = "tool"
    description: str
    input_schema: dict[str, Any]


class ToolListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ToolCard] = Field(default_factory=list)


class ImageGenerationResponse(BaseModel):
    tool_name: str
    status: Literal["succeeded"] = "succeeded"
    model: str
    prompt: str
    output_url: str
    local_path: str | None = None
    output_format: str | None = None
    prediction_id: str | None = None


class QwenImageEditResponse(BaseModel):
    tool_name: str
    status: Literal["succeeded"] = "succeeded"
    model: str
    prompt: str
    output_urls: list[str] = Field(default_factory=list)
    local_paths: list[str] = Field(default_factory=list)
    output_format: str | None = None
    prediction_id: str | None = None
