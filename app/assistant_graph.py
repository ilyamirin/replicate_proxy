from __future__ import annotations

import json
import logging
import re
from operator import add
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.clients.errors import InputValidationError
from app.clients.replicate import ReplicateClient
from app.clients.replicate_images import ReplicateImageClient
from app.clients.replicate_qwen_edit import ReplicateQwenEditClient
from app.config import ReplicateModel, Settings
from app.openwebui_meta import (
    OPENWEBUI_META_REQUEST_KEY,
    is_openwebui_meta_request,
    latest_user_text,
)
from app.schemas import ChatCompletionRequest, ChatMessage, build_messages_from_request
from app.tool_schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    QwenImageEditRequest,
    QwenImageEditResponse,
)

logger = logging.getLogger(__name__)


class AssistantState(TypedDict, total=False):
    history: Annotated[list[ChatMessage], add]
    current_request: dict[str, Any]
    request_messages: list[ChatMessage]
    route: dict[str, Any]
    response_text: str
    request_base_url: str


class AssistantGraphService:
    IMAGE_ASPECT_RATIOS = {
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
    }
    IMAGE_OUTPUT_FORMATS = {"png", "jpg"}
    IMAGE_RESOLUTIONS = {"512", "768", "1K", "2K"}
    QWEN_ASPECT_RATIOS = {
        "match_input_image",
        "1:1",
        "16:9",
        "9:16",
        "4:3",
        "3:4",
        "3:2",
        "2:3",
    }
    QWEN_OUTPUT_FORMATS = {"webp", "jpg", "png"}
    TEXT_VERBOSITIES = {"low", "medium", "high"}
    ROUTER_REASONING = {"minimal", "low", "medium", "high"}
    FULL_REASONING = {"none", "low", "medium", "high", "xhigh"}

    def __init__(
        self,
        settings: Settings,
        replicate_client: ReplicateClient,
        replicate_image_client: ReplicateImageClient,
        replicate_qwen_edit_client: ReplicateQwenEditClient,
    ) -> None:
        self.settings = settings
        self._replicate_client = replicate_client
        self._replicate_image_client = replicate_image_client
        self._replicate_qwen_edit_client = replicate_qwen_edit_client
        self._sqlite_cm = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._graph = None

    async def aclose(self) -> None:
        if self._sqlite_cm is not None:
            await self._sqlite_cm.__aexit__(None, None, None)
            self._sqlite_cm = None
            self._checkpointer = None
            self._graph = None

    async def create_reply(
        self,
        payload: ChatCompletionRequest,
        *,
        request_base_url: str,
    ) -> str:
        if payload.metadata.get(
            OPENWEBUI_META_REQUEST_KEY
        ) or is_openwebui_meta_request(payload):
            logger.info(
                "assistant meta request detected prompt=%r",
                self._log_preview(latest_user_text(payload)),
            )
            return await self._run_openwebui_meta_request(payload)

        conversation_id = self.require_conversation_id(payload)
        prompt_preview = self._log_preview(latest_user_text(payload))
        logger.info(
            "assistant request start conversation_id=%s user=%s prompt=%r",
            conversation_id,
            payload.user,
            prompt_preview,
        )
        await self._ensure_graph()
        request_messages = build_messages_from_request(payload)
        existing_history = await self._load_history(conversation_id)
        try:
            result = await self._graph.ainvoke(
                {
                    "history": self._new_history_messages(
                        existing_history,
                        request_messages,
                    ),
                    "current_request": payload.model_dump(exclude_none=True),
                    "request_messages": request_messages,
                    "request_base_url": request_base_url,
                },
                config={"configurable": {"thread_id": conversation_id}},
            )
        except Exception:
            logger.exception(
                "assistant request failed conversation_id=%s user=%s prompt=%r",
                conversation_id,
                payload.user,
                prompt_preview,
            )
            raise

        logger.info(
            "assistant request complete conversation_id=%s response_chars=%s",
            conversation_id,
            len(result["response_text"]),
        )
        return result["response_text"]

    def require_conversation_id(self, payload: ChatCompletionRequest) -> str:
        return self._conversation_id(payload)

    async def _ensure_graph(self) -> None:
        if self._graph is not None:
            return

        sqlite_path = Path(self.settings.assistant_sqlite_path)
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_cm = AsyncSqliteSaver.from_conn_string(str(sqlite_path))
        self._checkpointer = await self._sqlite_cm.__aenter__()
        await self._checkpointer.setup()

        graph = StateGraph(AssistantState)
        graph.add_node("route", self._route_request)
        graph.add_node("text", self._run_text)
        graph.add_node("image", self._run_image)
        graph.add_node("image_uncensored", self._run_image_uncensored)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._next_step,
            {
                "text": "text",
                "image": "image",
                "image_uncensored": "image_uncensored",
            },
        )
        graph.add_edge("text", END)
        graph.add_edge("image", END)
        graph.add_edge("image_uncensored", END)
        self._graph = graph.compile(checkpointer=self._checkpointer)

    async def _route_request(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        forced_route = self._forced_route(payload)
        if forced_route is not None:
            return {"route": forced_route}

        current_text = latest_user_text(payload)
        has_images = bool(self._extract_image_inputs(payload))
        recent_turns = "\n".join(
            f"{message.role}: {self._message_text(message)}"
            for message in state.get("history", [])[-6:]
        )
        router_prompt = (
            "You are a routing agent for a multimodal assistant. "
            "Return JSON only. Decide one action from: text, image, image_uncensored. "
            "Use text for normal answers. All text generation is handled by "
            "text_model='gpt-5.4'. "
            "Use image for normal image generation/editing. "
            "Use image_uncensored only for uncensored image edits when the user "
            "clearly "
            "asks for uncensored/adult sexual image editing of an existing image. "
            "If the user asks to analyze or describe an image, choose text with "
            "text_model='gpt-5.4'. "
            "For text actions include reasoning_effort, verbosity, and "
            "max_completion_tokens. For image actions include tool_prompt and any "
            "useful settings. "
            "JSON schema: "
            '{"action":"text|image|image_uncensored",'
            '"text_model":"gpt-5-nano|gpt-5.4|null",'
            '"reasoning_effort":"none|minimal|low|medium|high|xhigh|null",'
            '"verbosity":"low|medium|high|null",'
            '"max_completion_tokens":null,'
            '"tool_prompt":null,'
            '"aspect_ratio":null,'
            '"resolution":null,'
            '"output_format":null,'
            '"output_quality":null,'
            '"go_fast":null}'
        )
        router_request = ChatCompletionRequest(
            model=self.settings.assistant_router_model_id,
            reasoning_effort="medium",
            verbosity="low",
            messages=[
                ChatMessage(role="system", content=router_prompt),
                ChatMessage(
                    role="user",
                    content=(
                        f"Conversation history:\n{recent_turns}\n\n"
                        f"Current user text:\n{current_text}\n\n"
                        f"Has input images: {has_images}"
                    ),
                ),
            ],
        )
        decision_text = await self._replicate_client.create_reply(
            self._require_model(self.settings.assistant_router_model_id),
            router_request,
        )
        route = self._parse_route(decision_text, payload)
        logger.info(
            "assistant route conversation_id=%s action=%s text_model=%s",
            self._conversation_id(payload),
            route.get("action"),
            route.get("text_model"),
        )
        return {"route": route}

    async def _run_text(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        route = state["route"]
        model_id = self.settings.assistant_full_model_id
        history = state.get("history", [])
        text_payload = ChatCompletionRequest(
            **{
                **payload.model_dump(exclude_none=True),
                "model": model_id,
                "messages": history,
                "reasoning_effort": route.get("reasoning_effort")
                or self._default_reasoning(model_id),
                "verbosity": route.get("verbosity") or "medium",
                "max_completion_tokens": (
                    route.get("max_completion_tokens")
                    or self._default_max_completion_tokens(model_id)
                ),
            }
        )
        reply = await self._replicate_client.create_reply(
            self._require_model(model_id),
            text_payload,
        )
        return {
            "history": [ChatMessage(role="assistant", content=reply)],
            "response_text": reply,
        }

    async def _run_openwebui_meta_request(self, payload: ChatCompletionRequest) -> str:
        meta_payload = ChatCompletionRequest(
            model=self.settings.assistant_router_model_id,
            messages=build_messages_from_request(payload),
            reasoning_effort="low",
            verbosity="low",
            max_completion_tokens=1024,
        )
        return await self._replicate_client.create_reply(
            self._require_model(self.settings.assistant_router_model_id),
            meta_payload,
        )

    async def _run_image(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        route = state["route"]
        result = await self._replicate_image_client.generate_image(
            self.settings.replicate_image_model,
            ImageGenerationRequest(
                prompt=route.get("tool_prompt") or latest_user_text(payload),
                image_input=self._extract_image_inputs(payload),
                aspect_ratio=route.get("aspect_ratio"),
                resolution=route.get("resolution"),
                output_format=route.get("output_format"),
            ),
            tool_name=self.settings.replicate_image_tool_id,
        )
        response_text = self._format_image_response(
            result,
            request_base_url=state["request_base_url"],
            intro="Used `generate_image` to produce the result.",
        )
        return {
            "history": [ChatMessage(role="assistant", content=response_text)],
            "response_text": response_text,
        }

    async def _run_image_uncensored(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        image_input = self._extract_image_inputs(payload)
        if not image_input:
            raise InputValidationError(
                "edit_image_uncensored requires at least one input image."
            )
        route = state["route"]
        result = await self._replicate_qwen_edit_client.edit_image(
            self.settings.replicate_qwen_edit_model,
            QwenImageEditRequest(
                prompt=route.get("tool_prompt") or latest_user_text(payload),
                image_input=image_input,
                aspect_ratio=route.get("aspect_ratio"),
                go_fast=route.get("go_fast", True),
                output_format=route.get("output_format"),
                output_quality=route.get("output_quality"),
            ),
            tool_name=self.settings.replicate_qwen_edit_tool_id,
        )
        response_text = self._format_qwen_response(
            result,
            request_base_url=state["request_base_url"],
        )
        return {
            "history": [ChatMessage(role="assistant", content=response_text)],
            "response_text": response_text,
        }

    def _next_step(self, state: AssistantState) -> str:
        return state["route"]["action"]

    def _conversation_id(self, payload: ChatCompletionRequest) -> str:
        conversation_id = str(payload.metadata.get("conversation_id", "")).strip()
        if conversation_id:
            return conversation_id
        if payload.user:
            return payload.user
        raise InputValidationError(
            "assistant model requires metadata.conversation_id or user "
            "for persisted state."
        )

    def _extract_image_inputs(self, payload: ChatCompletionRequest) -> list[str]:
        if payload.image_input:
            return list(payload.image_input)

        image_inputs: list[str] = []
        for message in payload.messages:
            content = message.content
            if isinstance(content, str):
                continue
            for part in content:
                if part.type == "image_url":
                    image_inputs.append(part.image_url.url)
        return image_inputs

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        parts = [part.text for part in message.content if part.type == "text"]
        return "\n".join(parts).strip()

    async def _load_history(self, conversation_id: str) -> list[ChatMessage]:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": conversation_id}}
        )
        values = snapshot.values or {}
        history = values.get("history", [])
        return [self._coerce_message(message) for message in history]

    def _new_history_messages(
        self,
        existing_history: list[ChatMessage],
        request_messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        overlap = min(len(existing_history), len(request_messages))
        existing_dump = [self._message_dump(message) for message in existing_history]
        request_dump = [self._message_dump(message) for message in request_messages]
        for size in range(overlap, -1, -1):
            if size == 0 or existing_dump[-size:] == request_dump[:size]:
                return request_messages[size:]
        return request_messages

    @staticmethod
    def _coerce_message(message: ChatMessage | dict[str, Any]) -> ChatMessage:
        if isinstance(message, ChatMessage):
            return message
        return ChatMessage.model_validate(message)

    @staticmethod
    def _message_dump(message: ChatMessage) -> dict[str, Any]:
        return message.model_dump(exclude_none=True)

    @staticmethod
    def _log_preview(text: str, limit: int = 160) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def _parse_route(
        self,
        text: str,
        payload: ChatCompletionRequest,
    ) -> dict[str, Any]:
        parsed = self._extract_json(text)
        if isinstance(parsed, dict) and parsed.get("action") in {
            "text",
            "image",
            "image_uncensored",
        }:
            return self._normalize_route(parsed)
        return self._heuristic_route(payload)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _heuristic_route(self, payload: ChatCompletionRequest) -> dict[str, Any]:
        forced_route = self._forced_route(payload)
        if forced_route is not None:
            return forced_route
        return {
            "action": "text",
            "text_model": self.settings.assistant_full_model_id,
            "reasoning_effort": "medium",
            "verbosity": "medium",
            "max_completion_tokens": 65536,
        }

    def _forced_route(self, payload: ChatCompletionRequest) -> dict[str, Any] | None:
        text = latest_user_text(payload).lower()
        has_images = bool(self._extract_image_inputs(payload))
        if has_images and any(
            token in text
            for token in (
                "что изображ",
                "опиши",
                "describe",
                "what is",
                "what's in",
                "what is in the image",
            )
        ):
            return {
                "action": "text",
                "text_model": self.settings.assistant_full_model_id,
                "reasoning_effort": "medium",
                "verbosity": "medium",
                "max_completion_tokens": 65536,
            }
        if any(
            token in text
            for token in (
                "нарисуй",
                "сгенерируй картинку",
                "generate an image",
                "draw",
                "illustrate",
            )
        ):
            return {
                "action": "image",
                "tool_prompt": latest_user_text(payload),
                "aspect_ratio": "3:4",
                "resolution": "1K",
                "output_format": "png",
            }
        if has_images and any(
            token in text
            for token in (
                "без цензур",
                "голую",
                "обнаж",
                "nude",
                "uncensored",
                "sexual",
            )
        ):
            return {
                "action": "image_uncensored",
                "tool_prompt": latest_user_text(payload),
                "aspect_ratio": "match_input_image",
                "go_fast": True,
                "output_format": "png",
                "output_quality": 95,
            }
        return None

    def _normalize_route(self, route: dict[str, Any]) -> dict[str, Any]:
        action = route["action"]
        if action == "text":
            model_id = self.settings.assistant_full_model_id
            reasoning = route.get("reasoning_effort")
            allowed_reasoning = self.FULL_REASONING
            if reasoning not in allowed_reasoning:
                reasoning = self._default_reasoning(model_id)
            verbosity = route.get("verbosity")
            if verbosity not in self.TEXT_VERBOSITIES:
                verbosity = "medium"
            max_tokens = route.get("max_completion_tokens")
            if not isinstance(max_tokens, int) or max_tokens < 1:
                max_tokens = self._default_max_completion_tokens(model_id)
            return {
                "action": "text",
                "text_model": model_id,
                "reasoning_effort": reasoning,
                "verbosity": verbosity,
                "max_completion_tokens": max_tokens,
            }
        if action == "image":
            aspect_ratio = route.get("aspect_ratio")
            if aspect_ratio not in self.IMAGE_ASPECT_RATIOS:
                aspect_ratio = None
            resolution = route.get("resolution")
            if resolution not in self.IMAGE_RESOLUTIONS:
                resolution = None
            output_format = route.get("output_format")
            if output_format not in self.IMAGE_OUTPUT_FORMATS:
                output_format = None
            return {
                "action": "image",
                "tool_prompt": route.get("tool_prompt"),
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "output_format": output_format,
            }
        aspect_ratio = route.get("aspect_ratio")
        if aspect_ratio not in self.QWEN_ASPECT_RATIOS:
            aspect_ratio = "match_input_image"
        output_format = route.get("output_format")
        if output_format not in self.QWEN_OUTPUT_FORMATS:
            output_format = "png"
        output_quality = route.get("output_quality")
        if not isinstance(output_quality, int) or not 0 <= output_quality <= 100:
            output_quality = 95
        go_fast = route.get("go_fast")
        return {
            "action": "image_uncensored",
            "tool_prompt": route.get("tool_prompt"),
            "aspect_ratio": aspect_ratio,
            "go_fast": True if go_fast is None else bool(go_fast),
            "output_format": output_format,
            "output_quality": output_quality,
        }

    def _require_model(self, model_id: str) -> ReplicateModel:
        model = self.settings.replicate_model_map.get(model_id)
        if model is None:
            raise InputValidationError(
                f"Assistant model routing references unknown model: {model_id}"
            )
        return model

    def _default_reasoning(self, model_id: str) -> str:
        return "medium"

    def _default_max_completion_tokens(self, model_id: str) -> int:
        if model_id == self.settings.assistant_full_model_id:
            return 65536
        return 400

    def _format_image_response(
        self,
        result: ImageGenerationResponse,
        *,
        request_base_url: str,
        intro: str | None = None,
    ) -> str:
        lines: list[str] = []
        if intro:
            lines.append(intro)
            lines.append("")
        image_url = self._artifact_url(
            result.local_path,
            request_base_url=request_base_url,
            fallback=result.output_url,
        )
        lines.append(f"![generated image]({image_url})")
        lines.append("")
        lines.append(f"[download image]({image_url})")
        return "\n".join(lines)

    def _format_qwen_response(
        self,
        result: QwenImageEditResponse,
        *,
        request_base_url: str,
    ) -> str:
        lines = ["Used `edit_image_uncensored` to edit the image.", ""]
        for index, output_url in enumerate(result.output_urls):
            local_path = (
                result.local_paths[index] if index < len(result.local_paths) else None
            )
            public_url = self._artifact_url(
                local_path,
                request_base_url=request_base_url,
                fallback=output_url,
            )
            lines.append(f"Result {index + 1}:")
            lines.append(f"![edited image {index + 1}]({public_url})")
            lines.append(f"[download image {index + 1}]({public_url})")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _artifact_url(
        self,
        local_path: str | None,
        *,
        request_base_url: str,
        fallback: str,
    ) -> str:
        if not local_path:
            return fallback
        media_root = Path(self.settings.media_root).resolve()
        path = Path(local_path).resolve()
        try:
            relative = path.relative_to(media_root)
        except ValueError:
            return fallback
        base_url = (self.settings.public_base_url or request_base_url).rstrip("/")
        encoded_path = "/".join(quote(part) for part in relative.parts)
        return f"{base_url}{self.settings.media_path}/{encoded_path}"
