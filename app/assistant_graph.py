from __future__ import annotations

import json
import logging
import re
from operator import add
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.clients.errors import (
    InputValidationError,
    ReplicateError,
    UserFacingExecutionError,
    classify_replicate_error_message,
)
from app.clients.replicate import ReplicateClient
from app.clients.replicate_images import ReplicateImageClient
from app.clients.replicate_qwen_edit import ReplicateQwenEditClient
from app.config import ReplicateModel, Settings
from app.openwebui_meta import (
    OPENWEBUI_META_REQUEST_KEY,
    is_openwebui_meta_request,
    latest_user_text,
)
from app.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    InputTextPart,
    build_messages_from_request,
)
from app.tool_schemas import (
    ImageGenerationRequest,
    ImageGenerationResponse,
    QwenImageEditRequest,
    QwenImageEditResponse,
)
from app.user_facing_errors import assistant_error_message

logger = logging.getLogger(__name__)


class ResourceRef(BaseModel):
    id: str
    kind: str
    source: str
    label: str
    value: str


class PlannerDecision(BaseModel):
    intent: str
    confidence: float = 0.0
    target_resource_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    ambiguities: list[dict[str, Any]] = Field(default_factory=list)
    missing_params: list[str] = Field(default_factory=list)


class PendingClarification(BaseModel):
    kind: str
    question: str
    route: dict[str, Any]
    original_request: dict[str, Any]
    options: list[dict[str, Any]] = Field(default_factory=list)
    expected_answer_type: str = "single_choice"


class GuardResult(BaseModel):
    decision: str
    route: dict[str, Any] | None = None
    question: str | None = None
    pending_clarification: dict[str, Any] | None = None


class ClarificationResolution(BaseModel):
    selected_resource_values: list[str] = Field(default_factory=list)
    route: dict[str, Any] | None = None


class AssistantState(TypedDict, total=False):
    history: Annotated[list[dict[str, Any]], add]
    current_request: dict[str, Any]
    request_messages: list[dict[str, Any]]
    execution_messages: list[dict[str, Any]]
    pending_clarification: dict[str, Any] | None
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
        state_values = await self._load_state_values(conversation_id)
        existing_history = [
            self._coerce_message(message) for message in state_values.get("history", [])
        ]
        pending_clarification = state_values.get("pending_clarification")
        delta_messages = self._new_history_messages(existing_history, request_messages)
        graph_input: AssistantState = {
            "current_request": payload.model_dump(exclude_none=True),
            "request_messages": self._messages_dump(request_messages),
            "execution_messages": self._messages_dump(
                [*existing_history, *delta_messages]
            ),
            "pending_clarification": pending_clarification,
            "request_base_url": request_base_url,
        }
        if not pending_clarification:
            graph_input["history"] = self._messages_dump(delta_messages)
        try:
            result = await self._graph.ainvoke(
                graph_input,
                config={"configurable": {"thread_id": conversation_id}},
            )
        except UserFacingExecutionError as exc:
            logger.exception(
                "assistant request failed conversation_id=%s user=%s prompt=%r",
                conversation_id,
                payload.user,
                prompt_preview,
            )
            return assistant_error_message(exc)
        except ReplicateError as exc:
            category, retryable = classify_replicate_error_message(str(exc))
            logger.exception(
                "assistant request failed conversation_id=%s user=%s prompt=%r",
                conversation_id,
                payload.user,
                prompt_preview,
            )
            return assistant_error_message(
                UserFacingExecutionError(
                    stage="text_generation",
                    category=category,
                    retryable=retryable,
                    provider="replicate",
                    technical_message=str(exc),
                )
            )
        except Exception:
            logger.exception(
                "assistant request failed conversation_id=%s user=%s prompt=%r",
                conversation_id,
                payload.user,
                prompt_preview,
            )
            return assistant_error_message(RuntimeError("assistant execution failed"))

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
        self._sqlite_cm = aiosqlite.connect(str(sqlite_path))
        conn = await self._sqlite_cm.__aenter__()
        serializer = JsonPlusSerializer()
        self._checkpointer = AsyncSqliteSaver(conn, serde=serializer)
        await self._checkpointer.setup()

        graph = StateGraph(AssistantState)
        graph.add_node("route", self._route_request)
        graph.add_node("clarify", self._run_clarify)
        graph.add_node("text", self._run_text)
        graph.add_node("image", self._run_image)
        graph.add_node("image_uncensored", self._run_image_uncensored)
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._next_step,
            {
                "clarify": "clarify",
                "text": "text",
                "image": "image",
                "image_uncensored": "image_uncensored",
            },
        )
        graph.add_edge("clarify", END)
        graph.add_edge("text", END)
        graph.add_edge("image", END)
        graph.add_edge("image_uncensored", END)
        self._graph = graph.compile(checkpointer=self._checkpointer)

    async def _route_request(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        pending_clarification = state.get("pending_clarification")
        cleared_pending = False
        if pending_clarification:
            cancellation = self._maybe_cancel_clarification(payload)
            if cancellation is not None:
                return {
                    "route": {"action": "clarify"},
                    "response_text": cancellation,
                    "pending_clarification": None,
                    "history": [
                        self._message_dump(
                            ChatMessage(role="assistant", content=cancellation)
                        )
                    ],
                }
            selected_images = self._resolve_pending_clarification(
                pending_clarification,
                payload,
            )
            if selected_images is None:
                if self._looks_like_new_request(payload):
                    cleared_pending = True
                else:
                    return {
                        "route": {"action": "clarify"},
                        "response_text": pending_clarification["question"],
                        "pending_clarification": pending_clarification,
                    }
            else:
                original_payload = ChatCompletionRequest.model_validate(
                    pending_clarification["original_request"]
                )
                route = pending_clarification["route"]
                selected_values = list(selected_images.selected_resource_values)
                if selected_images.route is not None:
                    route = selected_images.route
                if not selected_values:
                    selected_values = self._resolve_route_selected_values(
                        route,
                        original_payload,
                    )
                resolved_payload = original_payload
                if selected_values:
                    resolved_payload = self._select_images_in_payload(
                        original_payload,
                        selected_values,
                    )
                base_execution = [
                    self._coerce_message(message)
                    for message in (
                        state.get("execution_messages") or state.get("history", [])
                    )
                ]
                execution_messages = base_execution
                if selected_values:
                    execution_messages = self._select_images_in_history(
                        base_execution,
                        selected_values,
                    )
                return {
                    "current_request": resolved_payload.model_dump(exclude_none=True),
                    "execution_messages": self._messages_dump(execution_messages),
                    "route": route,
                    "pending_clarification": None,
                }

        current_text = latest_user_text(payload)
        resources = self._collect_resources(payload)
        recent_turns = "\n".join(
            f"{message.role}: {self._message_text(message)}"
            for message in (
                self._coerce_message(message)
                for message in state.get("history", [])[-6:]
            )
        )
        resource_manifest = "\n".join(
            f"{resource.id}: kind={resource.kind}, source={resource.source}, "
            f"label={resource.label}"
            for resource in resources
        )
        router_prompt = (
            "You are a routing agent for a multimodal assistant. "
            "Return JSON only. Decide one intent from: text, image, image_uncensored. "
            "Use text for normal answers. All text generation is handled by gpt-5.4. "
            "Use image for normal image generation/editing. "
            "Use image_uncensored only for uncensored image edits when the user "
            "clearly "
            "asks for uncensored/adult sexual image editing of an existing image. "
            "If the user asks to analyze or describe an image, choose text. "
            "Do not answer the user directly; only route. "
            "If the user request is underspecified, vague, or could reasonably mean "
            "different actions, add an ambiguity item instead of guessing. "
            "When the request references files or media, use target_resource_ids to "
            "point to the matching resources. If there are multiple plausible "
            "resources, add an ambiguity item instead of guessing. "
            "Use ambiguity.kind='intent_selection' for vague requests about what "
            "kind of result to produce. Use ambiguity.kind='resource_selection' "
            "only when multiple actual resources are present. "
            "Do not use placeholder missing_params like 'param_name', "
            "'clarification', or 'details'. Only list a real executable parameter "
            "name such as 'aspect_ratio' or 'output_format'. "
            "For text actions include reasoning_effort, verbosity, "
            "max_completion_tokens, and tool_prompt when a sharper instruction "
            "would improve execution. "
            "Include confidence as a number from 0 to 1. "
            "For image actions include tool_prompt and any useful settings. "
            "JSON schema: "
            '{"intent":"text|image|image_uncensored",'
            '"confidence":0.0,'
            '"target_resource_ids":["resource_id"],'
            '"params":{"tool_prompt":null,"reasoning_effort":null,'
            '"verbosity":null,"max_completion_tokens":null,"aspect_ratio":null,'
            '"resolution":null,"output_format":null,"output_quality":null,'
            '"go_fast":null},'
            '"ambiguities":[{"kind":"resource_selection|intent_selection",'
            '"resource_kind":"image","resource_ids":["resource_id"],'
            '"options":["text","image"]}],'
            '"missing_params":["param_name"]}'
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
                        f"Available resources:\n{resource_manifest or 'none'}"
                    ),
                ),
            ],
        )
        try:
            decision_text = await self._replicate_client.create_reply(
                self._require_model(self.settings.assistant_router_model_id),
                router_request,
            )
        except ReplicateError as exc:
            category, retryable = classify_replicate_error_message(str(exc))
            raise UserFacingExecutionError(
                stage="planner",
                category=category,
                retryable=retryable,
                provider="replicate",
                model=self.settings.assistant_router_model_id,
                technical_message=str(exc),
            ) from exc
        decision = self._parse_decision(decision_text, payload, resources)
        guard_result = self._guard_decision(decision, payload, resources)
        if guard_result.decision == "clarify":
            return {
                "route": {"action": "clarify"},
                "response_text": guard_result.question,
                "pending_clarification": guard_result.pending_clarification,
            }
        route = guard_result.route or {"action": "text"}
        logger.info(
            "assistant route conversation_id=%s action=%s confidence=%s",
            self._conversation_id(payload),
            route.get("action"),
            route.get("confidence"),
        )
        result: dict[str, Any] = {"route": route}
        selected_resources = route.get("selected_resources") or []
        if selected_resources:
            base_execution = [
                self._coerce_message(message)
                for message in (
                    state.get("execution_messages") or state.get("history", [])
                )
            ]
            result["current_request"] = self._select_images_in_payload(
                payload,
                selected_resources,
            ).model_dump(exclude_none=True)
            result["execution_messages"] = self._messages_dump(
                self._select_images_in_history(
                    base_execution,
                    selected_resources,
                )
            )
        if cleared_pending:
            result["pending_clarification"] = None
        return result

    async def _run_text(self, state: AssistantState) -> dict[str, Any]:
        payload = ChatCompletionRequest(**state["current_request"])
        route = state["route"]
        model_id = self.settings.assistant_full_model_id
        history = [
            self._coerce_message(message)
            for message in (state.get("execution_messages") or state.get("history", []))
        ]
        if route.get("tool_prompt"):
            history = self._rewrite_latest_user_message(
                history,
                route["tool_prompt"],
            )
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
        try:
            reply = await self._replicate_client.create_reply(
                self._require_model(model_id),
                text_payload,
            )
        except ReplicateError as exc:
            category, retryable = classify_replicate_error_message(str(exc))
            raise UserFacingExecutionError(
                stage="text_generation",
                category=category,
                retryable=retryable,
                provider="replicate",
                model=model_id,
                technical_message=str(exc),
            ) from exc
        return {
            "history": [
                self._message_dump(ChatMessage(role="assistant", content=reply))
            ],
            "response_text": reply,
        }

    async def _run_clarify(self, state: AssistantState) -> dict[str, Any]:
        return {
            "history": [
                self._message_dump(
                    ChatMessage(role="assistant", content=state["response_text"])
                )
            ],
            "response_text": state["response_text"],
            "pending_clarification": state.get("pending_clarification"),
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
            "history": [
                self._message_dump(ChatMessage(role="assistant", content=response_text))
            ],
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
            "history": [
                self._message_dump(ChatMessage(role="assistant", content=response_text))
            ],
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

    async def _load_state_values(self, conversation_id: str) -> dict[str, Any]:
        snapshot = await self._graph.aget_state(
            {"configurable": {"thread_id": conversation_id}}
        )
        return snapshot.values or {}

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

    def _messages_dump(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        return [self._message_dump(message) for message in messages]

    @staticmethod
    def _log_preview(text: str, limit: int = 160) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."

    def _parse_decision(
        self,
        text: str,
        payload: ChatCompletionRequest,
        resources: list[ResourceRef],
    ) -> PlannerDecision:
        parsed = self._extract_json(text)
        if isinstance(parsed, dict) and parsed.get("intent") in {
            "text",
            "image",
            "image_uncensored",
        }:
            return self._normalize_decision(parsed)
        return self._heuristic_decision(payload, resources)

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

    def _heuristic_decision(
        self,
        payload: ChatCompletionRequest,
        resources: list[ResourceRef],
    ) -> PlannerDecision:
        ambiguities: list[dict[str, Any]] = []
        if len(resources) > 1:
            ambiguities.append(
                {
                    "kind": "resource_selection",
                    "resource_kind": resources[0].kind,
                    "resource_ids": [resource.id for resource in resources],
                }
            )
        return PlannerDecision(
            intent="text",
            confidence=0.0,
            params={
                "reasoning_effort": "medium",
                "verbosity": "medium",
                "max_completion_tokens": 65536,
                "tool_prompt": latest_user_text(payload),
            },
            ambiguities=ambiguities,
        )

    def _normalize_decision(self, decision: dict[str, Any]) -> PlannerDecision:
        params = decision.get("params")
        if not isinstance(params, dict):
            params = {
                "reasoning_effort": decision.get("reasoning_effort"),
                "verbosity": decision.get("verbosity"),
                "max_completion_tokens": decision.get("max_completion_tokens"),
                "tool_prompt": decision.get("tool_prompt"),
                "aspect_ratio": decision.get("aspect_ratio"),
                "resolution": decision.get("resolution"),
                "output_format": decision.get("output_format"),
                "output_quality": decision.get("output_quality"),
                "go_fast": decision.get("go_fast"),
            }
        return PlannerDecision(
            intent=decision["intent"],
            confidence=self._normalize_confidence(decision.get("confidence")),
            target_resource_ids=list(decision.get("target_resource_ids", [])),
            params=params,
            ambiguities=list(decision.get("ambiguities", [])),
            missing_params=list(decision.get("missing_params", [])),
        )

    def _decision_to_route(self, decision: PlannerDecision) -> dict[str, Any]:
        action = decision.intent
        params = decision.params
        if action == "text":
            model_id = self.settings.assistant_full_model_id
            reasoning = params.get("reasoning_effort")
            allowed_reasoning = self.FULL_REASONING
            if reasoning not in allowed_reasoning:
                reasoning = self._default_reasoning(model_id)
            verbosity = params.get("verbosity")
            if verbosity not in self.TEXT_VERBOSITIES:
                verbosity = "medium"
            max_tokens = params.get("max_completion_tokens")
            if not isinstance(max_tokens, int) or max_tokens < 1:
                max_tokens = self._default_max_completion_tokens(model_id)
            max_tokens = max(16, max_tokens)
            return {
                "action": "text",
                "reasoning_effort": reasoning,
                "verbosity": verbosity,
                "max_completion_tokens": max_tokens,
                "tool_prompt": params.get("tool_prompt"),
                "confidence": decision.confidence,
                "selected_resources": decision.target_resource_ids,
            }
        if action == "image":
            aspect_ratio = params.get("aspect_ratio")
            if aspect_ratio not in self.IMAGE_ASPECT_RATIOS:
                aspect_ratio = None
            resolution = params.get("resolution")
            if resolution not in self.IMAGE_RESOLUTIONS:
                resolution = None
            output_format = params.get("output_format")
            if output_format not in self.IMAGE_OUTPUT_FORMATS:
                output_format = None
            return {
                "action": "image",
                "tool_prompt": params.get("tool_prompt"),
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "output_format": output_format,
                "confidence": decision.confidence,
                "selected_resources": decision.target_resource_ids,
            }
        aspect_ratio = params.get("aspect_ratio")
        if aspect_ratio not in self.QWEN_ASPECT_RATIOS:
            aspect_ratio = "match_input_image"
        output_format = params.get("output_format")
        if output_format not in self.QWEN_OUTPUT_FORMATS:
            output_format = "png"
        output_quality = params.get("output_quality")
        if not isinstance(output_quality, int) or not 0 <= output_quality <= 100:
            output_quality = 95
        go_fast = params.get("go_fast")
        return {
            "action": "image_uncensored",
            "tool_prompt": params.get("tool_prompt"),
            "aspect_ratio": aspect_ratio,
            "go_fast": True if go_fast is None else bool(go_fast),
            "output_format": output_format,
            "output_quality": output_quality,
            "confidence": decision.confidence,
            "selected_resources": decision.target_resource_ids,
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

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        return 0.0

    def _guard_decision(
        self,
        decision: PlannerDecision,
        payload: ChatCompletionRequest,
        resources: list[ResourceRef],
    ) -> GuardResult:
        resolved_resources = self._resolve_target_resources(decision, resources)
        ambiguity = self._first_ambiguity(decision, payload, resources)
        if ambiguity is not None:
            if self._should_treat_ambiguity_as_intent(ambiguity, resources):
                pending = self._build_intent_selection_clarification(payload, decision)
                return GuardResult(
                    decision="clarify",
                    question=pending["question"],
                    pending_clarification=pending,
                )
            pending = self._build_resource_selection_clarification(
                payload,
                decision,
                ambiguity,
                resources,
            )
            return GuardResult(
                decision="clarify",
                question=pending["question"],
                pending_clarification=pending,
            )
        if self._has_specific_missing_params(decision):
            pending = self._build_missing_param_clarification(payload, decision)
            return GuardResult(
                decision="clarify",
                question=pending["question"],
                pending_clarification=pending,
            )
        if self._should_clarify_intent(decision, payload, resources):
            pending = self._build_intent_selection_clarification(payload, decision)
            return GuardResult(
                decision="clarify",
                question=pending["question"],
                pending_clarification=pending,
            )
        route = self._decision_to_route(decision)
        route["selected_resources"] = [
            resource.value for resource in resolved_resources
        ]
        return GuardResult(decision="execute", route=route)

    def _build_resource_selection_clarification(
        self,
        payload: ChatCompletionRequest,
        decision: PlannerDecision,
        ambiguity: dict[str, Any],
        resources: list[ResourceRef],
    ) -> dict[str, Any]:
        resource_ids = ambiguity.get("resource_ids") or [
            resource.id for resource in resources
        ]
        options = [
            {
                "label": str(index + 1),
                "resource_id": resource.id,
                "value": resource.value,
            }
            for index, resource in enumerate(resources)
            if resource.id in resource_ids
        ]
        option_labels = [option["label"] for option in options]
        question = (
            "Есть несколько подходящих ресурсов. С каким работать: "
            f"{', '.join(option_labels)}?"
        )
        return {
            "kind": "resource_selection",
            "question": question,
            "route": self._decision_to_route(decision),
            "original_request": payload.model_dump(exclude_none=True),
            "options": options,
            "expected_answer_type": "single_choice",
        }

    def _build_missing_param_clarification(
        self,
        payload: ChatCompletionRequest,
        decision: PlannerDecision,
    ) -> dict[str, Any]:
        missing = decision.missing_params[0]
        return {
            "kind": "missing_param",
            "question": f"Уточни параметр: {missing}.",
            "route": self._decision_to_route(decision),
            "original_request": payload.model_dump(exclude_none=True),
            "options": [],
            "expected_answer_type": "free_text",
            "missing_param": missing,
        }

    def _build_intent_selection_clarification(
        self,
        payload: ChatCompletionRequest,
        decision: PlannerDecision,
    ) -> dict[str, Any]:
        base_params = dict(decision.params)
        options: list[dict[str, Any]] = [
            {
                "label": "1",
                "intent": "text",
                "description": "ответить текстом",
                "route": self._decision_to_route(
                    decision.model_copy(
                        update={
                            "intent": "text",
                            "params": {
                                **base_params,
                                "tool_prompt": base_params.get("tool_prompt")
                                or latest_user_text(payload),
                            },
                        }
                    )
                ),
            },
            {
                "label": "2",
                "intent": "image",
                "description": "сгенерировать или отредактировать обычное изображение",
                "route": self._decision_to_route(
                    decision.model_copy(update={"intent": "image"})
                ),
            },
        ]
        if self._extract_image_inputs(payload):
            options.append(
                {
                    "label": str(len(options) + 1),
                    "intent": "image_uncensored",
                    "description": "сделать uncensored-редактирование изображения",
                    "route": self._decision_to_route(
                        decision.model_copy(update={"intent": "image_uncensored"})
                    ),
                }
            )
        descriptions = [f"{item['label']} - {item['description']}" for item in options]
        return {
            "kind": "intent_selection",
            "question": "Уточни, что именно нужно сделать: " + "; ".join(descriptions),
            "route": self._decision_to_route(decision),
            "original_request": payload.model_dump(exclude_none=True),
            "options": options,
            "expected_answer_type": "single_choice",
        }

    def _resolve_pending_clarification(
        self,
        pending: dict[str, Any],
        payload: ChatCompletionRequest,
    ) -> ClarificationResolution | None:
        kind = pending.get("kind")
        if kind == "missing_param":
            route = dict(pending.get("route") or {})
            missing_param = pending.get("missing_param")
            if not missing_param:
                return None
            route[missing_param] = latest_user_text(payload)
            return ClarificationResolution(route=route)
        if kind == "intent_selection":
            options = pending.get("options", [])
            choice = self._parse_single_choice(latest_user_text(payload), len(options))
            if choice is None:
                return None
            route = options[choice].get("route")
            if not isinstance(route, dict):
                return None
            return ClarificationResolution(route=route)
        if kind != "resource_selection":
            return None
        options = pending.get("options", [])
        if not options:
            return None
        choice = self._parse_single_choice(
            latest_user_text(payload),
            len(options),
        )
        if choice is None:
            return None
        return ClarificationResolution(
            selected_resource_values=[options[choice]["value"]]
        )

    def _first_ambiguity(
        self,
        decision: PlannerDecision,
        payload: ChatCompletionRequest,
        resources: list[ResourceRef],
    ) -> dict[str, Any] | None:
        if decision.ambiguities:
            return decision.ambiguities[0]
        if (
            len(resources) > 1
            and not decision.target_resource_ids
            and self._request_implies_single_resource(payload)
        ):
            return {
                "kind": "resource_selection",
                "resource_kind": resources[0].kind,
                "resource_ids": [resource.id for resource in resources],
            }
        return None

    @staticmethod
    def _should_treat_ambiguity_as_intent(
        ambiguity: dict[str, Any],
        resources: list[ResourceRef],
    ) -> bool:
        kind = ambiguity.get("kind")
        if kind == "intent_selection":
            return True
        if kind != "resource_selection":
            return False
        resource_ids = ambiguity.get("resource_ids") or []
        if not resources:
            return True
        if not resource_ids:
            return True
        return False

    @staticmethod
    def _resolve_target_resources(
        decision: PlannerDecision,
        resources: list[ResourceRef],
    ) -> list[ResourceRef]:
        if not decision.target_resource_ids:
            return resources
        target_ids = set(decision.target_resource_ids)
        selected = [resource for resource in resources if resource.id in target_ids]
        return selected or resources

    def _resolve_route_selected_values(
        self,
        route: dict[str, Any],
        payload: ChatCompletionRequest,
    ) -> list[str]:
        selected = route.get("selected_resources") or []
        if not selected:
            return []
        resources = self._collect_resources(payload)
        by_id = {resource.id: resource.value for resource in resources}
        resolved = [by_id.get(value, value) for value in selected]
        return [value for value in resolved if value]

    def _should_clarify_intent(
        self,
        decision: PlannerDecision,
        payload: ChatCompletionRequest,
        resources: list[ResourceRef],
    ) -> bool:
        text = latest_user_text(payload).strip()
        if decision.missing_params and not self._has_specific_missing_params(decision):
            return True
        if decision.confidence >= 0.25:
            return False
        if resources and decision.intent in {"text", "image", "image_uncensored"}:
            return False
        if len(text.split()) > 5:
            return False
        return True

    @staticmethod
    def _has_specific_missing_params(decision: PlannerDecision) -> bool:
        generic = {
            "param_name",
            "clarification",
            "details",
            "detail",
            "more_details",
            "more_context",
            "context",
        }
        return any(
            isinstance(param, str)
            and param.strip()
            and param.strip().lower() not in generic
            for param in decision.missing_params
        )

    @staticmethod
    def _request_implies_single_resource(payload: ChatCompletionRequest) -> bool:
        text = latest_user_text(payload).lower()
        singular_tokens = (
            "картинк",
            "изображен",
            "фото",
            "файл",
            "документ",
            "таблиц",
            "sheet",
            "spreadsheet",
            "document",
            "image",
            "picture",
            "photo",
            "audio",
            "video",
            "аудио",
            "видео",
            "эту",
            "этот",
            "этом",
            "ней",
            "нём",
        )
        return any(token in text for token in singular_tokens)

    def _collect_resources(self, payload: ChatCompletionRequest) -> list[ResourceRef]:
        image_inputs = self._extract_image_inputs(payload)
        return [
            ResourceRef(
                id=f"image_{index + 1}",
                kind="image",
                source="request",
                label=f"Изображение {index + 1}",
                value=image_url,
            )
            for index, image_url in enumerate(image_inputs)
        ]

    @staticmethod
    def _parse_single_choice(text: str, option_count: int) -> int | None:
        lowered = text.lower()
        match = re.search(r"\b(\d+)\b", lowered)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < option_count:
                return index
        ordinal_map = (
            ("перв", 0),
            ("first", 0),
            ("втор", 1),
            ("second", 1),
            ("трет", 2),
            ("third", 2),
        )
        for token, index in ordinal_map:
            if token in lowered and index < option_count:
                return index
        if "послед" in lowered or "last" in lowered:
            return option_count - 1
        return None

    @staticmethod
    def _maybe_cancel_clarification(payload: ChatCompletionRequest) -> str | None:
        text = latest_user_text(payload).lower().strip()
        if text in {"отмени", "отмена", "cancel", "never mind", "забудь"}:
            return "Отменила уточнение. Сформулируй новый запрос."
        return None

    def _looks_like_new_request(self, payload: ChatCompletionRequest) -> bool:
        if self._extract_image_inputs(payload):
            return True
        text = latest_user_text(payload).strip()
        words = text.split()
        lowered = text.lower()
        if lowered.startswith(
            (
                "нарисуй",
                "сделай",
                "создай",
                "напиши",
                "объясни",
                "найди",
                "расшифруй",
                "проанализируй",
                "draw",
                "generate",
                "create",
                "write",
                "explain",
                "find",
                "transcribe",
                "analyze",
            )
        ):
            return True
        if len(words) >= 6:
            return True
        if text.endswith("?"):
            return True
        return False

    def _select_images_in_payload(
        self,
        payload: ChatCompletionRequest,
        selected_images: list[str],
    ) -> ChatCompletionRequest:
        payload_data = payload.model_dump(exclude_none=True)
        if payload.image_input:
            payload_data["image_input"] = selected_images
            return ChatCompletionRequest(**payload_data)

        selected_set = set(selected_images)
        messages: list[ChatMessage] = []
        for message in payload.messages:
            content = message.content
            if isinstance(content, str):
                messages.append(message)
                continue
            new_parts = []
            for part in content:
                if part.type == "image_url":
                    if part.image_url.url in selected_set:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            if len(new_parts) == 1 and isinstance(new_parts[0], InputTextPart):
                messages.append(
                    ChatMessage(role=message.role, content=new_parts[0].text)
                )
            else:
                messages.append(ChatMessage(role=message.role, content=new_parts))
        payload_data["messages"] = [
            message.model_dump(exclude_none=True) for message in messages
        ]
        return ChatCompletionRequest(**payload_data)

    def _select_images_in_history(
        self,
        history: list[ChatMessage],
        selected_images: list[str],
    ) -> list[ChatMessage]:
        rewritten = list(history)
        for index in range(len(rewritten) - 1, -1, -1):
            message = rewritten[index]
            if message.role != "user":
                continue
            if isinstance(message.content, str):
                continue
            if any(part.type == "image_url" for part in message.content):
                rewritten[index] = self._filter_message_images(message, selected_images)
                break
        return rewritten

    def _filter_message_images(
        self,
        message: ChatMessage,
        selected_images: list[str],
    ) -> ChatMessage:
        if isinstance(message.content, str):
            return message
        selected_set = set(selected_images)
        new_parts = []
        for part in message.content:
            if part.type == "image_url":
                if part.image_url.url in selected_set:
                    new_parts.append(part)
            else:
                new_parts.append(part)
        if len(new_parts) == 1 and isinstance(new_parts[0], InputTextPart):
            return ChatMessage(role=message.role, content=new_parts[0].text)
        return ChatMessage(role=message.role, content=new_parts)

    def _rewrite_latest_user_message(
        self,
        messages: list[ChatMessage],
        prompt: str,
    ) -> list[ChatMessage]:
        rewritten: list[ChatMessage] = []
        replaced = False
        for message in reversed(messages):
            if not replaced and message.role == "user":
                rewritten_message = self._rewrite_message_text(message, prompt)
                rewritten.append(rewritten_message)
                replaced = True
            else:
                rewritten.append(message)
        return list(reversed(rewritten))

    @staticmethod
    def _rewrite_message_text(message: ChatMessage, prompt: str) -> ChatMessage:
        if isinstance(message.content, str):
            return ChatMessage(role=message.role, content=prompt)

        new_parts = []
        replaced = False
        for part in message.content:
            if part.type == "text" and not replaced:
                new_parts.append(InputTextPart(type="text", text=prompt))
                replaced = True
            elif part.type != "text":
                new_parts.append(part)
        if not replaced:
            new_parts.insert(0, InputTextPart(type="text", text=prompt))
        return ChatMessage(role=message.role, content=new_parts)

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
