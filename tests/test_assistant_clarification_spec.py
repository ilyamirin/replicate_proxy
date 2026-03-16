import asyncio
from pathlib import Path

from app.assistant_graph import AssistantGraphService
from app.config import AssistantModel, EchoModel, ReplicateModel, Settings
from app.schemas import ChatCompletionRequest, ChatMessage
from app.tool_schemas import ImageGenerationResponse


class SpecReplicateClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.calls = 0
        self.payloads = []

    async def create_reply(self, model, payload):
        self.calls += 1
        self.payloads.append(payload)
        return next(self._replies)


class SpecImageClient:
    def __init__(self, local_path: str) -> None:
        self.local_path = local_path
        self.calls = 0
        self.payloads = []

    async def generate_image(self, model, payload, *, tool_name):
        self.calls += 1
        self.payloads.append(payload)
        return ImageGenerationResponse(
            tool_name=tool_name,
            model=model.public_id,
            prompt=payload.prompt,
            output_url="https://replicate.example/out.png",
            local_path=self.local_path,
            output_format="png",
            prediction_id="pred-clarify",
        )


class SpecQwenClient:
    async def edit_image(self, model, payload, *, tool_name):
        raise AssertionError("Qwen client should not be called in clarification specs")


def make_settings(tmp_path: Path) -> Settings:
    media_root = tmp_path / "artifacts"
    media_root.mkdir(exist_ok=True)
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        app_reload=False,
        app_log_level="INFO",
        api_prefix="/v1",
        health_path="/health",
        public_base_url="http://service.test",
        media_path="/media",
        media_root=str(media_root),
        echo_empty_response="",
        echo_model=EchoModel(public_id="echo"),
        assistant_model=AssistantModel(public_id="assistant"),
        assistant_router_model_id="gpt-5-nano",
        assistant_full_model_id="gpt-5.4",
        assistant_sqlite_path=str(tmp_path / "data" / "langgraph.sqlite"),
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
        replicate_image_output_dir=str(media_root / "images"),
        replicate_image_download_output=True,
        replicate_qwen_edit_tool_id="edit_image_uncensored",
        replicate_qwen_edit_model=ReplicateModel(
            public_id="qwen-image-edit-plus",
            owner="qwen",
            name="qwen-image-edit-plus",
        ),
        replicate_qwen_edit_output_dir=str(media_root / "qwen-edit"),
        replicate_qwen_edit_download_output=True,
        replicate_qwen_edit_force_disable_safety_checker=True,
        replicate_local_image_input_roots=(str(tmp_path / "fixtures"),),
        replicate_default_verbosity=None,
        replicate_default_max_completion_tokens=None,
        replicate_sync_wait_seconds=60,
        replicate_poll_interval_seconds=0.0,
        replicate_poll_timeout_seconds=1.0,
        replicate_http_timeout_seconds=5.0,
        replicate_transport_retries=2,
        replicate_transport_retry_backoff_seconds=0.0,
    )


def test_assistant_spec_uses_planner_for_obvious_image_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "artifacts" / "images" / "planned.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"image","confidence":0.96,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Нарисуй акварельную картинку с ослом на '
                'лугу.","aspect_ratio":"3:4","resolution":"1K",'
                '"output_format":"png"},"ambiguities":[],"missing_params":[]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(str(image_path)),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-plan-image"},
                messages=[
                    ChatMessage(
                        role="user",
                        content="Нарисуй акварельную картинку с ослом на лугу.",
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert "generated image" in reply
        assert replicate_client.calls == 1
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_clarifies_ambiguous_single_image_request(
    tmp_path: Path,
) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.71,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":512},"ambiguities":[],"missing_params":[]}'
            ),
            "На изображении текста не видно.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-image-selection"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Что написано на картинке?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert "с каким работать" in reply.lower()
        assert "1" in reply and "2" in reply
        assert replicate_client.calls == 1
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_resumes_after_attachment_clarification(tmp_path: Path) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.69,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":512},"ambiguities":[],"missing_params":[]}'
            ),
            "На второй картинке написано: CAPTAIN BARNABY.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-resume"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Что написано на картинке?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-resume"},
                messages=[ChatMessage(role="user", content="На второй.")],
            ),
            request_base_url="http://service.test/",
        )
        assert "с каким работать" in first_reply.lower()
        assert second_reply == "На второй картинке написано: CAPTAIN BARNABY."
        assert replicate_client.calls == 2
        selected_message = replicate_client.payloads[1].messages[0]
        assert not isinstance(selected_message.content, str)
        selected_urls = [
            part.image_url.url
            for part in selected_message.content
            if part.type == "image_url"
        ]
        assert selected_urls == ["https://example.com/second.png"]
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_cancels_pending_clarification(tmp_path: Path) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.69,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":512},"ambiguities":[],"missing_params":[]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-cancel"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Что написано на картинке?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-cancel"},
                messages=[ChatMessage(role="user", content="отмени")],
            ),
            request_base_url="http://service.test/",
        )
        assert "с каким работать" in first_reply.lower()
        assert second_reply == "Отменила уточнение. Сформулируй новый запрос."
        assert replicate_client.calls == 1
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_replaces_pending_clarification_with_new_request(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "artifacts" / "images" / "replace.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.69,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":512},"ambiguities":[],"missing_params":[]}'
            ),
            (
                '{"intent":"image","confidence":0.95,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Нарисуй синего осла.",'
                '"aspect_ratio":"3:4","resolution":"1K","output_format":"png"},'
                '"ambiguities":[],"missing_params":[]}'
            ),
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(str(image_path)),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-replace"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Что написано на картинке?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-replace"},
                messages=[ChatMessage(role="user", content="Нарисуй синего осла.")],
            ),
            request_base_url="http://service.test/",
        )
        assert "generated image" in second_reply
        assert replicate_client.calls == 2
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_applies_planner_selected_resource_without_clarification(
    tmp_path: Path,
) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.88,'
                '"target_resource_ids":["image_2"],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":512,'
                '"tool_prompt":"Прочитай текст на выбранном изображении."},'
                '"ambiguities":[],"missing_params":[]}'
            ),
            "На второй картинке написано: CAPTAIN BARNABY.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-direct-selection"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": "Что написано на второй картинке?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert reply == "На второй картинке написано: CAPTAIN BARNABY."
        assert replicate_client.calls == 2
        selected_message = replicate_client.payloads[1].messages[0]
        assert not isinstance(selected_message.content, str)
        selected_urls = [
            part.image_url.url
            for part in selected_message.content
            if part.type == "image_url"
        ]
        assert selected_urls == ["https://example.com/second.png"]
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_clarifies_low_confidence_vague_request(tmp_path: Path) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.12,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Сделай красиво"},"ambiguities":[],'
                '"missing_params":[]}'
            ),
            "Сделаю текстовый вариант.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-intent-selection"},
                messages=[ChatMessage(role="user", content="Сделай красиво")],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-intent-selection"},
                messages=[ChatMessage(role="user", content="1")],
            ),
            request_base_url="http://service.test/",
        )
        assert "уточни, что именно нужно сделать" in first_reply.lower()
        assert "1 -" in first_reply
        assert second_reply == "Сделаю текстовый вариант."
        assert replicate_client.calls == 2
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_resumes_after_missing_param_clarification(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "artifacts" / "images" / "ratio.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    image_client = SpecImageClient(str(image_path))
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"image","confidence":0.77,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Нарисуй постер с ослом"},'
                '"ambiguities":[],"missing_params":["aspect_ratio"]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=image_client,
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-missing-param"},
                messages=[ChatMessage(role="user", content="Нарисуй постер с ослом")],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-missing-param"},
                messages=[ChatMessage(role="user", content="3:4")],
            ),
            request_base_url="http://service.test/",
        )
        assert "уточни параметр: aspect_ratio" in first_reply.lower()
        assert "generated image" in second_reply
        assert image_client.calls == 1
        assert image_client.payloads[0].aspect_ratio == "3:4"
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_falls_back_to_intent_clarification_for_bad_planner_ambiguity(
    tmp_path: Path,
) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.11,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Сделай красиво"},'
                '"ambiguities":[{"kind":"resource_selection","resource_kind":"image"}],'
                '"missing_params":[]}'
            ),
            "Сделаю текстовый вариант.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-bad-ambiguity"},
                messages=[ChatMessage(role="user", content="Сделай красиво")],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-bad-ambiguity"},
                messages=[ChatMessage(role="user", content="1")],
            ),
            request_base_url="http://service.test/",
        )
        assert "уточни, что именно нужно сделать" in first_reply.lower()
        assert "1 -" in first_reply
        assert second_reply == "Сделаю текстовый вариант."
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_ignores_generic_missing_param_and_clarifies_resource(
    tmp_path: Path,
) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.61,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Что написано на картинке?"},'
                '"ambiguities":[],"missing_params":["param_name"]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-generic-param-resource"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=[
                            {"type": "text", "text": "Что написано на картинке?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/first.png"},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/second.png"},
                            },
                        ],
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert "с каким работать" in first_reply.lower()
        assert "1" in first_reply and "2" in first_reply
        await service.aclose()

    asyncio.run(run())


def test_assistant_spec_ignores_generic_missing_param_and_clarifies_intent(
    tmp_path: Path,
) -> None:
    replicate_client = SpecReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.52,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Сделай красиво"},'
                '"ambiguities":[],"missing_params":["clarification"]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=SpecImageClient(
            str(tmp_path / "artifacts" / "images" / "unused.png")
        ),
        replicate_qwen_edit_client=SpecQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "clarify-generic-param-intent"},
                messages=[ChatMessage(role="user", content="Сделай красиво")],
            ),
            request_base_url="http://service.test/",
        )
        assert "уточни, что именно нужно сделать" in first_reply.lower()
        assert "1 -" in first_reply
        await service.aclose()

    asyncio.run(run())
