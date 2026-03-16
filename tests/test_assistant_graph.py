import asyncio
from pathlib import Path

from app.assistant_graph import AssistantGraphService
from app.clients.errors import InputValidationError
from app.config import AssistantModel, EchoModel, ReplicateModel, Settings
from app.schemas import ChatCompletionRequest, ChatMessage
from app.tool_schemas import ImageGenerationResponse


class FakeReplicateClient:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.calls = 0
        self.payloads = []

    async def create_reply(self, model, payload):
        self.calls += 1
        self.payloads.append(payload)
        return next(self._replies)


class MemoryAwareFakeReplicateClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create_reply(self, model, payload):
        self.calls += 1
        first_message = payload.messages[0]
        if (
            first_message.role == "system"
            and isinstance(first_message.content, str)
            and "routing agent" in first_message.content
        ):
            return (
                '{"intent":"text","confidence":0.8,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"low","verbosity":"medium",'
                '"max_completion_tokens":100},"ambiguities":[],"missing_params":[]}'
            )

        history = "\n".join(
            (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            )
            for message in payload.messages
        )
        if "Как меня зовут" in history and "меня зовут Илья" in history:
            return "Тебя зовут Илья, и ты любишь груши."
        return "Понял: тебя зовут Илья и ты любишь груши."


class MetaAwareFakeReplicateClient:
    def __init__(self) -> None:
        self.calls = 0
        self.payloads = []

    async def create_reply(self, model, payload):
        self.calls += 1
        self.payloads.append(payload)
        first_message = payload.messages[0]
        if (
            first_message.role == "system"
            and isinstance(first_message.content, str)
            and "routing agent" in first_message.content
        ):
            return (
                '{"intent":"text","confidence":0.8,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"medium",'
                '"max_completion_tokens":400},"ambiguities":[],"missing_params":[]}'
            )

        last_text = payload.messages[-1].content
        if isinstance(last_text, str) and last_text.startswith("### Task:"):
            return '{"title":"🤖 Тестовый чат"}'

        history = "\n".join(
            message.content
            if isinstance(message.content, str)
            else str(message.content)
            for message in payload.messages
        )
        assert "### Task:" not in history
        return "Обычный ответ без загрязнения memory."


class FakeImageClient:
    def __init__(self, local_path: str) -> None:
        self.local_path = local_path

    async def generate_image(self, model, payload, *, tool_name):
        return ImageGenerationResponse(
            tool_name=tool_name,
            model=model.public_id,
            prompt=payload.prompt,
            output_url="https://replicate.example/out.png",
            local_path=self.local_path,
            output_format="png",
            prediction_id="pred-1",
        )


class FakeQwenClient:
    async def edit_image(self, model, payload, *, tool_name):
        raise AssertionError("Qwen client should not be called in this test")


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


def test_assistant_graph_requires_conversation_id_or_user(tmp_path: Path) -> None:
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=FakeReplicateClient([]),
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        try:
            await service.create_reply(
                ChatCompletionRequest(
                    model="assistant",
                    messages=[ChatMessage(role="user", content="hello")],
                ),
                request_base_url="http://service.test/",
            )
        except InputValidationError as exc:
            assert "conversation_id" in str(exc)
        else:
            raise AssertionError("InputValidationError was not raised")

    asyncio.run(run())


def test_assistant_graph_returns_markdown_with_local_media_url(tmp_path: Path) -> None:
    image_path = tmp_path / "artifacts" / "images" / "out.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=FakeReplicateClient(
            [
                (
                    '{"intent":"image","confidence":0.95,"target_resource_ids":[],'
                    '"params":{"tool_prompt":"draw a fox","aspect_ratio":"3:4",'
                    '"resolution":"1K","output_format":"png"},'
                    '"ambiguities":[],"missing_params":[]}'
                )
            ]
        ),
        replicate_image_client=FakeImageClient(str(image_path)),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                user="user-1",
                messages=[ChatMessage(role="user", content="draw a fox")],
            ),
            request_base_url="http://service.test/",
        )
        assert "![generated image](http://service.test/media/images/out.png)" in reply
        assert "[download image](http://service.test/media/images/out.png)" in reply
        assert (tmp_path / "data" / "langgraph.sqlite").exists()
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_uses_planner_for_draw_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "artifacts" / "images" / "planned.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png")
    replicate_client = FakeReplicateClient(
        [
            (
                '{"intent":"image","confidence":0.95,"target_resource_ids":[],'
                '"params":{"tool_prompt":"Draw a donkey in a meadow",'
                '"aspect_ratio":"3:4","resolution":"1K","output_format":"png"},'
                '"ambiguities":[],"missing_params":[]}'
            )
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=FakeImageClient(str(image_path)),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                user="user-2",
                messages=[
                    ChatMessage(role="user", content="Draw a donkey in a meadow")
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert "generated image" in reply
        assert replicate_client.calls == 1
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_uses_persisted_history_for_text_replies(
    tmp_path: Path,
) -> None:
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=MemoryAwareFakeReplicateClient(),
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        first_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "memory-test"},
                messages=[
                    ChatMessage(
                        role="user",
                        content="Запомни: меня зовут Илья и я люблю груши.",
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        second_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "memory-test"},
                messages=[
                    ChatMessage(
                        role="user",
                        content="Как меня зовут и что я люблю?",
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )

        assert "Илья" in first_reply
        assert "Илья" in second_reply
        assert "груши" in second_reply
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_persists_plain_dict_history(tmp_path: Path) -> None:
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=FakeReplicateClient(
            [
                (
                    '{"intent":"text","confidence":0.8,"target_resource_ids":[],'
                    '"params":{"reasoning_effort":"medium","verbosity":"medium",'
                    '"max_completion_tokens":64},"ambiguities":[],"missing_params":[]}'
                ),
                "Ответ.",
            ]
        ),
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "dict-history-test"},
                messages=[ChatMessage(role="user", content="Привет")],
            ),
            request_base_url="http://service.test/",
        )
        state = await service._load_state_values("dict-history-test")
        assert state["history"]
        assert isinstance(state["history"][0], dict)
        assert state["history"][0]["role"] == "user"
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_uses_unbounded_planner_and_large_full_model_budget(
    tmp_path: Path,
) -> None:
    replicate_client = FakeReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.8,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"medium",'
                '"max_completion_tokens":null},"ambiguities":[],"missing_params":[]}'
            ),
            "Полный ответ.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "planner-budget-test"},
                messages=[
                    ChatMessage(
                        role="user",
                        content="Привет! Кто ты и что умеешь?",
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert reply == "Полный ответ."
        assert replicate_client.payloads[0].max_completion_tokens is None
        assert replicate_client.payloads[1].model == "gpt-5.4"
        assert replicate_client.payloads[1].max_completion_tokens == 65536
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_clamps_text_budget_to_provider_minimum(tmp_path: Path) -> None:
    replicate_client = FakeReplicateClient(
        [
            (
                '{"intent":"text","confidence":0.95,"target_resource_ids":[],'
                '"params":{"reasoning_effort":"medium","verbosity":"low",'
                '"max_completion_tokens":8},"ambiguities":[],"missing_params":[]}'
            ),
            "Короткий ответ.",
        ]
    )
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "planner-min-budget-test"},
                messages=[ChatMessage(role="user", content="What is 2 + 2?")],
            ),
            request_base_url="http://service.test/",
        )
        assert reply == "Короткий ответ."
        assert replicate_client.payloads[1].max_completion_tokens == 16
        await service.aclose()

    asyncio.run(run())


def test_assistant_graph_does_not_persist_openwebui_meta_requests(
    tmp_path: Path,
) -> None:
    replicate_client = MetaAwareFakeReplicateClient()
    service = AssistantGraphService(
        make_settings(tmp_path),
        replicate_client=replicate_client,
        replicate_image_client=FakeImageClient(
            str(tmp_path / "artifacts" / "images" / "out.png")
        ),
        replicate_qwen_edit_client=FakeQwenClient(),
    )

    async def run() -> None:
        meta_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "owui-meta-test"},
                messages=[
                    ChatMessage(
                        role="user",
                        content=(
                            "### Task:\nGenerate a concise, 3-5 word title "
                            "with an emoji "
                            "summarizing the chat history.\n### Output:\n"
                            'JSON format: { "title": "your concise title here" }\n'
                            "### Chat History:\n<chat_history>\nUSER: "
                            "hello\n</chat_history>"
                        ),
                    )
                ],
            ),
            request_base_url="http://service.test/",
        )
        assert meta_reply == '{"title":"🤖 Тестовый чат"}'

        normal_reply = await service.create_reply(
            ChatCompletionRequest(
                model="assistant",
                metadata={"conversation_id": "owui-meta-test"},
                messages=[ChatMessage(role="user", content="Привет! Кто ты?")],
            ),
            request_base_url="http://service.test/",
        )
        assert normal_reply == "Обычный ответ без загрязнения memory."
        assert replicate_client.calls == 3
        await service.aclose()

    asyncio.run(run())
