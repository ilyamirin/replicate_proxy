from app.openwebui_meta import (
    OPENWEBUI_META_REQUEST_KEY,
    is_openwebui_meta_request,
    mark_openwebui_meta_request,
)
from app.schemas import ChatCompletionRequest, ChatMessage


def test_openwebui_meta_request_detection() -> None:
    payload = ChatCompletionRequest(
        model="assistant",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    "### Task:\nGenerate a concise, 3-5 word title with an emoji "
                    "summarizing the chat history.\n### Output:\n"
                    'JSON format: { "title": "your concise title here" }\n'
                    "### Chat History:\n<chat_history>\nUSER: hello\n</chat_history>"
                ),
            )
        ],
    )

    assert is_openwebui_meta_request(payload) is True
    mark_openwebui_meta_request(payload)
    assert payload.metadata[OPENWEBUI_META_REQUEST_KEY] is True


def test_openwebui_meta_request_detection_ignores_normal_prompt() -> None:
    payload = ChatCompletionRequest(
        model="assistant",
        messages=[ChatMessage(role="user", content="Привет! Кто ты?")],
    )

    assert is_openwebui_meta_request(payload) is False
    mark_openwebui_meta_request(payload)
    assert OPENWEBUI_META_REQUEST_KEY not in payload.metadata
