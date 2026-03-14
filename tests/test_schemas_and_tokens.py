from pydantic import ValidationError

from app.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    build_messages_from_request,
    content_to_string,
)
from app.tokens import TokenCounter


def test_content_to_string_preserves_text_and_image_parts() -> None:
    message = ChatMessage(
        role="user",
        content=[
            {"type": "text", "text": "describe image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/cat.png",
                    "detail": "high",
                },
            },
        ],
    )

    assert content_to_string(message.content) == (
        "describe image\n"
        '{"type":"image_url","image_url":{"url":"https://example.com/cat.png","detail":"high"}}'
    )


def test_chat_completion_request_accepts_multimodal_content_parts() -> None:
    payload = ChatCompletionRequest(
        model="gpt-5.4",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/cat.png"},
                    },
                ],
            }
        ],
    )

    assert payload.messages[0].role == "user"
    assert isinstance(payload.messages[0].content, list)


def test_chat_completion_request_rejects_invalid_content_part_type() -> None:
    try:
        ChatCompletionRequest(
            model="gpt-5.4",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio_url": {"url": "x"}}],
                }
            ],
        )
    except ValidationError as exc:
        assert "union_tag_invalid" in str(exc)
    else:
        raise AssertionError("ValidationError was not raised")


def test_token_counter_counts_multimodal_prompt_content() -> None:
    counter = TokenCounter()
    plain_text = [ChatMessage(role="user", content="describe image")]
    multimodal = [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "describe image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
            ],
        )
    ]

    assert counter.count_messages(multimodal) > counter.count_messages(plain_text)


def test_build_messages_from_request_uses_native_input_fields() -> None:
    payload = ChatCompletionRequest(
        model="gpt-5-nano",
        prompt="Describe this image",
        system_prompt="Reply briefly.",
        image_input=["https://example.com/cat.png"],
    )

    messages = build_messages_from_request(payload)
    assert messages[0].role == "system"
    assert messages[0].content == "Reply briefly."
    assert messages[1].role == "user"
    assert content_to_string(messages[1].content).startswith("Describe this image")
