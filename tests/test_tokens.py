from app.schemas import ChatMessage
from app.tokens import TokenCounter


def test_token_counter_counts_messages_and_completion() -> None:
    counter = TokenCounter()
    messages = [
        ChatMessage(role="system", content="Be short."),
        ChatMessage(role="user", content="Hello world"),
    ]

    usage = counter.build_usage(messages, "Hi")

    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
