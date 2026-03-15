from __future__ import annotations

from app.schemas import ChatCompletionRequest, build_messages_from_request

OPENWEBUI_META_REQUEST_KEY = "openwebui_meta_request"


def latest_user_text(payload: ChatCompletionRequest) -> str:
    messages = build_messages_from_request(payload)
    user_contents = []
    for message in messages:
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            text = message.content
        else:
            text = "\n".join(
                part.text for part in message.content if part.type == "text"
            ).strip()
        if text:
            user_contents.append(text)
    if user_contents:
        return user_contents[-1]
    return payload.prompt or ""


def is_openwebui_meta_request(payload: ChatCompletionRequest) -> bool:
    text = latest_user_text(payload)
    if not text.startswith("### Task:"):
        return False
    return any(
        marker in text
        for marker in (
            'JSON format: { "follow_ups":',
            'JSON format: { "title":',
            'JSON format: { "tags":',
            "<chat_history>",
        )
    )


def mark_openwebui_meta_request(payload: ChatCompletionRequest) -> None:
    if is_openwebui_meta_request(payload):
        payload.metadata[OPENWEBUI_META_REQUEST_KEY] = True
