from app.schemas import ChatMessage, content_to_string


class EchoService:
    def __init__(self, empty_response: str) -> None:
        self.empty_response = empty_response

    async def create_reply(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return content_to_string(message.content)
        return self.empty_response
