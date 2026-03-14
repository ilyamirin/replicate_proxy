from app.schemas import ChatMessage
from app.config import get_settings


class EchoService:
    async def create_reply(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return get_settings().echo_empty_response


echo_service = EchoService()
