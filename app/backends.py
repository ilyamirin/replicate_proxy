from __future__ import annotations

from dataclasses import dataclass

from app.clients.replicate import ReplicateClient
from app.config import Settings
from app.services import EchoService
from app.tokens import TokenCounter


@dataclass
class AppServices:
    settings: Settings
    echo_service: EchoService
    replicate_client: ReplicateClient
    token_counter: TokenCounter

    async def aclose(self) -> None:
        await self.replicate_client.aclose()


def build_services(settings: Settings) -> AppServices:
    return AppServices(
        settings=settings,
        echo_service=EchoService(settings.echo_empty_response),
        replicate_client=ReplicateClient(settings),
        token_counter=TokenCounter(),
    )
