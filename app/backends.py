from __future__ import annotations

from dataclasses import dataclass

from app.clients.replicate import ReplicateClient
from app.clients.replicate_files import ReplicateFilesClient
from app.clients.replicate_images import ReplicateImageClient
from app.config import Settings
from app.services import EchoService
from app.tokens import TokenCounter


@dataclass
class AppServices:
    settings: Settings
    echo_service: EchoService
    replicate_files_client: ReplicateFilesClient
    replicate_client: ReplicateClient
    replicate_image_client: ReplicateImageClient
    token_counter: TokenCounter

    async def aclose(self) -> None:
        await self.replicate_client.aclose()
        await self.replicate_image_client.aclose()
        await self.replicate_files_client.aclose()


def build_services(settings: Settings) -> AppServices:
    files_client = ReplicateFilesClient(settings)
    return AppServices(
        settings=settings,
        echo_service=EchoService(settings.echo_empty_response),
        replicate_files_client=files_client,
        replicate_client=ReplicateClient(settings, files_client=files_client),
        replicate_image_client=ReplicateImageClient(
            settings,
            files_client=files_client,
        ),
        token_counter=TokenCounter(),
    )
