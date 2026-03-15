from __future__ import annotations

import base64
import binascii
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.clients.errors import InputValidationError, ReplicateError
from app.clients.retry import retry_transport
from app.config import Settings

REPLICATE_FILE_PREFIX = "https://api.replicate.com/v1/files/"


class ReplicateFilesClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        download_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._owns_http_client = http_client is None
        self._owns_download_client = download_client is None
        self._http_client = http_client or httpx.AsyncClient(
            base_url=settings.replicate_base_url,
            timeout=settings.replicate_http_timeout_seconds,
        )
        self._download_client = download_client or httpx.AsyncClient(
            timeout=settings.replicate_http_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "replicate-proxy/0.1"},
        )
        self._cache: dict[str, str] = {}

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()
        if self._owns_download_client:
            await self._download_client.aclose()

    async def prepare_image_url(self, source: str) -> str:
        if source.startswith(REPLICATE_FILE_PREFIX):
            return source
        cached = self._cache.get(source)
        if cached is not None:
            return cached

        if source.startswith("data:"):
            filename, content, content_type = self._decode_data_url(source)
            uploaded_url = await self._upload_file(filename, content, content_type)
        elif self._is_remote_url(source):
            uploaded_url = await self._upload_remote_url(source)
        else:
            uploaded_url = await self._upload_local_path(source)

        self._cache[source] = uploaded_url
        return uploaded_url

    async def _upload_remote_url(self, source: str) -> str:
        response = await retry_transport(
            lambda: self._download_client.get(source),
            retries=self.settings.replicate_transport_retries,
            backoff_seconds=self.settings.replicate_transport_retry_backoff_seconds,
            error_prefix="Replicate image download transport error",
        )
        if response.is_error:
            raise ReplicateError(
                f"Image download error {response.status_code} for {source}: "
                f"{response.text[:200]}"
            )

        content_type = response.headers.get("Content-Type", "application/octet-stream")
        content_type = content_type.split(";", 1)[0]
        filename = self._filename_from_url(source, content_type)
        return await self._upload_file(filename, response.content, content_type)

    async def _upload_file(
        self,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str:
        if not self.settings.replicate_api_token:
            raise ReplicateError("REPLICATE_API_TOKEN is not set.")

        response = await retry_transport(
            lambda: self._http_client.post(
                "/files",
                headers={
                    "Authorization": f"Bearer {self.settings.replicate_api_token}"
                },
                files={"content": (filename, content, content_type)},
            ),
            retries=self.settings.replicate_transport_retries,
            backoff_seconds=self.settings.replicate_transport_retry_backoff_seconds,
            error_prefix="Replicate file upload transport error",
        )
        if response.is_error:
            raise ReplicateError(
                f"Replicate file upload error {response.status_code}: {response.text}"
            )

        payload = response.json()
        file_url = payload.get("urls", {}).get("get")
        if not file_url:
            raise ReplicateError("Replicate file upload did not return a file URL.")
        return str(file_url)

    async def _upload_local_path(self, source: str) -> str:
        path = Path(source).expanduser().resolve()
        self._validate_local_path(path)
        if not path.is_file():
            raise InputValidationError(f"Image file not found: {source}")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not content_type.startswith("image/"):
            raise InputValidationError(
                f"Only image files are allowed for local image_input: {source}"
            )
        return await self._upload_file(path.name, path.read_bytes(), content_type)

    @staticmethod
    def _decode_data_url(source: str) -> tuple[str, bytes, str]:
        header, encoded = source.split(",", 1)
        if ";base64" not in header:
            raise ReplicateError(
                "Only base64 data URLs are supported for image upload."
            )

        content_type = header.removeprefix("data:").split(";", 1)[0]
        if not content_type:
            content_type = "application/octet-stream"
        try:
            content = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ReplicateError("Invalid base64 image data URL.") from exc
        extension = mimetypes.guess_extension(content_type) or ".bin"
        return f"upload{extension}", content, content_type

    @staticmethod
    def _filename_from_url(source: str, content_type: str) -> str:
        path = urlparse(source).path
        filename = path.rsplit("/", 1)[-1]
        if filename:
            return filename
        extension = mimetypes.guess_extension(content_type) or ".bin"
        return f"download{extension}"

    @staticmethod
    def _is_remote_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme in {"http", "https"}

    def _validate_local_path(self, path: Path) -> None:
        if any(
            path.is_relative_to(Path(root))
            for root in self.settings.replicate_local_image_input_roots
        ):
            return
        allowed = ", ".join(self.settings.replicate_local_image_input_roots)
        raise InputValidationError(
            f"Local image_input path is not allowed. Allowed roots: {allowed}"
        )
