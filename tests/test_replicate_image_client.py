import asyncio
import json
from pathlib import Path

import httpx

from app.clients.replicate_files import ReplicateFilesClient
from app.clients.replicate_images import ReplicateImageClient
from app.config import AssistantModel, EchoModel, ReplicateModel, Settings
from app.tool_schemas import ImageGenerationRequest


class FakeFilesClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def prepare_image_url(self, source: str) -> str:
        self.calls.append(source)
        return f"https://api.replicate.com/v1/files/{len(self.calls)}"

    async def aclose(self) -> None:
        return None


def make_settings(tmp_dir: Path | None = None) -> Settings:
    output_dir = tmp_dir or Path("artifacts/test-images")
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        app_reload=False,
        app_log_level="INFO",
        api_prefix="/v1",
        health_path="/health",
        public_base_url=None,
        media_path="/media",
        media_root="artifacts",
        echo_empty_response="",
        echo_model=EchoModel(public_id="echo"),
        assistant_model=AssistantModel(public_id="assistant"),
        assistant_router_model_id="gpt-5-nano",
        assistant_full_model_id="gpt-5.4",
        assistant_sqlite_path="data/test-langgraph.sqlite",
        replicate_api_token="token",
        replicate_base_url="https://api.replicate.com/v1",
        replicate_model_map={},
        replicate_image_tool_id="generate_image",
        replicate_image_model=ReplicateModel(
            public_id="nano-banana-2",
            owner="google",
            name="nano-banana-2",
        ),
        replicate_image_output_dir=str(output_dir),
        replicate_image_download_output=True,
        replicate_qwen_edit_tool_id="edit_image_uncensored",
        replicate_qwen_edit_model=ReplicateModel(
            public_id="qwen-image-edit-plus",
            owner="qwen",
            name="qwen-image-edit-plus",
        ),
        replicate_qwen_edit_output_dir=str(output_dir / "qwen"),
        replicate_qwen_edit_download_output=True,
        replicate_qwen_edit_force_disable_safety_checker=True,
        replicate_local_image_input_roots=(str(output_dir),),
        replicate_default_verbosity=None,
        replicate_default_max_completion_tokens=None,
        replicate_sync_wait_seconds=60,
        replicate_poll_interval_seconds=0.0,
        replicate_poll_timeout_seconds=1.0,
        replicate_http_timeout_seconds=5.0,
        replicate_transport_retries=2,
        replicate_transport_retry_backoff_seconds=0.0,
    )


def test_replicate_image_client_posts_expected_payload(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "delivery.replicate.com":
            return httpx.Response(
                200,
                content=b"png-bytes",
                headers={"Content-Type": "image/png"},
            )
        return httpx.Response(
            200,
            json={
                "id": "pred-1",
                "status": "succeeded",
                "output": ["https://delivery.replicate.com/image.png"],
            },
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        files_client = FakeFilesClient()
        async with (
            httpx.AsyncClient(
                base_url=make_settings(tmp_path).replicate_base_url,
                transport=transport,
            ) as http_client,
            httpx.AsyncClient(transport=transport) as download_client,
        ):
            client = ReplicateImageClient(
                make_settings(tmp_path),
                http_client=http_client,
                files_client=files_client,
                download_client=download_client,
            )
            result = await client.generate_image(
                make_settings(tmp_path).replicate_image_model,
                ImageGenerationRequest(
                    prompt="draw a fox",
                    image_input=["/tmp/ref.png"],
                    aspect_ratio="3:4",
                    resolution="1K",
                    output_format="png",
                ),
                tool_name="generate_image",
            )

        body = json.loads(calls[0].content)
        assert body["input"] == {
            "prompt": "draw a fox",
            "image_input": ["https://api.replicate.com/v1/files/1"],
            "aspect_ratio": "3:4",
            "resolution": "1K",
            "output_format": "png",
        }
        assert files_client.calls == ["/tmp/ref.png"]
        assert result.output_url == "https://delivery.replicate.com/image.png"
        assert result.local_path is not None
        assert Path(result.local_path).is_file()

    asyncio.run(run())


def test_replicate_image_client_polls_until_output_ready(tmp_path: Path) -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "id": "pred-1",
                    "status": "processing",
                    "output": None,
                    "urls": {"get": "https://api.replicate.com/v1/predictions/pred-1"},
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "pred-1",
                    "status": "succeeded",
                    "output": ["https://delivery.replicate.com/image.jpg"],
                },
            ),
            httpx.Response(
                200,
                content=b"jpg-bytes",
                headers={"Content-Type": "image/jpeg"},
            ),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with (
            httpx.AsyncClient(
                base_url=make_settings(tmp_path).replicate_base_url,
                transport=transport,
            ) as http_client,
            httpx.AsyncClient(transport=transport) as download_client,
        ):
            client = ReplicateImageClient(
                make_settings(tmp_path),
                http_client=http_client,
                files_client=FakeFilesClient(),
                download_client=download_client,
            )
            result = await client.generate_image(
                make_settings(tmp_path).replicate_image_model,
                ImageGenerationRequest(prompt="draw a fox"),
                tool_name="generate_image",
            )

        assert result.output_url.endswith(".jpg")
        assert result.local_path is not None

    asyncio.run(run())


def test_replicate_image_client_retries_transient_prediction_failure(
    tmp_path: Path,
) -> None:
    responses = iter(
        [
            httpx.Response(
                200,
                json={
                    "id": "pred-1",
                    "status": "failed",
                    "error": (
                        "Service is currently unavailable due to high demand. "
                        "Please try again later. (E003)"
                    ),
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "pred-2",
                    "status": "succeeded",
                    "output": ["https://delivery.replicate.com/image.png"],
                },
            ),
            httpx.Response(
                200,
                content=b"png-bytes",
                headers={"Content-Type": "image/png"},
            ),
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return next(responses)

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        async with (
            httpx.AsyncClient(
                base_url=make_settings(tmp_path).replicate_base_url,
                transport=transport,
            ) as http_client,
            httpx.AsyncClient(transport=transport) as download_client,
        ):
            client = ReplicateImageClient(
                make_settings(tmp_path),
                http_client=http_client,
                files_client=FakeFilesClient(),
                download_client=download_client,
            )
            result = await client.generate_image(
                make_settings(tmp_path).replicate_image_model,
                ImageGenerationRequest(prompt="draw a fox"),
                tool_name="generate_image",
            )

        assert result.output_url.endswith(".png")
        assert result.local_path is not None

    asyncio.run(run())


def test_replicate_files_client_uploads_local_paths(tmp_path: Path) -> None:
    source = tmp_path / "cat.png"
    source.write_bytes(b"png-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files"
        return httpx.Response(
            200,
            json={"urls": {"get": "https://api.replicate.com/v1/files/1"}},
        )

    async def run() -> None:
        transport = httpx.MockTransport(handler)
        settings = make_settings(tmp_path)
        async with httpx.AsyncClient(
            base_url=settings.replicate_base_url,
            transport=transport,
        ) as http_client:
            client = ReplicateFilesClient(
                settings,
                http_client=http_client,
            )
            file_url = await client.prepare_image_url(str(source))

        assert file_url == "https://api.replicate.com/v1/files/1"

    asyncio.run(run())


def test_replicate_files_client_rejects_disallowed_local_paths(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source = tmp_path / "cat.png"
    source.write_bytes(b"png-bytes")

    async def run() -> None:
        client = ReplicateFilesClient(make_settings(allowed_root))
        try:
            await client.prepare_image_url(str(source))
        except Exception as exc:
            assert "Allowed roots" in str(exc)
        else:
            raise AssertionError("Expected local path validation to fail")
        finally:
            await client.aclose()

    asyncio.run(run())
