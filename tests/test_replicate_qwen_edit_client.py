import asyncio
import json
from pathlib import Path

import httpx

from app.clients.replicate_qwen_edit import ReplicateQwenEditClient
from app.config import EchoModel, ReplicateModel, Settings
from app.tool_schemas import QwenImageEditRequest


class FakeFilesClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def prepare_image_url(self, source: str) -> str:
        self.calls.append(source)
        return f"https://api.replicate.com/v1/files/{len(self.calls)}"

    async def aclose(self) -> None:
        return None


def make_settings(tmp_dir: Path | None = None) -> Settings:
    output_dir = tmp_dir or Path("artifacts/test-qwen-edit")
    return Settings(
        app_name="Test App",
        app_host="127.0.0.1",
        app_port=8000,
        api_prefix="/v1",
        health_path="/health",
        echo_empty_response="",
        echo_model=EchoModel(public_id="echo"),
        replicate_api_token="token",
        replicate_base_url="https://api.replicate.com/v1",
        replicate_model_map={},
        replicate_image_tool_id="generate_image",
        replicate_image_model=ReplicateModel(
            public_id="nano-banana-2",
            owner="google",
            name="nano-banana-2",
        ),
        replicate_image_output_dir=str(output_dir / "nano"),
        replicate_image_download_output=True,
        replicate_qwen_edit_tool_id="edit_image_uncensored",
        replicate_qwen_edit_model=ReplicateModel(
            public_id="qwen-image-edit-plus",
            owner="qwen",
            name="qwen-image-edit-plus",
        ),
        replicate_qwen_edit_output_dir=str(output_dir),
        replicate_qwen_edit_download_output=True,
        replicate_qwen_edit_force_disable_safety_checker=True,
        replicate_local_image_input_roots=(str(output_dir),),
        replicate_default_verbosity=None,
        replicate_default_max_completion_tokens=None,
        replicate_sync_wait_seconds=60,
        replicate_poll_interval_seconds=0.0,
        replicate_poll_timeout_seconds=1.0,
        replicate_http_timeout_seconds=5.0,
    )


def test_qwen_edit_client_posts_expected_payload(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "delivery.replicate.com":
            return httpx.Response(
                200,
                content=b"webp-bytes",
                headers={"Content-Type": "image/webp"},
            )
        return httpx.Response(
            200,
            json={
                "id": "pred-1",
                "status": "succeeded",
                "output": ["https://delivery.replicate.com/image.webp"],
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
            client = ReplicateQwenEditClient(
                make_settings(tmp_path),
                http_client=http_client,
                files_client=files_client,
                download_client=download_client,
            )
            result = await client.edit_image(
                make_settings(tmp_path).replicate_qwen_edit_model,
                QwenImageEditRequest(
                    prompt="turn this into a watercolor cover",
                    image_input=["tests/fixtures/vision-comic.jpeg"],
                    aspect_ratio="match_input_image",
                    go_fast=True,
                    seed=7,
                    output_format="webp",
                    output_quality=90,
                ),
                tool_name="edit_image_uncensored",
            )

        body = json.loads(calls[0].content)
        assert body["input"] == {
            "prompt": "turn this into a watercolor cover",
            "image": ["https://api.replicate.com/v1/files/1"],
            "aspect_ratio": "match_input_image",
            "go_fast": True,
            "seed": 7,
            "output_format": "webp",
            "output_quality": 90,
            "disable_safety_checker": True,
        }
        assert files_client.calls == ["tests/fixtures/vision-comic.jpeg"]
        assert result.output_urls == ["https://delivery.replicate.com/image.webp"]
        assert len(result.local_paths) == 1
        assert Path(result.local_paths[0]).is_file()

    asyncio.run(run())


def test_qwen_edit_client_polls_until_output_ready(tmp_path: Path) -> None:
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
                    "output": [
                        "https://delivery.replicate.com/image-1.png",
                        "https://delivery.replicate.com/image-2.png",
                    ],
                },
            ),
            httpx.Response(
                200,
                content=b"png-1",
                headers={"Content-Type": "image/png"},
            ),
            httpx.Response(
                200,
                content=b"png-2",
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
            client = ReplicateQwenEditClient(
                make_settings(tmp_path),
                http_client=http_client,
                files_client=FakeFilesClient(),
                download_client=download_client,
            )
            result = await client.edit_image(
                make_settings(tmp_path).replicate_qwen_edit_model,
                QwenImageEditRequest(
                    prompt="turn this into a watercolor cover",
                    image_input=["tests/fixtures/vision-comic.jpeg"],
                ),
                tool_name="edit_image_uncensored",
            )

        assert len(result.output_urls) == 2
        assert len(result.local_paths) == 2

    asyncio.run(run())
