import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_REPLICATE_MODEL_MAP = "gpt-5.4=openai/gpt-5.4"
DEFAULT_REPLICATE_IMAGE_TOOL_MODEL = "google/nano-banana-2"
DEFAULT_REPLICATE_QWEN_EDIT_TOOL_MODEL = "qwen/qwen-image-edit-plus"
DEFAULT_LOCAL_IMAGE_INPUT_ROOTS = "tests/fixtures,artifacts/uploads"


@dataclass(frozen=True)
class EchoModel:
    public_id: str


@dataclass(frozen=True)
class ReplicateModel:
    public_id: str
    owner: str
    name: str


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    api_prefix: str
    health_path: str
    echo_empty_response: str
    echo_model: EchoModel
    replicate_api_token: str
    replicate_base_url: str
    replicate_model_map: dict[str, ReplicateModel]
    replicate_image_tool_id: str
    replicate_image_model: ReplicateModel
    replicate_image_output_dir: str
    replicate_image_download_output: bool
    replicate_qwen_edit_tool_id: str
    replicate_qwen_edit_model: ReplicateModel
    replicate_qwen_edit_output_dir: str
    replicate_qwen_edit_download_output: bool
    replicate_qwen_edit_force_disable_safety_checker: bool
    replicate_local_image_input_roots: tuple[str, ...]
    replicate_default_verbosity: str | None
    replicate_default_max_completion_tokens: int | None
    replicate_sync_wait_seconds: int
    replicate_poll_interval_seconds: float
    replicate_poll_timeout_seconds: float
    replicate_http_timeout_seconds: float


def _normalize_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


def _parse_model_map(raw: str) -> dict[str, ReplicateModel]:
    models: dict[str, ReplicateModel] = {}
    for entry in raw.split(","):
        public_id, target = entry.split("=", 1)
        owner, name = target.split("/", 1)
        model = ReplicateModel(
            public_id=public_id.strip(),
            owner=owner.strip(),
            name=name.strip(),
        )
        models[model.public_id] = model
    return models


def _parse_replicate_model(raw: str, public_id: str | None = None) -> ReplicateModel:
    owner, name = raw.split("/", 1)
    return ReplicateModel(
        public_id=public_id or name.strip(),
        owner=owner.strip(),
        name=name.strip(),
    )


def _optional_str_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _optional_int_env(name: str) -> int | None:
    value = _optional_str_env(name)
    return int(value) if value is not None else None


def _bool_env(name: str, default: bool) -> bool:
    value = _optional_str_env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_local_roots_env(raw: str) -> tuple[str, ...]:
    roots: list[str] = []
    for entry in raw.split(","):
        value = entry.strip()
        if not value:
            continue
        roots.append(str(Path(value).resolve()))
    return tuple(roots)


def load_settings() -> Settings:
    load_dotenv(override=False)

    return Settings(
        app_name=os.getenv("APP_NAME", "Minimal OpenAI Chat Completions Server"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        api_prefix=_normalize_path(os.getenv("APP_API_PREFIX", "/v1")),
        health_path=_normalize_path(os.getenv("APP_HEALTH_PATH", "/health")),
        echo_empty_response=os.getenv("APP_ECHO_EMPTY_RESPONSE", ""),
        echo_model=EchoModel(public_id=os.getenv("ECHO_MODEL_ID", "echo")),
        replicate_api_token=os.getenv("REPLICATE_API_TOKEN", ""),
        replicate_base_url=os.getenv(
            "REPLICATE_BASE_URL", "https://api.replicate.com/v1"
        ),
        replicate_model_map=_parse_model_map(
            os.getenv("REPLICATE_MODEL_MAP", DEFAULT_REPLICATE_MODEL_MAP)
        ),
        replicate_image_tool_id=os.getenv("REPLICATE_IMAGE_TOOL_ID", "generate_image"),
        replicate_image_model=_parse_replicate_model(
            os.getenv(
                "REPLICATE_IMAGE_TOOL_MODEL",
                DEFAULT_REPLICATE_IMAGE_TOOL_MODEL,
            )
        ),
        replicate_image_output_dir=os.getenv(
            "REPLICATE_IMAGE_OUTPUT_DIR", "artifacts/images"
        ),
        replicate_image_download_output=_bool_env(
            "REPLICATE_IMAGE_DOWNLOAD_OUTPUT", True
        ),
        replicate_qwen_edit_tool_id=os.getenv(
            "REPLICATE_QWEN_EDIT_TOOL_ID", "edit_image_uncensored"
        ),
        replicate_qwen_edit_model=_parse_replicate_model(
            os.getenv(
                "REPLICATE_QWEN_EDIT_TOOL_MODEL",
                DEFAULT_REPLICATE_QWEN_EDIT_TOOL_MODEL,
            )
        ),
        replicate_qwen_edit_output_dir=os.getenv(
            "REPLICATE_QWEN_EDIT_OUTPUT_DIR", "artifacts/qwen-edit"
        ),
        replicate_qwen_edit_download_output=_bool_env(
            "REPLICATE_QWEN_EDIT_DOWNLOAD_OUTPUT", True
        ),
        replicate_qwen_edit_force_disable_safety_checker=_bool_env(
            "REPLICATE_QWEN_EDIT_FORCE_DISABLE_SAFETY_CHECKER", True
        ),
        replicate_local_image_input_roots=_parse_local_roots_env(
            os.getenv(
                "REPLICATE_LOCAL_IMAGE_INPUT_ROOTS",
                DEFAULT_LOCAL_IMAGE_INPUT_ROOTS,
            )
        ),
        replicate_default_verbosity=_optional_str_env("REPLICATE_DEFAULT_VERBOSITY"),
        replicate_default_max_completion_tokens=_optional_int_env(
            "REPLICATE_DEFAULT_MAX_COMPLETION_TOKENS"
        ),
        replicate_sync_wait_seconds=int(os.getenv("REPLICATE_SYNC_WAIT_SECONDS", "60")),
        replicate_poll_interval_seconds=float(
            os.getenv("REPLICATE_POLL_INTERVAL_SECONDS", "0.5")
        ),
        replicate_poll_timeout_seconds=float(
            os.getenv("REPLICATE_POLL_TIMEOUT_SECONDS", "90")
        ),
        replicate_http_timeout_seconds=float(
            os.getenv("REPLICATE_HTTP_TIMEOUT_SECONDS", "90")
        ),
    )


def snapshot_settings(settings: Settings) -> Settings:
    return Settings(
        app_name=settings.app_name,
        app_host=settings.app_host,
        app_port=settings.app_port,
        api_prefix=settings.api_prefix,
        health_path=settings.health_path,
        echo_empty_response=settings.echo_empty_response,
        echo_model=EchoModel(public_id=settings.echo_model.public_id),
        replicate_api_token=settings.replicate_api_token,
        replicate_base_url=settings.replicate_base_url,
        replicate_model_map=dict(settings.replicate_model_map),
        replicate_image_tool_id=settings.replicate_image_tool_id,
        replicate_image_model=ReplicateModel(
            public_id=settings.replicate_image_model.public_id,
            owner=settings.replicate_image_model.owner,
            name=settings.replicate_image_model.name,
        ),
        replicate_image_output_dir=settings.replicate_image_output_dir,
        replicate_image_download_output=settings.replicate_image_download_output,
        replicate_qwen_edit_tool_id=settings.replicate_qwen_edit_tool_id,
        replicate_qwen_edit_model=ReplicateModel(
            public_id=settings.replicate_qwen_edit_model.public_id,
            owner=settings.replicate_qwen_edit_model.owner,
            name=settings.replicate_qwen_edit_model.name,
        ),
        replicate_qwen_edit_output_dir=settings.replicate_qwen_edit_output_dir,
        replicate_qwen_edit_download_output=settings.replicate_qwen_edit_download_output,
        replicate_qwen_edit_force_disable_safety_checker=(
            settings.replicate_qwen_edit_force_disable_safety_checker
        ),
        replicate_local_image_input_roots=tuple(
            settings.replicate_local_image_input_roots
        ),
        replicate_default_verbosity=settings.replicate_default_verbosity,
        replicate_default_max_completion_tokens=(
            settings.replicate_default_max_completion_tokens
        ),
        replicate_sync_wait_seconds=settings.replicate_sync_wait_seconds,
        replicate_poll_interval_seconds=settings.replicate_poll_interval_seconds,
        replicate_poll_timeout_seconds=settings.replicate_poll_timeout_seconds,
        replicate_http_timeout_seconds=settings.replicate_http_timeout_seconds,
    )
