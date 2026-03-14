import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_REPLICATE_MODEL_MAP = "gpt-5.4=openai/gpt-5.4"


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
    replicate_reasoning_effort: str
    replicate_verbosity: str
    replicate_max_completion_tokens: int
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
        replicate_reasoning_effort=os.getenv("REPLICATE_REASONING_EFFORT", "minimal"),
        replicate_verbosity=os.getenv("REPLICATE_VERBOSITY", "low"),
        replicate_max_completion_tokens=int(
            os.getenv("REPLICATE_MAX_COMPLETION_TOKENS", "4096")
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
        replicate_reasoning_effort=settings.replicate_reasoning_effort,
        replicate_verbosity=settings.replicate_verbosity,
        replicate_max_completion_tokens=settings.replicate_max_completion_tokens,
        replicate_sync_wait_seconds=settings.replicate_sync_wait_seconds,
        replicate_poll_interval_seconds=settings.replicate_poll_interval_seconds,
        replicate_poll_timeout_seconds=settings.replicate_poll_timeout_seconds,
        replicate_http_timeout_seconds=settings.replicate_http_timeout_seconds,
    )
