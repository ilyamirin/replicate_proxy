import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    api_prefix: str
    health_path: str
    echo_empty_response: str


def _normalize_path(path: str) -> str:
    return path if path.startswith("/") else f"/{path}"


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=False)

    return Settings(
        app_name=os.getenv("APP_NAME", "Minimal OpenAI Chat Completions Server"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        api_prefix=_normalize_path(os.getenv("APP_API_PREFIX", "/v1")),
        health_path=_normalize_path(os.getenv("APP_HEALTH_PATH", "/health")),
        echo_empty_response=os.getenv("APP_ECHO_EMPTY_RESPONSE", ""),
    )
