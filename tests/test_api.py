from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_completions_echoes_last_user_message() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ignored"},
                {"role": "user", "content": "echo this"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["model"] == "gpt-4o-mini"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "echo this",
    }
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_chat_completions_returns_empty_string_without_user_message() -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "system", "content": "No user input here."}],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == ""


def test_settings_loaded_from_env(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("APP_NAME", "Env App")
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("APP_API_PREFIX", "api")
    monkeypatch.setenv("APP_HEALTH_PATH", "status")
    monkeypatch.setenv("APP_ECHO_EMPTY_RESPONSE", "fallback")

    settings = get_settings()

    assert settings.app_name == "Env App"
    assert settings.app_port == 9000
    assert settings.api_prefix == "/api"
    assert settings.health_path == "/status"
    assert settings.echo_empty_response == "fallback"

    get_settings.cache_clear()
