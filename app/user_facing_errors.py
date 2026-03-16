from __future__ import annotations

from app.clients.errors import UserFacingExecutionError


def classify_exception(error: Exception) -> tuple[str, bool]:
    if isinstance(error, UserFacingExecutionError):
        return error.category, error.retryable
    return "internal_failure", False


def _default_message(stage: str | None) -> str:
    if stage == "planner":
        return (
            "Что-то пошло не так на этапе маршрутизации запроса. "
            "Попробуй повторить его еще раз."
        )
    if stage == "text_generation":
        return (
            "Что-то пошло не так на этапе обработки текста. Попробуй повторить запрос."
        )
    if stage == "image_generation":
        return (
            "Что-то пошло не так на этапе генерации изображения. "
            "Попробуй повторить запрос."
        )
    if stage == "image_edit":
        return (
            "Что-то пошло не так на этапе редактирования изображения. "
            "Попробуй повторить запрос."
        )
    if stage == "file_upload":
        return "Что-то пошло не так на этапе загрузки файла. Попробуй повторить запрос."
    return "Что-то пошло не так. Попробуй повторить запрос."


def user_facing_message(error: Exception) -> str:
    if isinstance(error, UserFacingExecutionError):
        return error.message or _default_message(error.stage)
    return _default_message(None)


def assistant_error_message(error: Exception) -> str:
    return user_facing_message(error)


def tool_error_payload(tool_name: str, error: Exception) -> dict:
    category, retryable = classify_exception(error)
    provider = None
    model = None
    stage = None
    if isinstance(error, UserFacingExecutionError):
        provider = error.provider
        model = error.model
        stage = error.stage
    return {
        "tool_name": tool_name,
        "status": "failed",
        "error_type": "execution_error",
        "category": category,
        "message": user_facing_message(error),
        "retryable": retryable,
        "provider": provider,
        "model": model,
        "stage": stage,
    }


def api_error_payload(error: Exception) -> dict:
    category, retryable = classify_exception(error)
    provider = None
    model = None
    stage = None
    if isinstance(error, UserFacingExecutionError):
        provider = error.provider
        model = error.model
        stage = error.stage
    return {
        "error": {
            "type": "execution_error",
            "category": category,
            "message": user_facing_message(error),
            "retryable": retryable,
            "provider": provider,
            "model": model,
            "stage": stage,
        }
    }
