from app.config import ReplicateModel

MODEL_REASONING_EFFORTS = {
    "gpt-5.4": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5-nano": {"minimal", "low", "medium", "high"},
}

MODEL_MAX_COMPLETION_TOKEN_BOUNDS = {
    "claude-4.5-sonnet": (1024, 64000),
}


def allowed_reasoning_efforts(model: ReplicateModel) -> set[str] | None:
    return MODEL_REASONING_EFFORTS.get(model.name)


def completion_token_bounds(model: ReplicateModel) -> tuple[int, int] | None:
    return MODEL_MAX_COMPLETION_TOKEN_BOUNDS.get(model.name)
