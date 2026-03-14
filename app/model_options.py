from app.config import ReplicateModel

MODEL_REASONING_EFFORTS = {
    "gpt-5.4": {"none", "low", "medium", "high", "xhigh"},
    "gpt-5-nano": {"minimal", "low", "medium", "high"},
}


def allowed_reasoning_efforts(model: ReplicateModel) -> set[str] | None:
    return MODEL_REASONING_EFFORTS.get(model.name)
