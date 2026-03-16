class ReplicateError(RuntimeError):
    """Replicate API request failed."""


class InputValidationError(ValueError):
    """Client input is invalid before the upstream request is made."""


class UserFacingExecutionError(RuntimeError):
    """Execution failed but should be reported to the user in a friendly form."""

    def __init__(
        self,
        *,
        stage: str,
        category: str,
        message: str | None = None,
        retryable: bool = True,
        provider: str | None = None,
        model: str | None = None,
        technical_message: str | None = None,
    ) -> None:
        super().__init__(message or technical_message or "Execution failed.")
        self.stage = stage
        self.category = category
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.technical_message = technical_message


def classify_replicate_error_message(message: str) -> tuple[str, bool]:
    lowered = message.lower()
    if any(token in lowered for token in ("e003", "high demand", "try again later")):
        return "provider_overloaded", True
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout", True
    if any(token in lowered for token in ("transport error", "server disconnected")):
        return "transport_failure", True
    if any(
        token in lowered
        for token in ("unauthorized", "authentication", "invalid token", "api token")
    ):
        return "auth_failed", False
    if any(token in lowered for token in ("safety", "policy", "not allowed")):
        return "policy_rejected", False
    if "api error 5" in lowered:
        return "provider_internal_failure", True
    if "prediction failed" in lowered or "ended with status: failed" in lowered:
        return "provider_internal_failure", True
    return "provider_internal_failure", True
