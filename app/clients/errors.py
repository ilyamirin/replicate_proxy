class ReplicateError(RuntimeError):
    """Replicate API request failed."""


class InputValidationError(ValueError):
    """Client input is invalid before the upstream request is made."""
