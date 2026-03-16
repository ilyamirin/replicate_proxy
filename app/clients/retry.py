from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from app.clients.errors import ReplicateError

T = TypeVar("T")


async def retry_transport(
    action: Callable[[], Awaitable[httpx.Response]],
    *,
    retries: int,
    backoff_seconds: float,
    error_prefix: str,
) -> httpx.Response:
    last_error: httpx.TransportError | None = None
    for attempt in range(retries + 1):
        try:
            return await action()
        except httpx.TransportError as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(backoff_seconds * (2**attempt))

    raise ReplicateError(f"{error_prefix}: {last_error}") from last_error


async def retry_replicate_error(
    action: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_seconds: float,
    should_retry: Callable[[ReplicateError], bool],
) -> T:
    last_error: ReplicateError | None = None
    for attempt in range(retries + 1):
        try:
            return await action()
        except ReplicateError as exc:
            last_error = exc
            if attempt >= retries or not should_retry(exc):
                raise
            await asyncio.sleep(backoff_seconds * (2**attempt))

    if last_error is not None:
        raise last_error
    raise ReplicateError("Replicate retry loop failed without an error.")


def is_transient_prediction_error(error: ReplicateError) -> bool:
    message = str(error).lower()
    return "high demand" in message or "(e003)" in message
