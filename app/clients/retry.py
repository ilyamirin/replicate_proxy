from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx

from app.clients.errors import ReplicateError


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
