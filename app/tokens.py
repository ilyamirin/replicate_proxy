from __future__ import annotations

import hashlib
import os
from pathlib import Path

import tiktoken

from app.schemas import ChatMessage, Usage

O200K_BASE_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIKTOKEN_CACHE_DIR = PROJECT_ROOT / ".tiktoken-cache"
TIKTOKEN_CACHE_KEY = hashlib.sha1(  # noqa: S324
    O200K_BASE_URL.encode(),
    usedforsecurity=False,
).hexdigest()
RAW_O200K_FILE = TIKTOKEN_CACHE_DIR / "o200k_base.tiktoken"
CACHED_O200K_FILE = TIKTOKEN_CACHE_DIR / TIKTOKEN_CACHE_KEY


class TokenCounter:
    def __init__(self) -> None:
        self._prepare_cache()
        os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)
        self._encoding = tiktoken.get_encoding("o200k_base")

    def count_text(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[ChatMessage]) -> int:
        total = 3
        for message in messages:
            total += 3
            total += len(self._encoding.encode(message.role))
            total += len(self._encoding.encode(message.content))
        return total

    def build_usage(
        self,
        messages: list[ChatMessage],
        completion_text: str,
    ) -> Usage:
        prompt_tokens = self.count_messages(messages)
        completion_tokens = self.count_text(completion_text)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    @staticmethod
    def _prepare_cache() -> None:
        TIKTOKEN_CACHE_DIR.mkdir(exist_ok=True)
        if CACHED_O200K_FILE.exists():
            return
        if RAW_O200K_FILE.exists():
            CACHED_O200K_FILE.write_bytes(RAW_O200K_FILE.read_bytes())
            return
        raise RuntimeError(
            "Missing .tiktoken-cache/o200k_base.tiktoken. "
            "Download it before starting the service."
        )
