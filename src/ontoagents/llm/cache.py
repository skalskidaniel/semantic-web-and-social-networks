"""Content-addressed disk cache in front of any :class:`LLMClient`.

Makes reruns free and reproducible: the same transcript, model and sampling settings
always return the same text. Cached responses keep their original token counts (so cost
metrics describe the true cost) and are flagged ``cached=True``.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from ontoagents.llm.client import LLMClient, LLMResponse, Message


class CachingLLMClient:
    """Decorator that memoises completions on disk."""

    def __init__(self, inner: LLMClient, cache_dir: Path, namespace: str = "") -> None:
        self._inner = inner
        self._dir = cache_dir
        self._namespace = namespace
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self) -> str:
        """Model identifier of the wrapped client."""
        return self._inner.model_name

    def _key(
        self, messages: Sequence[Message], temperature: float | None, max_tokens: int | None
    ) -> str:
        blob = json.dumps(
            {
                "ns": self._namespace,
                "model": self._inner.model_name,
                "messages": [m.model_dump() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return a cached response or delegate and store the result."""
        key = self._key(messages, temperature, max_tokens)
        path = self._dir / key[:2] / f"{key}.json"
        if path.exists():
            cached = LLMResponse.model_validate_json(path.read_text(encoding="utf-8"))
            return cached.model_copy(update={"cached": True})
        response = self._inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json(), encoding="utf-8")
        tmp.replace(path)
        return response
