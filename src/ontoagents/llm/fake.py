"""Deterministic LLM stand-in for tests and dry runs."""

from collections.abc import Callable, Sequence

from ontoagents.llm.client import LLMResponse, Message

Handler = Callable[[Sequence[Message]], str]


class FakeLLMClient:
    """Returns scripted replies.

    Either pass a list of replies (consumed in order; the last one repeats) or a handler
    that maps the transcript to a reply. Every call is recorded in :attr:`calls`.
    """

    def __init__(
        self,
        replies: Sequence[str] | None = None,
        handler: Handler | None = None,
        model: str = "fake",
    ) -> None:
        if (replies is None) == (handler is None):
            raise ValueError("pass exactly one of replies or handler")
        self._replies = list(replies or [])
        self._handler = handler
        self._model = model
        self.calls: list[list[Message]] = []

    @property
    def model_name(self) -> str:
        """Model identifier."""
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the next scripted reply."""
        self.calls.append(list(messages))
        if self._handler is not None:
            text = self._handler(messages)
        else:
            index = min(len(self.calls) - 1, len(self._replies) - 1)
            text = self._replies[index]
        prompt_chars = sum(len(m.content) for m in messages)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=max(1, prompt_chars // 4),
            output_tokens=max(1, len(text) // 4),
        )


def last_user_message(messages: Sequence[Message]) -> str:
    """Content of the final user message (handy inside fake handlers)."""
    return next((m.content for m in reversed(messages) if m.role == "user"), "")
