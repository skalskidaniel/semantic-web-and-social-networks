"""Provider-agnostic LLM interface."""

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """A chat message."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """A completion plus its accounting."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries."""


@runtime_checkable
class LLMClient(Protocol):
    """Anything that completes a chat transcript."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model."""
        ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Return the assistant's reply to ``messages``."""
        ...


def system(content: str) -> Message:
    """Shorthand for a system message."""
    return Message(role="system", content=content)


def user(content: str) -> Message:
    """Shorthand for a user message."""
    return Message(role="user", content=content)
