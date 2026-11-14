"""Adapter from any LangChain chat model to :class:`LLMClient`."""

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ontoagents.llm.client import LLMError, LLMResponse, Message


def _to_langchain(message: Message) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return HumanMessage(content=message.content)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


class LangChainClient:
    """Wraps a LangChain ``BaseChatModel``.

    ``seed`` is forwarded only to providers that accept it (OpenAI-compatible); Anthropic
    models ignore it, which is recorded in the experiment config rather than silently faked.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        model: str,
        *,
        seed: int | None = None,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._model = model
        self._seed = seed
        self._temperature = default_temperature
        self._max_tokens = default_max_tokens

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
        """Invoke the chat model and read token usage from ``usage_metadata``."""
        kwargs: dict[str, Any] = {}
        temp = temperature if temperature is not None else self._temperature
        if temp is not None:
            kwargs["temperature"] = temp
        limit = max_tokens if max_tokens is not None else self._max_tokens
        if limit is not None:
            kwargs["max_tokens"] = limit
        if self._seed is not None and "openai" in type(self._chat_model).__module__:
            kwargs["seed"] = self._seed
        try:
            reply = self._chat_model.invoke([_to_langchain(m) for m in messages], **kwargs)
        except Exception as exc:  # provider SDKs raise many exception types
            raise LLMError(f"{self._model}: {exc}") from exc
        usage = getattr(reply, "usage_metadata", None) or {}
        return LLMResponse(
            text=_text(reply.content),
            model=self._model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


def build_langchain_client(
    provider: str,
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    seed: int | None,
) -> LangChainClient:
    """Instantiate a provider chat model (requires the matching optional extra)."""
    chat_model: BaseChatModel
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # ty: ignore[import-not-found]

        chat_model = ChatAnthropic(model=model, max_tokens=max_tokens or 8192)  # type: ignore[call-arg]
    elif provider == "openai":
        from langchain_openai import ChatOpenAI  # ty: ignore[import-not-found]

        chat_model = ChatOpenAI(model=model)
    else:
        raise ValueError(f"unsupported LangChain provider {provider!r}")
    return LangChainClient(
        chat_model,
        model,
        seed=seed,
        default_temperature=temperature,
        default_max_tokens=max_tokens,
    )
