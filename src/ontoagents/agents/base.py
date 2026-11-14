"""Shared agent plumbing: prompt rendering, output parsing, usage accounting."""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ontoagents.domain import Usage
from ontoagents.llm.client import LLMClient, LLMResponse, Message

PROMPT_DIR = Path(__file__).parent / "prompts"
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


class PromptRenderer:
    """Renders jinja2 templates from ``agents/prompts``."""

    def __init__(self, directory: Path = PROMPT_DIR) -> None:
        self._env = Environment(
            loader=FileSystemLoader(directory),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
            autoescape=False,  # prompts, not HTML  # noqa: S701
        )

    def render(self, template: str, **context: Any) -> str:
        """Render ``template`` (file name without directory)."""
        return self._env.get_template(template).render(**context).strip()


def extract_block(text: str, languages: Sequence[str]) -> str | None:
    """Content of the first fenced block tagged with one of ``languages`` (or untagged).

    If several blocks match, they are concatenated (LLMs sometimes split Turtle output).
    """
    wanted = {lang.lower() for lang in languages}
    tagged = [body for lang, body in _FENCE_RE.findall(text) if lang.lower() in wanted]
    if tagged:
        return "\n".join(b.strip() for b in tagged)
    untagged = [body for lang, body in _FENCE_RE.findall(text) if not lang]
    if untagged:
        return untagged[0].strip()
    return None


def extract_json(text: str) -> Any:
    """Parse the first JSON block (fenced or bare) in ``text``."""
    block = extract_block(text, ("json",))
    candidates = [block] if block is not None else []
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start >= 0:
        candidates.append(text[start:])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return json.JSONDecoder().raw_decode(candidate.strip())[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")


class LLMAgent:
    """Base class: holds the injected client and accounts usage per call."""

    def __init__(
        self,
        llm: LLMClient,
        renderer: PromptRenderer | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._llm = llm
        self._renderer = renderer or PromptRenderer()
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _call(self, messages: Sequence[Message]) -> tuple[LLMResponse, Usage]:
        response = self._llm.complete(
            messages, temperature=self._temperature, max_tokens=self._max_tokens
        )
        usage = Usage(
            calls=1,
            cached_calls=int(response.cached),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return response, usage
