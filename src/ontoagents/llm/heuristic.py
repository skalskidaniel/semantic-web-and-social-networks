"""A rule-based fake LLM that plays every agent role, for offline dry runs of the pipeline.

It recognises the role from the system prompt, extracts content words from the CQs and
emits matching Turtle / SPARQL / JSON. Output quality is meaningless; the point is to
exercise the full loop (generation, translation, validation, evaluation) without a model.
"""

import json
import re
from collections.abc import Sequence

from ontoagents.llm.client import LLMResponse, Message

_CQ_LINE_RE = re.compile(r"^- \[([^\]]+)\] (.+)$", re.MULTILINE)
_BASE_RE = re.compile(r'"?:"? \(?= ?<([^>]+)>')
_WORD_RE = re.compile(r"[A-Za-z]{4,}")
_STOP = frozenset(
    """what which where when who whom whose does have that this these those with from into
    about there their they them were been being some many much more most other than also
    each every given used using between kind type types list find there here such only
    same similar possible could would should""".split()
)


def _nouns(question: str) -> list[str]:
    seen: dict[str, None] = {}
    for word in _WORD_RE.findall(question):
        w = word.lower()
        if w in _STOP:
            continue
        if w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        seen.setdefault(w[:1].upper() + w[1:], None)
    return list(seen)[:3] or ["Thing0"]


def _cqs(text: str) -> list[tuple[str, str]]:
    return _CQ_LINE_RE.findall(text)


def _turtle(cqs: Sequence[tuple[str, str]]) -> str:
    lines: list[str] = []
    for _, question in cqs:
        nouns = _nouns(question)
        for n in nouns:
            lines.append(f':{n} a owl:Class ; rdfs:label "{n}" ; rdfs:comment "{n}." .')
        if len(nouns) >= 2:
            a, b = nouns[0], nouns[1]
            lines.append(
                f':has{b} a owl:ObjectProperty ; rdfs:label "has {b}" ; '
                f'rdfs:comment "Links {a} to {b}." ; rdfs:domain :{a} ; rdfs:range :{b} .'
            )
    return "\n".join(dict.fromkeys(lines))


def _sparql(question: str) -> str:
    nouns = _nouns(question)
    if len(nouns) >= 2:
        a, b = nouns[0], nouns[1]
        return f"SELECT ?a ?b WHERE {{ ?a a :{a} ; :has{b} ?b . ?b a :{b} . }}"
    return f"SELECT ?a WHERE {{ ?a a :{nouns[0]} . }}"


def heuristic_reply(messages: Sequence[Message]) -> str:
    """Role-dispatching reply."""
    system_text = " ".join(m.content for m in messages if m.role == "system")
    user_text = "\n".join(m.content for m in messages if m.role == "user")
    cqs = _cqs(user_text)
    if "plan the order" in system_text:
        batches = [
            {"cq_ids": [i for i, _ in cqs[k : k + 5]], "rationale": "sequential"}
            for k in range(0, len(cqs), 5)
        ]
        return "```json\n" + json.dumps({"batches": batches}) + "\n```"
    if "translate competency questions" in system_text:
        queries = [{"cq_id": i, "sparql": _sparql(q)} for i, q in cqs]
        return "```json\n" + json.dumps({"queries": queries}) + "\n```"
    if "meticulous ontology reviewer" in system_text:
        return "1. Declare every class and property used by the failing queries."
    return "```turtle\n" + _turtle(cqs) + "\n```"


class HeuristicLLMClient:
    """:class:`LLMClient` backed by :func:`heuristic_reply`."""

    def __init__(self, model: str = "heuristic") -> None:
        self._model = model

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
        """Deterministic reply with character-based token estimates."""
        text = heuristic_reply(messages)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=sum(len(m.content) for m in messages) // 4,
            output_tokens=len(text) // 4,
        )
