"""Ontogenia-style baseline: one metacognitive prompt with all CQs and the ODPs.

Reference: Lippolis et al., "Ontology Generation using Large Language Models" (ESWC 2025),
``papers/ontology_generation_using_llm.pdf``. This reproduction is single-pass: no
validation, no repair, no translator. It receives exactly the same inputs as the system.
"""

import time

from ontoagents.agents.base import LLMAgent, PromptRenderer, extract_block
from ontoagents.domain import (
    GenerationInputs,
    GenerationResult,
    OntologyDraft,
    TraceEvent,
    TurtleParseError,
)
from ontoagents.llm.client import LLMClient, system, user
from ontoagents.patterns.repository import PatternRepository


class OntogeniaBaseline(LLMAgent):
    """Single-pass generator. If the Turtle does not parse, the result is the empty draft."""

    def __init__(
        self,
        llm: LLMClient,
        renderer: PromptRenderer | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        patterns: PatternRepository | None = None,
        pattern_k: int = 5,
    ) -> None:
        super().__init__(llm, renderer, temperature=temperature, max_tokens=max_tokens)
        self._patterns = patterns
        self._pattern_k = pattern_k

    @property
    def name(self) -> str:
        """Identifier used in result files."""
        return "ontogenia"

    def generate(self, inputs: GenerationInputs) -> GenerationResult:
        """Generate the whole ontology in one call."""
        start = time.perf_counter()
        patterns = self._patterns.retrieve(inputs.cqs, self._pattern_k) if self._patterns else []
        messages = [
            system(self._renderer.render("ontogenia_system.j2", base_iri=inputs.base_iri)),
            user(
                self._renderer.render(
                    "ontogenia_user.j2",
                    description=inputs.description,
                    patterns=[p.to_prompt() for p in patterns],
                    cqs=list(inputs.cqs),
                )
            ),
        ]
        response, usage = self._call(messages)
        turtle = extract_block(response.text, ("turtle", "ttl", "owl", "rdf"))
        events = [TraceEvent(kind="generate", data={"patterns": [p.name for p in patterns]})]
        try:
            draft = OntologyDraft.empty(inputs.base_iri).merge(turtle or response.text)
            stop = None
        except TurtleParseError as exc:
            draft = OntologyDraft.empty(inputs.base_iri)
            stop = "unparseable output"
            events.append(TraceEvent(kind="parse_error", data={"error": str(exc)[:500]}))
        return GenerationResult(
            draft=draft,
            usage=usage,
            wall_time_s=time.perf_counter() - start,
            trace=tuple(events),
            stop_reason=stop,
        )
