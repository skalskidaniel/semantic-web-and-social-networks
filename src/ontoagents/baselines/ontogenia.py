"""Ontogenia baseline: Metacognitive Prompting + ODPs, without executable validation.

Reference: Lippolis et al., "Ontology Generation using Large Language Models" (ESWC 2025),
``papers/ontology_generation_using_llm.pdf``, Sect. "Ontogenia technique":

    "The model processes one CQ at a time. When prompted to model a set of CQs, it models
    and merges the generated ontology at each step and provides it in the context. The
    inputs for the prompt are: a user story, ..., a set of ODPs and possibly previous output."

``mode="cq_by_cq"`` (default) follows that procedure: one call per CQ, the full ontology so
far in context, the model returns the complete updated ontology. ``mode="single_pass"`` is
the cheaper variant from the original scaffold plan (all CQs in one call). Neither mode
validates or repairs; that is the point of comparison with the agent system. The baseline
receives exactly the same :class:`GenerationInputs` as the system (the story is
``description.txt`` when ``use_description`` is on).
"""

import time
from typing import Literal

from ontoagents.agents.base import LLMAgent, PromptRenderer, extract_block
from ontoagents.domain import (
    CompetencyQuestion,
    GenerationInputs,
    GenerationResult,
    OntologyDraft,
    TraceEvent,
    TurtleParseError,
    Usage,
)
from ontoagents.llm.client import LLMClient, system, user
from ontoagents.patterns.repository import PatternRepository

OntogeniaMode = Literal["cq_by_cq", "single_pass"]


class OntogeniaBaseline(LLMAgent):
    """Ontogenia reproduction. Unparseable steps keep the previous ontology (see trace)."""

    def __init__(
        self,
        llm: LLMClient,
        renderer: PromptRenderer | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        patterns: PatternRepository | None = None,
        pattern_k: int = 3,
        mode: OntogeniaMode = "cq_by_cq",
    ) -> None:
        super().__init__(llm, renderer, temperature=temperature, max_tokens=max_tokens)
        self._patterns = patterns
        self._pattern_k = pattern_k
        self._mode: OntogeniaMode = mode

    @property
    def name(self) -> str:
        """Identifier used in result files."""
        return "ontogenia" if self._mode == "cq_by_cq" else "ontogenia-single"

    def _step(
        self,
        inputs: GenerationInputs,
        cqs: list[CompetencyQuestion],
        previous: OntologyDraft | None,
    ) -> tuple[str, Usage, list[str]]:
        k = self._pattern_k if self._mode == "cq_by_cq" else max(self._pattern_k, 5)
        patterns = self._patterns.retrieve(cqs, k) if self._patterns else []
        messages = [
            system(
                self._renderer.render(
                    "ontogenia_system.j2",
                    base_iri=inputs.base_iri,
                    incremental=self._mode == "cq_by_cq",
                )
            ),
            user(
                self._renderer.render(
                    "ontogenia_user.j2",
                    description=inputs.description,
                    patterns=[p.to_prompt() for p in patterns],
                    cqs=cqs,
                    previous=previous.turtle if previous is not None else None,
                )
            ),
        ]
        response, usage = self._call(messages)
        turtle = extract_block(response.text, ("turtle", "ttl", "owl", "rdf"))
        return turtle if turtle is not None else response.text, usage, [p.name for p in patterns]

    def generate(self, inputs: GenerationInputs) -> GenerationResult:
        """Generate the ontology CQ by CQ (or in one call in ``single_pass`` mode)."""
        start = time.perf_counter()
        empty = OntologyDraft.empty(inputs.base_iri)
        draft = empty
        usage = Usage()
        events: list[TraceEvent] = []
        steps = [[cq] for cq in inputs.cqs] if self._mode == "cq_by_cq" else [list(inputs.cqs)]
        parse_failures = 0
        for index, step_cqs in enumerate(steps):
            previous = draft if self._mode == "cq_by_cq" else None
            text, step_usage, pattern_names = self._step(inputs, step_cqs, previous)
            usage = usage + step_usage
            try:
                # The model returns the complete ontology, which replaces the previous one,
                # so a step may also revise earlier modelling (unlike the agents' fragments).
                draft = empty.merge(text)
                parsed = True
            except TurtleParseError as exc:
                parse_failures += 1
                parsed = False
                events.append(
                    TraceEvent(kind="parse_error", data={"step": index, "error": str(exc)[:300]})
                )
            events.append(
                TraceEvent(
                    kind="generate",
                    data={
                        "step": index,
                        "cq_ids": [c.id for c in step_cqs],
                        "patterns": pattern_names,
                        "parsed": parsed,
                    },
                )
            )
        stop = None
        if parse_failures == len(steps):
            stop = "unparseable output"
        elif parse_failures:
            stop = f"{parse_failures}/{len(steps)} steps unparseable (previous ontology kept)"
        return GenerationResult(
            draft=draft,
            usage=usage,
            wall_time_s=time.perf_counter() - start,
            trace=tuple(events),
            stop_reason=stop,
        )
