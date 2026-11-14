"""The multi-agent system as an :class:`OntologyGenerator`."""

import time
from typing import Any

from ontoagents.domain import (
    CQStatus,
    GenerationInputs,
    GenerationResult,
    OntologyDraft,
    TraceEvent,
    Usage,
)
from ontoagents.graph.builder import OntologyGraphBuilder
from ontoagents.graph.state import GraphState, initial_state


class AgentSystem:
    """Runs the compiled LangGraph graph on :class:`GenerationInputs`."""

    def __init__(self, builder: OntologyGraphBuilder, name: str = "agents") -> None:
        self._graph = builder.build()
        self._name = name

    @property
    def name(self) -> str:
        """Identifier used in result files."""
        return self._name

    def generate(self, inputs: GenerationInputs) -> GenerationResult:
        """Plan, then loop generate -> merge -> translate -> validate -> commit/repair/rollback."""
        start = time.perf_counter()
        state = initial_state(
            ontology=inputs.ontology,
            base_iri=inputs.base_iri,
            cqs=[cq.model_dump(exclude={"has_placeholders"}) for cq in inputs.cqs],
            committed_turtle=OntologyDraft.empty(inputs.base_iri).turtle,
            description=inputs.description,
        )
        final: GraphState = self._graph.invoke(state, {"recursion_limit": 100_000})
        return self.to_result(final, time.perf_counter() - start)

    @staticmethod
    def to_result(final: GraphState, wall_time_s: float) -> GenerationResult:
        """Convert the final graph state into a :class:`GenerationResult`."""
        committed = set(final.get("committed_cqs", []))
        all_ids = [c["id"] for c in final["cqs"]]
        status = {k: CQStatus(v) for k, v in final.get("cq_status", {}).items()}
        for cq_id in all_ids:
            if cq_id not in status:
                status[cq_id] = CQStatus.FAIL
        trace_events: list[dict[str, Any]] = final.get("trace", [])
        return GenerationResult(
            draft=OntologyDraft(base_iri=final["base_iri"], turtle=final["committed_turtle"]),
            usage=Usage(
                calls=final.get("llm_calls", 0),
                cached_calls=final.get("cached_calls", 0),
                input_tokens=final.get("input_tokens", 0),
                output_tokens=final.get("output_tokens", 0),
            ),
            wall_time_s=wall_time_s,
            trace=tuple(TraceEvent.model_validate(e) for e in trace_events),
            sparql=dict(final.get("sparql_cache", {})),
            cq_status=status,
            failed_cqs=tuple(i for i in all_ids if i not in committed),
            stop_reason=final.get("stop_reason"),
        )
