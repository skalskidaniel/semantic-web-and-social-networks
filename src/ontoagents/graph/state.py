"""Graph state: plain JSON-serialisable values only, so checkpointing works."""

import operator
from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer: right-biased dict union."""
    return {**left, **right}


class GraphState(TypedDict, total=False):
    """State of one generation run.

    Fields annotated with a reducer accumulate deltas returned by nodes; all others are
    overwritten. Counters live here; limits live in :class:`BudgetPolicy`.
    """

    # inputs (set once)
    ontology: str
    base_iri: str
    description: str | None
    cqs: list[dict[str, Any]]

    # plan and cursor
    batches: list[list[str]]
    batch_index: int
    retried: bool

    # drafts (canonical Turtle text, never rdflib graphs)
    committed_turtle: str
    fragment: str | None
    previous_fragment: str | None
    candidate_turtle: str | None
    candidate_error: str | None

    # per-attempt artefacts
    batch_sparql: dict[str, str]
    report: dict[str, Any] | None
    feedback: str | None
    repair_count: int

    # accumulated results
    sparql_cache: Annotated[dict[str, str], merge_dicts]
    cq_status: Annotated[dict[str, str], merge_dicts]
    committed_cqs: Annotated[list[str], operator.add]
    failed_batches: Annotated[list[str], operator.add]

    # counters
    iterations: Annotated[int, operator.add]
    llm_calls: Annotated[int, operator.add]
    cached_calls: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]

    trace: Annotated[list[dict[str, Any]], operator.add]
    stop_reason: str | None


def initial_state(
    ontology: str,
    base_iri: str,
    cqs: list[dict[str, Any]],
    committed_turtle: str,
    description: str | None,
) -> GraphState:
    """A fresh state with every counter at zero."""
    return GraphState(
        ontology=ontology,
        base_iri=base_iri,
        description=description,
        cqs=cqs,
        batches=[],
        batch_index=-1,
        retried=False,
        committed_turtle=committed_turtle,
        fragment=None,
        previous_fragment=None,
        candidate_turtle=None,
        candidate_error=None,
        batch_sparql={},
        report=None,
        feedback=None,
        repair_count=0,
        sparql_cache={},
        cq_status={},
        committed_cqs=[],
        failed_batches=[],
        iterations=0,
        llm_calls=0,
        cached_calls=0,
        input_tokens=0,
        output_tokens=0,
        trace=[],
        stop_reason=None,
    )
