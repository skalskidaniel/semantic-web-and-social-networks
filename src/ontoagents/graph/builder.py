"""Wiring of the LangGraph graph. All behaviour lives in injected classes and pure routers."""

from collections.abc import Sequence
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict

from ontoagents.agents.cq_translator import CQToSparqlAgent
from ontoagents.agents.critic import CriticRepairAgent, raw_feedback
from ontoagents.agents.generator import GeneratorAgent
from ontoagents.agents.planner import Planner
from ontoagents.domain import (
    CompetencyQuestion,
    CQStatus,
    OntologyDraft,
    PipelineReport,
    TurtleParseError,
    Usage,
)
from ontoagents.graph import routing
from ontoagents.graph.budget import BudgetPolicy
from ontoagents.graph.state import GraphState
from ontoagents.patterns.repository import PatternRepository
from ontoagents.validation.base import ValidationContext
from ontoagents.validation.pipeline import ValidatorPipeline


class SystemFlags(BaseModel):
    """Feature switches used for ablations."""

    model_config = ConfigDict(frozen=True)

    use_translator: bool = True
    use_critic: bool = True
    use_patterns: bool = True
    pattern_k: int = 3
    retry_failed: bool = True


def _usage_delta(usage: Usage) -> dict[str, int]:
    return {
        "llm_calls": usage.calls,
        "cached_calls": usage.cached_calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


def _event(kind: str, **data: Any) -> list[dict[str, Any]]:
    return [{"kind": kind, "data": data}]


class GraphNodes:
    """Node implementations; each returns a partial state update."""

    def __init__(
        self,
        planner: Planner,
        generator: GeneratorAgent,
        pipeline: ValidatorPipeline,
        flags: SystemFlags,
        translator: CQToSparqlAgent | None = None,
        critic: CriticRepairAgent | None = None,
        patterns: PatternRepository | None = None,
    ) -> None:
        if flags.use_translator and translator is None:
            raise ValueError("use_translator requires a translator")
        if flags.use_critic and critic is None:
            raise ValueError("use_critic requires a critic")
        self._planner = planner
        self._generator = generator
        self._pipeline = pipeline
        self._flags = flags
        self._translator = translator
        self._critic = critic
        self._patterns = patterns

    # helpers -------------------------------------------------------------------------
    @staticmethod
    def _cqs(state: GraphState) -> dict[str, CompetencyQuestion]:
        return {c["id"]: CompetencyQuestion.model_validate(c) for c in state["cqs"]}

    def _batch(self, state: GraphState) -> list[CompetencyQuestion]:
        cqs = self._cqs(state)
        return [cqs[i] for i in state["batches"][state["batch_index"]]]

    @staticmethod
    def _committed(state: GraphState) -> OntologyDraft:
        return OntologyDraft(base_iri=state["base_iri"], turtle=state["committed_turtle"])

    # nodes ---------------------------------------------------------------------------
    def plan(self, state: GraphState) -> dict[str, Any]:
        """Split all CQs into ordered batches."""
        cqs = list(self._cqs(state).values())
        plan = self._planner.plan(cqs, state.get("description"))
        batches = [list(b.cq_ids) for b in plan.batches]
        return {
            "batches": batches,
            "batch_index": -1,
            **_usage_delta(plan.usage),
            "trace": _event(
                "plan",
                planner=self._planner.name,
                batches=batches,
                rationales=[b.rationale for b in plan.batches],
                notes=plan.notes,
            ),
        }

    def select(self, state: GraphState) -> dict[str, Any]:
        """Advance to the next batch; append a retry round for failed CQs once."""
        batches = list(state["batches"])
        index = state["batch_index"] + 1
        update: dict[str, Any] = {}
        if index >= len(batches) and self._flags.retry_failed and not state.get("retried"):
            committed = set(state.get("committed_cqs", []))
            failed = [c["id"] for c in state["cqs"] if c["id"] not in committed]
            if failed:
                size = max(len(b) for b in batches) if batches else 5
                retry = [failed[i : i + size] for i in range(0, len(failed), size)]
                batches = batches + retry
                update["trace"] = _event("retry_round", cq_ids=failed)
            update["retried"] = True
        update.update(
            {
                "batches": batches,
                "batch_index": index,
                "repair_count": 0,
                "feedback": None,
                "fragment": None,
                "previous_fragment": None,
                "candidate_turtle": None,
                "candidate_error": None,
                "batch_sparql": {},
                "report": None,
            }
        )
        return update

    def generate(self, state: GraphState) -> dict[str, Any]:
        """Ask the generator for a fragment (fresh or repair attempt)."""
        batch = self._batch(state)
        committed = self._committed(state)
        patterns = (
            self._patterns.retrieve(batch, self._flags.pattern_k)
            if self._flags.use_patterns and self._patterns is not None
            else []
        )
        out = self._generator.generate(
            batch,
            committed.signature(),
            state["base_iri"],
            patterns=patterns,
            description=state.get("description"),
            feedback=state.get("feedback"),
            previous_fragment=state.get("previous_fragment"),
        )
        return {
            "fragment": out.fragment,
            "iterations": 1,
            **_usage_delta(out.usage),
            "trace": _event(
                "generate",
                batch=state["batch_index"],
                attempt=state.get("repair_count", 0),
                patterns=[p.name for p in patterns],
                fragment_chars=len(out.fragment),
            ),
        }

    def merge(self, state: GraphState) -> dict[str, Any]:
        """Merge the fragment into a candidate; the committed draft is untouched."""
        try:
            candidate = self._committed(state).merge(state.get("fragment") or "")
        except TurtleParseError as exc:
            return {"candidate_turtle": None, "candidate_error": str(exc)}
        return {"candidate_turtle": candidate.turtle, "candidate_error": None}

    def translate(self, state: GraphState) -> dict[str, Any]:
        """Translate the current batch against the candidate's signature."""
        assert self._translator is not None
        candidate = OntologyDraft(
            base_iri=state["base_iri"], turtle=state["candidate_turtle"] or ""
        )
        out = self._translator.translate(
            self._batch(state), candidate.signature(), state["base_iri"]
        )
        return {
            "batch_sparql": out.queries,
            **_usage_delta(out.usage),
            "trace": _event(
                "translate", batch=state["batch_index"], parse_retries=out.parse_retries
            ),
        }

    def validate(self, state: GraphState) -> dict[str, Any]:
        """Run the deterministic validator pipeline."""
        candidate_turtle = state.get("candidate_turtle")
        candidate = (
            OntologyDraft(base_iri=state["base_iri"], turtle=candidate_turtle)
            if candidate_turtle
            else None
        )
        context = ValidationContext(
            base_iri=state["base_iri"],
            candidate=candidate,
            fragment=state.get("fragment"),
            batch_sparql=state.get("batch_sparql", {}),
            cached_sparql=state.get("sparql_cache", {}),
            previous_status={k: CQStatus(v) for k, v in state.get("cq_status", {}).items()},
        )
        report = self._pipeline.run(context)
        statuses = {c.cq_id: c.status.value for c in report.cq_checks("cq_coverage")}
        return {
            "report": report.model_dump(mode="json"),
            "trace": _event(
                "validate",
                batch=state["batch_index"],
                attempt=state.get("repair_count", 0),
                passed=report.passed,
                failed_validator=report.failed_validator,
                cq_status=statuses,
                n_errors=sum(len(r.errors) for r in report.reports),
            ),
        }

    def repair(self, state: GraphState) -> dict[str, Any]:
        """Critic (or raw report when ablated) produces feedback for the next attempt."""
        report = PipelineReport.model_validate(state["report"])
        update: dict[str, Any] = {
            "repair_count": state.get("repair_count", 0) + 1,
            "previous_fragment": state.get("fragment"),
        }
        if self._flags.use_critic:
            assert self._critic is not None
            out = self._critic.critique(
                self._batch(state),
                self._committed(state).signature(),
                state["base_iri"],
                state.get("fragment") or "",
                report,
            )
            update.update({"feedback": out.instructions, **_usage_delta(out.usage)})
        else:
            update["feedback"] = raw_feedback(report)
        update["trace"] = _event(
            "repair", batch=state["batch_index"], critic=self._flags.use_critic
        )
        return update

    def commit(self, state: GraphState) -> dict[str, Any]:
        """Accept the candidate and cache the batch's SPARQL for regression checks."""
        report = PipelineReport.model_validate(state["report"])
        batch_ids = state["batches"][state["batch_index"]]
        if self._flags.use_translator:
            checks = {c.cq_id: c.status.value for c in report.cq_checks("cq_coverage")}
            statuses = {i: checks.get(i, CQStatus.UNTRANSLATABLE.value) for i in batch_ids}
            sparql = state.get("batch_sparql", {})
            cache = {k: v for k, v in sparql.items() if CQStatus(statuses.get(k, "fail")).is_ok}
        else:
            statuses = {i: CQStatus.UNCHECKED.value for i in batch_ids}
            cache = {}
        return {
            "committed_turtle": state["candidate_turtle"],
            "sparql_cache": cache,
            "cq_status": statuses,
            "committed_cqs": list(batch_ids),
            "trace": _event("commit", batch=state["batch_index"], cq_ids=list(batch_ids)),
        }

    def rollback(self, state: GraphState) -> dict[str, Any]:
        """Discard the candidate; the batch is marked failed."""
        batch_ids = state["batches"][state["batch_index"]]
        report = PipelineReport.model_validate(state.get("report") or {})
        checks = {c.cq_id: c.status.value for c in report.cq_checks("cq_coverage")}
        statuses = {
            cq_id: checks.get(cq_id, CQStatus.FAIL.value)
            for cq_id in batch_ids
            if cq_id not in state.get("committed_cqs", [])
        }
        return {
            "candidate_turtle": None,
            "cq_status": statuses,
            "failed_batches": [f"b{state['batch_index']}"],
            "trace": _event(
                "rollback",
                batch=state["batch_index"],
                cq_ids=list(batch_ids),
                failed_validator=report.failed_validator,
            ),
        }

    def stop(self, state: GraphState, policy: BudgetPolicy) -> dict[str, Any]:
        """Record why the run ended early."""
        reason = policy.exhausted(state) or "stopped"
        return {"stop_reason": reason, "trace": _event("stop", reason=reason)}


class OntologyGraphBuilder:
    """Assembles the graph from injected components."""

    def __init__(self, nodes: GraphNodes, policy: BudgetPolicy, flags: SystemFlags) -> None:
        self._nodes = nodes
        self._policy = policy
        self._flags = flags

    def build(self) -> Any:  # CompiledStateGraph; generic typing is noisy
        """Compile the state graph."""
        n = self._nodes
        graph = StateGraph(GraphState)  # ty: ignore[invalid-argument-type]
        graph.add_node("plan", n.plan)
        graph.add_node("select", n.select)
        graph.add_node("generate", n.generate)
        graph.add_node("merge", n.merge)
        graph.add_node("translate", n.translate)
        graph.add_node("validate", n.validate)
        graph.add_node("repair", n.repair)
        graph.add_node("commit", n.commit)
        graph.add_node("rollback", n.rollback)
        graph.add_node("stop", partial(n.stop, policy=self._policy))

        graph.add_edge(START, "plan")
        graph.add_edge("plan", "select")
        graph.add_conditional_edges(
            "select",
            partial(routing.after_select, policy=self._policy),
            {"generate": "generate", "stop": "stop", "finish": END},
        )
        graph.add_edge("generate", "merge")
        graph.add_conditional_edges(
            "merge",
            partial(routing.after_merge, translator_enabled=self._flags.use_translator),
            {"translate": "translate", "validate": "validate"},
        )
        graph.add_edge("translate", "validate")
        graph.add_conditional_edges(
            "validate",
            partial(routing.after_validate, policy=self._policy),
            {"commit": "commit", "repair": "repair", "rollback": "rollback"},
        )
        graph.add_edge("repair", "generate")
        graph.add_edge("commit", "select")
        graph.add_edge("rollback", "select")
        graph.add_edge("stop", END)
        return graph.compile()


def node_names() -> Sequence[str]:
    """Node names, for documentation and tests."""
    return (
        "plan",
        "select",
        "generate",
        "merge",
        "translate",
        "validate",
        "repair",
        "commit",
        "rollback",
        "stop",
    )
