"""Pure routing functions: ``(state, policy) -> next node``."""

from typing import Literal

from ontoagents.domain import PipelineReport
from ontoagents.graph.budget import BudgetPolicy
from ontoagents.graph.state import GraphState

AfterSelect = Literal["generate", "stop", "finish"]
AfterMerge = Literal["translate", "validate"]
AfterValidate = Literal["commit", "repair", "rollback"]


def after_select(state: GraphState, policy: BudgetPolicy) -> AfterSelect:
    """Finish when no batch is left; stop early when the budget is used up."""
    if state.get("batch_index", 0) >= len(state.get("batches", [])):
        return "finish"
    if policy.exhausted(state):
        return "stop"
    return "generate"


def after_merge(state: GraphState, translator_enabled: bool) -> AfterMerge:
    """Skip translation when the fragment did not even parse, or when ablated."""
    if state.get("candidate_error") or not translator_enabled:
        return "validate"
    return "translate"


def after_validate(state: GraphState, policy: BudgetPolicy) -> AfterValidate:
    """Commit on green; repair while allowed; otherwise roll back the batch."""
    report = PipelineReport.model_validate(state.get("report") or {})
    if report.reports and report.passed:
        return "commit"
    if policy.exhausted(state) or not policy.can_repair(state):
        return "rollback"
    return "repair"
