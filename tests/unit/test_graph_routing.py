from ontoagents.domain import PipelineReport, ValidationReport
from ontoagents.graph.budget import BudgetPolicy
from ontoagents.graph.routing import after_merge, after_select, after_validate


def report(passed: bool) -> dict:
    return PipelineReport(
        reports=(ValidationReport(validator="syntax", passed=passed),)
    ).model_dump(mode="json")


def test_after_select():
    policy = BudgetPolicy(max_llm_calls=10)
    assert after_select({"batch_index": 2, "batches": [[], []]}, policy) == "finish"
    assert after_select({"batch_index": 0, "batches": [[]], "llm_calls": 3}, policy) == "generate"
    assert after_select({"batch_index": 0, "batches": [[]], "llm_calls": 10}, policy) == "stop"


def test_after_merge():
    assert after_merge({"candidate_error": "boom"}, True) == "validate"
    assert after_merge({"candidate_error": None}, False) == "validate"
    assert after_merge({"candidate_error": None}, True) == "translate"


def test_after_validate():
    policy = BudgetPolicy(max_repair=2)
    assert after_validate({"report": report(True)}, policy) == "commit"
    assert after_validate({"report": report(False), "repair_count": 1}, policy) == "repair"
    assert after_validate({"report": report(False), "repair_count": 2}, policy) == "rollback"
    exhausted = BudgetPolicy(max_repair=5, max_tokens=100)
    state = {"report": report(False), "input_tokens": 90, "output_tokens": 20}
    assert after_validate(state, exhausted) == "rollback"


def test_budget_reasons():
    assert BudgetPolicy().exhausted({"llm_calls": 10**9}) is None
    assert BudgetPolicy(max_iterations=3).exhausted({"iterations": 3}) == "budget: iterations"
    assert BudgetPolicy(max_tokens=5).exhausted({"input_tokens": 5}) == "budget: tokens"
