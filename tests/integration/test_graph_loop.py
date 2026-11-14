"""The agent loop end to end with a scripted fake LLM (no network, no Java)."""

import json
import re
from collections.abc import Callable, Sequence

import pytest

from ontoagents.agents.cq_translator import CQToSparqlAgent
from ontoagents.agents.critic import CriticRepairAgent
from ontoagents.agents.generator import GeneratorAgent
from ontoagents.agents.planner import GroupPlanner
from ontoagents.domain import CompetencyQuestion, CQStatus, GenerationInputs
from ontoagents.graph.budget import BudgetPolicy
from ontoagents.graph.builder import GraphNodes, OntologyGraphBuilder, SystemFlags
from ontoagents.graph.system import AgentSystem
from ontoagents.llm.client import Message
from ontoagents.llm.fake import FakeLLMClient
from ontoagents.validation.cq_coverage import CQCoverageValidator
from ontoagents.validation.pipeline import ValidatorPipeline
from ontoagents.validation.regression import RegressionValidator
from ontoagents.validation.syntax import SyntaxValidator

CQ_RE = re.compile(r"^- \[([^\]]+)\]", re.MULTILINE)

#: per-CQ ground truth for the fake: (good fragment, query)
SCRIPT: dict[str, tuple[str, str]] = {
    "c1": (
        ":Animal a owl:Class . :Lion a owl:Class ; rdfs:subClassOf :Animal .",
        "SELECT ?x WHERE { ?x a :Lion }",
    ),
    "c2": (
        ":eats a owl:ObjectProperty ; rdfs:domain :Animal ; rdfs:range :Animal .",
        "SELECT ?x ?y WHERE { ?x a :Lion ; :eats ?y }",
    ),
    "c3": (":Plant a owl:Class .", "SELECT ?p WHERE { ?p a :Plant }"),
    "c4": (":Keeper a owl:Class .", "SELECT ?k WHERE { ?k a :Keeper }"),
}


class ScriptedLLM:
    """Plays generator/translator/critic. ``generator_override(batch_ids, attempt)`` may
    return a custom fragment to simulate failures."""

    def __init__(
        self, generator_override: Callable[[tuple[str, ...], int], str | None] | None = None
    ) -> None:
        self.override = generator_override
        self.attempts: dict[tuple[str, ...], int] = {}
        self.translator_batches: list[tuple[str, ...]] = []
        self.roles: list[str] = []
        self.client = FakeLLMClient(handler=self.reply)

    def reply(self, messages: Sequence[Message]) -> str:
        sys_text = messages[0].content
        user_text = messages[-1].content
        ids = tuple(CQ_RE.findall(user_text))
        if "translate competency questions" in sys_text:
            self.roles.append("translator")
            self.translator_batches.append(ids)
            queries = [{"cq_id": i, "sparql": SCRIPT[i][1]} for i in ids]
            return "```json\n" + json.dumps({"queries": queries}) + "\n```"
        if "meticulous ontology reviewer" in sys_text:
            self.roles.append("critic")
            return "1. Fix it."
        self.roles.append("generator")
        attempt = self.attempts.get(ids, 0)
        self.attempts[ids] = attempt + 1
        if self.override is not None:
            custom = self.override(ids, attempt)
            if custom is not None:
                return f"```turtle\n{custom}\n```"
        return "```turtle\n" + "\n".join(SCRIPT[i][0] for i in ids) + "\n```"


def make_system(
    llm: ScriptedLLM,
    *,
    policy: BudgetPolicy | None = None,
    flags: SystemFlags | None = None,
    batch_size: int = 2,
) -> AgentSystem:
    flags = flags or SystemFlags(use_patterns=False, retry_failed=False)
    validators = [SyntaxValidator()]
    if flags.use_translator:
        validators += [CQCoverageValidator(), RegressionValidator()]
    nodes = GraphNodes(
        planner=GroupPlanner(batch_size),
        generator=GeneratorAgent(llm.client),
        pipeline=ValidatorPipeline(validators),
        flags=flags,
        translator=CQToSparqlAgent(llm.client) if flags.use_translator else None,
        critic=CriticRepairAgent(llm.client) if flags.use_critic else None,
    )
    policy = policy or BudgetPolicy(max_repair=3)
    return AgentSystem(OntologyGraphBuilder(nodes, policy, flags))


def inputs(base_iri: str, ids: Sequence[str] = ("c1", "c2", "c3", "c4")) -> GenerationInputs:
    return GenerationInputs(
        ontology="zoo",
        base_iri=base_iri,
        cqs=tuple(CompetencyQuestion(id=i, ontology="zoo", question=f"Q {i}?") for i in ids),
    )


def kinds(result) -> list[str]:
    return [e.kind for e in result.trace]


def test_happy_path_commits_every_batch(base_iri):
    llm = ScriptedLLM()
    result = make_system(llm).generate(inputs(base_iri))
    assert result.failed_cqs == ()
    assert all(s is CQStatus.PASS for s in result.cq_status.values())
    assert kinds(result).count("commit") == 2
    names = {c.iri.removeprefix(base_iri) for c in result.draft.signature().classes}
    assert names == {"Animal", "Lion", "Plant", "Keeper"}
    assert set(result.sparql) == {"c1", "c2", "c3", "c4"}
    assert result.usage.calls == 4  # 2 batches x (generator + translator)


def test_translator_only_sees_current_batch_and_cache_is_reused(base_iri):
    llm = ScriptedLLM()
    make_system(llm).generate(inputs(base_iri))
    assert llm.translator_batches == [("c1", "c2"), ("c3", "c4")]


def test_repair_limit_then_rollback(base_iri):
    def broken_second_batch(ids, attempt):
        return ":Plant a owl:Class" if ids == ("c3", "c4") else None  # missing dot

    llm = ScriptedLLM(broken_second_batch)
    result = make_system(llm, policy=BudgetPolicy(max_repair=3)).generate(inputs(base_iri))
    assert llm.attempts[("c3", "c4")] == 4  # first try + 3 repairs
    assert kinds(result).count("repair") == 3
    assert kinds(result).count("rollback") == 1
    assert set(result.failed_cqs) == {"c3", "c4"}
    # rollback: the committed draft does not contain anything from the failed batch
    names = {c.iri.removeprefix(base_iri) for c in result.draft.signature().classes}
    assert names == {"Animal", "Lion"}
    assert result.cq_status["c1"] is CQStatus.PASS
    assert result.cq_status["c3"] is CQStatus.FAIL


def test_failed_batch_does_not_poison_later_batches(base_iri):
    def broken_first(ids, attempt):
        return "garbage ((" if ids == ("c1", "c2") else None

    llm = ScriptedLLM(broken_first)
    result = make_system(llm, policy=BudgetPolicy(max_repair=1)).generate(inputs(base_iri))
    assert set(result.failed_cqs) == {"c1", "c2"}
    assert result.cq_status["c3"] is CQStatus.PASS and result.cq_status["c4"] is CQStatus.PASS


def test_regression_failure_triggers_repair(base_iri):
    def punning_then_fixed(ids, attempt):
        if ids == ("c3", "c4") and attempt == 0:
            # makes :Lion a property, breaking the cached query of c1
            return ":Plant a owl:Class . :Keeper a owl:Class . :Lion a owl:ObjectProperty ."
        return None

    llm = ScriptedLLM(punning_then_fixed)
    result = make_system(llm).generate(inputs(base_iri))
    validations = [e.data for e in result.trace if e.kind == "validate"]
    assert validations[1]["failed_validator"] == "regression"
    assert validations[2]["passed"]
    assert result.failed_cqs == ()
    assert "critic" in llm.roles


def test_budget_stop(base_iri):
    llm = ScriptedLLM()
    result = make_system(llm, policy=BudgetPolicy(max_llm_calls=2)).generate(inputs(base_iri))
    assert result.stop_reason == "budget: llm_calls"
    assert set(result.failed_cqs) == {"c3", "c4"}
    assert result.usage.calls == 2


def test_ablation_without_critic_uses_raw_feedback(base_iri):
    def fail_once(ids, attempt):
        return "bad (" if attempt == 0 else None

    llm = ScriptedLLM(fail_once)
    flags = SystemFlags(use_critic=False, use_patterns=False, retry_failed=False)
    result = make_system(llm, flags=flags).generate(inputs(base_iri))
    assert "critic" not in llm.roles
    assert result.failed_cqs == ()
    assert kinds(result).count("repair") == 2


def test_ablation_without_translator_marks_unchecked(base_iri):
    llm = ScriptedLLM()
    flags = SystemFlags(use_translator=False, use_patterns=False, retry_failed=False)
    result = make_system(llm, flags=flags).generate(inputs(base_iri))
    assert "translator" not in llm.roles
    assert set(result.cq_status.values()) == {CQStatus.UNCHECKED}
    assert result.sparql == {}


def test_retry_failed_round_at_the_end(base_iri):
    def fail_first_round(ids, attempt):
        return "bad (" if ids == ("c1", "c2") and attempt < 2 else None

    llm = ScriptedLLM(fail_first_round)
    flags = SystemFlags(use_patterns=False, retry_failed=True)
    result = make_system(llm, flags=flags, policy=BudgetPolicy(max_repair=1)).generate(
        inputs(base_iri)
    )
    assert "retry_round" in kinds(result)
    assert result.failed_cqs == ()


@pytest.mark.reasoner
def test_with_consistency_validator(base_iri):
    from ontoagents.validation.consistency import ConsistencyValidator

    def unsat_once(ids, attempt):
        if ids == ("c3", "c4") and attempt == 0:
            return (
                ":Plant a owl:Class ; owl:disjointWith :Animal . :Keeper a owl:Class . "
                ":Liger rdfs:subClassOf :Lion , :Plant ."
            )
        return None

    llm = ScriptedLLM(unsat_once)
    flags = SystemFlags(use_patterns=False, retry_failed=False)
    nodes = GraphNodes(
        planner=GroupPlanner(2),
        generator=GeneratorAgent(llm.client),
        pipeline=ValidatorPipeline(
            [
                SyntaxValidator(),
                ConsistencyValidator(),
                CQCoverageValidator(),
                RegressionValidator(),
            ]
        ),
        flags=flags,
        translator=CQToSparqlAgent(llm.client),
        critic=CriticRepairAgent(llm.client),
    )
    system = AgentSystem(OntologyGraphBuilder(nodes, BudgetPolicy(), flags))
    result = system.generate(inputs(base_iri))
    failed = [e.data["failed_validator"] for e in result.trace if e.kind == "validate"]
    assert "consistency" in failed
    assert result.failed_cqs == ()
