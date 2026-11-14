import json

from ontoagents.agents.base import extract_block, extract_json
from ontoagents.agents.cq_translator import CQToSparqlAgent
from ontoagents.agents.generator import GeneratorAgent
from ontoagents.agents.planner import GroupPlanner, LLMPlanner
from ontoagents.domain import CompetencyQuestion, OntologyDraft
from ontoagents.llm.fake import FakeLLMClient, last_user_message
from ontoagents.patterns.repository import StaticPatternRepository


def cq(i, q="Which animals eat plants?", group=None):
    return CompetencyQuestion(id=i, ontology="t", question=q, group=group)


def test_extract_block_prefers_tagged_and_concatenates():
    text = "intro\n```turtle\n:A a owl:Class .\n```\nmore\n```turtle\n:B a owl:Class .\n```"
    assert extract_block(text, ("turtle",)) == ":A a owl:Class .\n:B a owl:Class ."
    assert extract_block("```\nraw\n```", ("turtle",)) == "raw"
    assert extract_block("no block", ("turtle",)) is None


def test_extract_json_fenced_and_bare():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} trailing') == {"a": [1, 2]}


def test_generator_extracts_fragment_and_renders_context(base_iri):
    llm = FakeLLMClient(["Reasoning...\n```turtle\n:Animal a owl:Class .\n```"])
    out = GeneratorAgent(llm).generate(
        [cq("c1")],
        OntologyDraft.empty(base_iri).signature(),
        base_iri,
        patterns=StaticPatternRepository().all()[:1],
        description="A zoo.",
        feedback="1. add :Plant",
        previous_fragment=":Old a owl:Class .",
    )
    assert out.fragment == ":Animal a owl:Class ."
    prompt = last_user_message(llm.calls[0])
    for needle in ("[c1] Which animals eat plants?", "A zoo.", "1. add :Plant", ":Old a owl:Class"):
        assert needle in prompt
    assert base_iri in llm.calls[0][0].content
    assert out.usage.calls == 1


def test_translator_retries_unparseable_query(base_iri):
    bad = json.dumps({"queries": [{"cq_id": "c1", "sparql": "SELECT ?x WHERE { ?x a "}]})
    good = json.dumps({"queries": [{"cq_id": "c1", "sparql": "SELECT ?x WHERE { ?x a :A }"}]})
    llm = FakeLLMClient([f"```json\n{bad}\n```", f"```json\n{good}\n```"])
    out = CQToSparqlAgent(llm).translate(
        [cq("c1"), cq("c2")], OntologyDraft.empty(base_iri).signature(), base_iri
    )
    assert out.queries["c1"] == "SELECT ?x WHERE { ?x a :A }"
    assert out.queries["c2"].startswith("UNTRANSLATABLE")  # omitted by the model
    assert out.parse_retries == 1 and out.usage.calls == 2
    assert "does not parse" in last_user_message(llm.calls[1])


def test_group_planner_respects_groups_and_size():
    cqs = [cq("a", group="g1"), cq("b", group="g2"), cq("c", group="g1"), cq("d", group="g1")]
    plan = GroupPlanner(batch_size=2).plan(cqs)
    assert [b.cq_ids for b in plan.batches] == [("a", "c"), ("d",), ("b",)]


def test_llm_planner_partitions_and_appends_missing():
    reply = json.dumps({"batches": [{"cq_ids": ["c", "a", "zzz"], "rationale": "r"}]})
    planner = LLMPlanner(FakeLLMClient([reply]), batch_size=5)
    plan = planner.plan([cq("a"), cq("b"), cq("c")])
    assert [b.cq_ids for b in plan.batches] == [("c", "a"), ("b",)]
    assert plan.notes and "omitted 1" in plan.notes


def test_llm_planner_falls_back_on_garbage():
    plan = LLMPlanner(FakeLLMClient(["no json here"]), batch_size=2).plan(
        [cq("a"), cq("b"), cq("c")]
    )
    assert [b.cq_ids for b in plan.batches] == [("a", "b"), ("c",)]
    assert plan.notes and "fell back" in plan.notes
    assert plan.usage.calls == 1


def test_pattern_retrieval():
    repo = StaticPatternRepository()
    top = repo.retrieve([cq("x", "Which parts does a machine consist of?")], k=2)
    assert top[0].name == "PartOf"
    assert repo.retrieve([cq("y", "Zzz?")]) == []
