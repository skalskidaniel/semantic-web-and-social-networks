import pytest
from pydantic import ValidationError

from ontoagents.domain import (
    CompetencyQuestion,
    CQStatus,
    OntologyDraft,
    TurtleParseError,
    Usage,
    local_name,
)

FRAGMENT = """
:Animal a owl:Class ; rdfs:label "Animal" .
:Lion a owl:Class ; rdfs:subClassOf :Animal .
:eats a owl:ObjectProperty ; rdfs:domain :Animal ; rdfs:range :Animal .
:name a owl:DatatypeProperty ; rdfs:range xsd:string .
"""


def test_cq_placeholders_flag():
    cq = CompetencyQuestion(id="a", ontology="x", question="Is [this animal] a herbivore?")
    assert cq.has_placeholders
    assert not CompetencyQuestion(id="b", ontology="x", question="Which animals?").has_placeholders


def test_cq_has_no_gold_query_field():
    with pytest.raises(ValidationError):
        CompetencyQuestion(id="a", ontology="x", question="q?", rq=["a.rq"])  # type: ignore[call-arg]


def test_empty_draft_has_ontology_header(base_iri):
    draft = OntologyDraft.empty(base_iri)
    assert "owl:Ontology" in draft.turtle
    assert draft.signature().classes == ()


def test_merge_is_immutable_and_canonical(base_iri):
    empty = OntologyDraft.empty(base_iri)
    merged = empty.merge(FRAGMENT)
    assert empty.signature().classes == ()
    assert {c.iri for c in merged.signature().classes} == {base_iri + "Animal", base_iri + "Lion"}
    # canonical: re-merging the same fragment is a no-op
    assert merged.merge(FRAGMENT).turtle == merged.turtle


def test_merge_rejects_invalid_turtle(base_iri):
    with pytest.raises(TurtleParseError):
        OntologyDraft.empty(base_iri).merge(":Broken a owl:Class")


def test_signature_details(base_iri):
    sig = OntologyDraft.empty(base_iri).merge(FRAGMENT).signature()
    eats = sig.get_property(base_iri + "eats")
    assert eats is not None and eats.kind == "object"
    assert eats.domains == (base_iri + "Animal",)
    name = sig.get_property(base_iri + "name")
    assert name is not None and name.kind == "datatype" and name.domains == ()
    assert sig.superclasses(base_iri + "Lion") == {base_iri + "Lion", base_iri + "Animal"}
    text = sig.to_prompt(base_iri)
    assert ":Lion subClassOf :Animal" in text
    assert "domain=? range=xsd:string" in text


def test_signature_union_ranges(base_iri):
    draft = OntologyDraft.empty(base_iri).merge(
        ":A a owl:Class . :B a owl:Class . "
        ":p a owl:ObjectProperty ; rdfs:range [ a owl:Class ; owl:unionOf (:A :B) ] ."
    )
    prop = draft.signature().get_property(base_iri + "p")
    assert prop is not None and set(prop.ranges) == {base_iri + "A", base_iri + "B"}


def test_diff(base_iri):
    a = OntologyDraft.empty(base_iri)
    b = a.merge(FRAGMENT)
    diff = a.diff(b)
    assert base_iri + "Lion" in diff.added_classes
    assert base_iri + "eats" in diff.added_properties
    assert diff.removed_triples == 0 and diff.added_triples > 0
    assert b.diff(b).is_empty


def test_usage_addition():
    total = Usage(calls=1, input_tokens=10, output_tokens=5) + Usage(calls=2, cached_calls=1)
    assert total.calls == 3 and total.cached_calls == 1 and total.total_tokens == 15


def test_local_name():
    assert local_name("http://x.org/a#Foo") == "Foo"
    assert local_name("http://x.org/a/Bar") == "Bar"
    assert local_name("http://x.org/a/") == "http://x.org/a/"


def test_status_is_ok():
    assert CQStatus.PASS.is_ok and CQStatus.WARN.is_ok
    assert not CQStatus.FAIL.is_ok and not CQStatus.UNTRANSLATABLE.is_ok
