from pathlib import Path

import pytest

from ontoagents.data.catalog import CatalogEntry
from ontoagents.data.reference import ReferenceOntology
from ontoagents.domain import CompetencyQuestion, GenerationResult, OntologyDraft
from ontoagents.evaluation.alignment import LexicalAligner, Term, normalise
from ontoagents.evaluation.metrics import (
    CostMetric,
    EvaluationContext,
    OntologyStatsMetric,
    ReferenceSPARQLMetric,
    StructuralAgreementMetric,
    TermAlignmentMetric,
)
from ontoagents.evaluation.report import render_markdown

REF_NS = "http://ref.example.org/zoo#"
REFERENCE = f"""
@prefix : <{REF_NS}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
:animal a owl:Class . :carnivore a owl:Class ; rdfs:subClassOf :animal .
:lion a owl:Class ; rdfs:subClassOf :carnivore . :plant a owl:Class .
:eats a owl:ObjectProperty ; rdfs:domain :animal .
<http://erlangen-crm.org/current/E21_Person> a owl:Class .
"""
GOLD = f"PREFIX : <{REF_NS}>\nSELECT ?x WHERE {{ ?x a :lion ; :eats ?y }}\n"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hasPart", "has part"),
        ("E21_Person", "person"),
        ("P14_carried_out_by", "carried out by"),
        ("MusicWorks", "music work"),
        ("Categories", "category"),
        ("HTTPServer", "http server"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


def test_aligner_is_one_to_one_and_thresholded():
    gen = [Term("g#Lion"), Term("g#Lions"), Term("g#Rock")]
    ref = [Term("r#lion"), Term("r#plant")]
    alignment = LexicalAligner().align(gen, ref)
    refs = [m.reference for m in alignment.matches]
    assert len(refs) == len(set(refs))
    assert alignment.mapping(0.99) in ({"g#Lion": "r#lion"}, {"g#Lions": "r#lion"})
    assert LexicalAligner().align([], ref).matches == ()


def test_property_alignment_ignores_has_prefix():
    gen = [Term("g#hasEaten", "has eaten")]
    ref = [Term("r#eaten")]
    assert LexicalAligner().align(gen, ref, is_property=True).matches[0].score == 1.0


@pytest.fixture
def reference(tmp_path: Path) -> ReferenceOntology:
    onto_dir = tmp_path / "zoo"
    (onto_dir / "dm").mkdir(parents=True)
    (onto_dir / "rq").mkdir()
    (onto_dir / "dm" / "zoo.ttl").write_text(REFERENCE)
    (onto_dir / "rq" / "q1.rq").write_text(GOLD)
    entry = CatalogEntry(
        name="zoo", directory=onto_dir, dm_files=(onto_dir / "dm" / "zoo.ttl",), has_cqs=False
    )
    return ReferenceOntology.load(entry)


def make_ctx(reference, turtle, base_iri):
    draft = OntologyDraft.empty(base_iri).merge(turtle)
    return EvaluationContext(
        ontology="zoo",
        cqs=[CompetencyQuestion(id="c", ontology="zoo", question="Which lions eat?")],
        result=GenerationResult(draft=draft),
        reference=reference,
    )


GENERATED = """
:Animal a owl:Class . :Lion a owl:Class ; rdfs:subClassOf :Animal .
:Rock a owl:Class .
:eats a owl:ObjectProperty ; rdfs:domain :Lion .
"""


def test_term_alignment_metric(reference, base_iri):
    values = TermAlignmentMetric("class").compute(make_ctx(reference, GENERATED, base_iri))
    # generated: Animal, Lion, Rock ; reference: animal, carnivore, lion, plant, E21_Person
    assert values["class_n_matched"] == 2
    assert values["class_precision"] == pytest.approx(2 / 3)
    assert values["class_recall"] == pytest.approx(2 / 5)
    assert values["class_recall_imported"] == 0.0
    assert values["class_recall_own"] == pytest.approx(2 / 4)
    assert values["class_recall_cq_relevant"] == 1.0  # gold query uses :lion only
    assert values["class_f1@1"] == values["class_f1"]


def test_structural_agreement(reference, base_iri):
    values = StructuralAgreementMetric().compute(make_ctx(reference, GENERATED, base_iri))
    assert values["subclass_edges_aligned"] == 1
    assert values["subclass_agreement"] == 1.0  # Lion ⊑ Animal holds transitively
    assert values["domain_range_agreement"] == 1.0  # Lion is below the reference domain


def test_reference_sparql_rewriting(reference, base_iri):
    values = ReferenceSPARQLMetric().compute(make_ctx(reference, GENERATED, base_iri))
    assert values["gold_queries_parseable"] == 1
    assert values["gold_query_pass_rate"] == 1.0
    poor = ReferenceSPARQLMetric().compute(make_ctx(reference, ":Rock a owl:Class .", base_iri))
    assert poor["gold_query_pass_rate"] == 0.0


def test_stats_and_cost(reference, base_iri):
    ctx = make_ctx(reference, GENERATED, base_iri)
    stats = OntologyStatsMetric().compute(ctx)
    assert stats["n_classes"] == 3 and stats["property_domain_rate"] == 1.0
    assert CostMetric().compute(ctx)["llm_calls"] == 0


def test_report_renders_tables():
    rows = [
        {"experiment": "e1", "system": "s", "ontology": "a", "metrics": {"class_f1": 0.5}},
        {"experiment": "e2", "system": "t", "ontology": "a", "metrics": {"class_f1": 0.7}},
        {"experiment": "e2", "system": "t", "ontology": "b", "error": "Boom\ntrace"},
    ]
    text = render_markdown(rows)
    assert "## e1" in text and "## Macro averages" in text and "ERROR: Boom" in text
