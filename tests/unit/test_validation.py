import pytest

from ontoagents.domain import CQStatus, OntologyDraft
from ontoagents.validation.base import ValidationContext
from ontoagents.validation.consistency import ConsistencyValidator
from ontoagents.validation.cq_coverage import CQCoverageValidator, StructuralQueryChecker
from ontoagents.validation.pipeline import ValidatorPipeline
from ontoagents.validation.regression import RegressionValidator
from ontoagents.validation.syntax import SyntaxValidator

ZOO = """
:Animal a owl:Class . :Plant a owl:Class . :Lion a owl:Class ; rdfs:subClassOf :Animal .
:Animal owl:disjointWith :Plant .
:eats a owl:ObjectProperty ; rdfs:domain :Animal ;
    rdfs:range [ a owl:Class ; owl:unionOf (:Animal :Plant) ] .
:likes a owl:ObjectProperty .
:name a owl:DatatypeProperty ; rdfs:domain :Animal ; rdfs:range xsd:string .
"""


@pytest.fixture
def draft(base_iri):
    return OntologyDraft.empty(base_iri).merge(ZOO)


def check(draft, query):
    return StructuralQueryChecker().check("q", query, draft.signature(), draft.base_iri)


@pytest.mark.parametrize(
    ("query", "status"),
    [
        ("SELECT ?a ?b WHERE { ?a a :Lion ; :eats ?b . ?b a :Plant }", CQStatus.PASS),
        ("SELECT ?a WHERE { ?a a :Plant ; :eats ?b }", CQStatus.FAIL),  # domain mismatch
        ("SELECT ?a WHERE { ?a a :Zebra }", CQStatus.FAIL),  # unknown class
        ("SELECT ?a WHERE { ?a :flies ?b }", CQStatus.FAIL),  # unknown property
        ("SELECT ?a WHERE { ?a a :eats }", CQStatus.FAIL),  # property used as class
        ("SELECT ?a WHERE { ?a :likes ?b }", CQStatus.WARN),  # no domain/range declared
        ("SELECT ?a WHERE { ?a :eats :SomeIndividual }", CQStatus.PASS),  # individuals allowed
        ("SELECT ?n WHERE { ?a a :Lion ; :eats/:name ?n }", CQStatus.PASS),  # property path
        ("SELECT ?a WHERE { ?a :eats 'grass' }", CQStatus.FAIL),  # literal on object property
        ("UNTRANSLATABLE: opinion question", CQStatus.UNTRANSLATABLE),
        ("SELECT ?a WHERE { ?a a ", CQStatus.FAIL),  # does not parse
        (
            "SELECT ?x WHERE { ?x rdfs:subClassOf :Animal, "
            "[ a owl:Restriction ; owl:onProperty :eats ; owl:someValuesFrom ?y ] }",
            CQStatus.PASS,
        ),  # TBox-style query (CLaRO gold queries look like this)
    ],
)
def test_structural_checker(draft, query, status):
    assert check(draft, query).status is status


def test_prefixes_are_added_automatically(draft):
    assert check(draft, "SELECT ?a WHERE { ?a rdf:type :Lion }").status is CQStatus.PASS


def test_syntax_validator(base_iri, draft):
    ok = SyntaxValidator().validate(ValidationContext(base_iri=base_iri, fragment=ZOO))
    assert ok.passed
    bad = SyntaxValidator().validate(ValidationContext(base_iri=base_iri, fragment=":x a"))
    assert not bad.passed and "does not parse" in bad.issues[0].message
    foreign = SyntaxValidator().validate(
        ValidationContext(base_iri=base_iri, fragment=":x a <http://xmlns.com/foaf/0.1/Person> .")
    )
    assert not foreign.passed
    seealso = SyntaxValidator().validate(
        ValidationContext(base_iri=base_iri, fragment=":X rdfs:seeAlso <http://wikipedia.org/X> .")
    )
    assert seealso.passed


def test_coverage_validator_fails_batch_on_any_fail(base_iri, draft):
    ctx = ValidationContext(
        base_iri=base_iri,
        candidate=draft,
        batch_sparql={
            "good": "SELECT ?a WHERE { ?a a :Lion }",
            "bad": "SELECT ?a WHERE { ?a a :X }",
        },
    )
    report = CQCoverageValidator().validate(ctx)
    assert not report.passed
    assert {c.cq_id: c.status for c in report.cq_checks} == {
        "good": CQStatus.PASS,
        "bad": CQStatus.FAIL,
    }


def test_untranslatable_does_not_fail_batch(base_iri, draft):
    ctx = ValidationContext(
        base_iri=base_iri, candidate=draft, batch_sparql={"q": "UNTRANSLATABLE"}
    )
    assert CQCoverageValidator().validate(ctx).passed


def test_regression_detects_broken_earlier_cq(base_iri, draft):
    punned = draft.merge(":Lion a owl:ObjectProperty .")  # Lion stops being a class
    ctx = ValidationContext(
        base_iri=base_iri,
        candidate=punned,
        cached_sparql={"old": "SELECT ?a WHERE { ?a a :Lion }"},
        previous_status={"old": CQStatus.PASS},
    )
    report = RegressionValidator().validate(ctx)
    assert not report.passed and "regression" in report.issues[0].message
    # CQs that were not passing before are not regressions
    ctx2 = ctx.model_copy(update={"previous_status": {"old": CQStatus.FAIL}})
    assert RegressionValidator().validate(ctx2).passed


def test_pipeline_stops_at_first_failure(base_iri):
    pipeline = ValidatorPipeline([SyntaxValidator(), CQCoverageValidator()])
    report = pipeline.run(ValidationContext(base_iri=base_iri, fragment="not turtle"))
    assert [r.validator for r in report.reports] == ["syntax"]
    assert report.failed_validator == "syntax"
    assert "does not parse" in report.to_feedback()


@pytest.mark.reasoner
def test_consistency_validator(base_iri, draft):
    validator = ConsistencyValidator()
    assert validator.validate(ValidationContext(base_iri=base_iri, candidate=draft)).passed
    unsat = draft.merge(":Liger rdfs:subClassOf :Lion , :Plant .")
    report = validator.validate(ValidationContext(base_iri=base_iri, candidate=unsat))
    assert not report.passed and "Liger" in report.issues[0].message
    inconsistent = draft.merge(":leo a :Lion , :Plant .")  # individual violating disjointness
    report = validator.validate(ValidationContext(base_iri=base_iri, candidate=inconsistent))
    assert not report.passed
