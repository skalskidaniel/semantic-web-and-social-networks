import pytest
from rdflib import Graph

from ontoagents.data.catalog import OntologyCatalog
from ontoagents.data.loader import YamlCQLoader, normalise_question, subsample
from ontoagents.data.reference import (
    ReferenceOntology,
    query_iris,
    sanitise_gold_query,
    term_origin,
)


@pytest.fixture(scope="module")
def catalog(request):
    from tests.conftest import DATA_DIR

    return OntologyCatalog(DATA_DIR)


def test_normalise_question():
    assert (
        normalise_question("  Is [ this animal ]  a herbivore ? ")
        == "Is [this animal] a herbivore?"
    )


def test_placeholder_and_duplicate_ids_get_stable_ids():
    records = [
        {"ID": "<CQId>", "question": "A?"},
        {"ID": "x1", "question": "B?"},
        {"ID": "x1", "question": "C?"},
        {"ID": "", "question": "  "},  # empty question dropped
    ]
    cqs = YamlCQLoader.from_records("onto", records)
    assert [c.id for c in cqs] == ["onto-001", "x1", "onto-003"]
    assert YamlCQLoader.from_records("onto", records) == cqs


def test_rq_field_is_dropped():
    cqs = YamlCQLoader.from_records("o", [{"ID": "a", "question": "Q?", "rq": ["a.rq"]}])
    dumped = cqs[0].model_dump_json()
    assert "a.rq" not in dumped and "rq" not in type(cqs[0]).model_fields


def test_all_dataset_cqs_load_with_unique_ids(catalog, data_dir):
    loader = YamlCQLoader(data_dir)
    total = 0
    for name in catalog.names():
        cqs = loader.load(name)
        assert cqs, name
        assert len({c.id for c in cqs}) == len(cqs), name
        total += len(cqs)
    assert total == 831


def test_no_gold_query_names_leak_into_cqs(catalog, data_dir):
    loader = YamlCQLoader(data_dir)
    for entry in catalog.select("with_reference"):
        rq_dir = entry.directory / "rq"
        names = {p.name for p in rq_dir.iterdir()} if rq_dir.exists() else set()
        blob = " ".join(c.model_dump_json() for c in loader.load(entry.name))
        assert not any(n in blob for n in names if n.endswith((".rq", ".sparql"))), entry.name


def test_subsample_is_deterministic_and_ordered():
    items = list(range(10))
    assert subsample(items, 4) == [0, 2, 5, 7]  # type: ignore[arg-type]
    assert subsample(items, None) == items  # type: ignore[arg-type]
    assert subsample(items, 20) == items  # type: ignore[arg-type]


def test_catalog_flags_and_lookup(catalog):
    assert not catalog.get("SILKNOW").has_reference
    assert catalog.get("NORIA-O").name == "noria-o"
    assert "SILKNOW" not in [e.name for e in catalog.select("with_reference")]
    assert catalog.get("awo").base_iri == "http://ontoagents.example.org/awo#"
    with pytest.raises(KeyError):
        catalog.get("nope")


def test_reference_multi_file_merge_and_time_import(catalog):
    ref = ReferenceOntology.load(catalog.get("DemCare"), catalog.shared_files())
    assert len(ref.loaded_files) == 5  # 4 modules + shared/time.owl
    assert any(p.name == "time.owl" for p in ref.loaded_files)
    assert ref.class_iris("imported")  # OWL-Time terms tagged as imported
    assert all(i.startswith("http://www.demcare.eu/") for i in ref.class_iris("own"))


def test_reference_tags_crm_and_frbroo_as_imported(catalog):
    ref = ReferenceOntology.load(catalog.get("DOREMUS"), catalog.shared_files())
    imported = ref.class_iris("imported")
    assert any(i.startswith("http://erlangen-crm.org/current/") for i in imported)
    assert any(i.startswith("http://erlangen-crm.org/efrbroo/") for i in imported)
    assert term_origin("http://data.doremus.org/ontology#M2_Foo") == "own"


def test_gold_queries_are_linked_to_cq_ids(catalog):
    ref = ReferenceOntology.load(catalog.get("awo"))
    queries = {q.name: q for q in ref.gold_queries()}
    assert queries["awo_1.rq"].cq_ids == ("awo_1",)
    assert ref.cq_relevant_terms()


def test_sanitise_handles_dataset_dialects():
    template = "SELECT ?x WHERE { $PPx1$ rdfs:subClassOf ?x . ?y owl:someValuesFrom $?PPx2$ }"
    fixed = sanitise_gold_query(template)
    assert "?PPx1" in fixed and "?PPx2" in fixed and "PREFIX rdfs:" in fixed
    virtuoso = (
        "PREFIX ex: <http://ex.org/>\nSELECT DISTINCT ?a, SAMPLE(?b) as ?c WHERE { ?a ex:p ?b }"
        " GROUP BY ?a"
    )
    assert "http://ex.org/p" in query_iris(virtuoso)


def test_query_iris_expands_prefixes():
    iris = query_iris("PREFIX : <http://ex.org/#> SELECT ?x WHERE { ?x a :Cat ; :eats ?y }")
    assert {"http://ex.org/#Cat", "http://ex.org/#eats"} <= iris


def test_every_reference_parses(catalog):
    for entry in catalog.select("with_reference"):
        ref = ReferenceOntology.load(entry, catalog.shared_files())
        assert not ref.parse_errors, entry.name
        assert isinstance(ref.graph, Graph) and len(ref.graph) > 0
