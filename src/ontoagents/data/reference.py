"""Reference ontologies and gold SPARQL: evaluation-only data.

Import-linter forbids the generation side (agents, graph, validation, llm, patterns,
baselines) from importing this module.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict
from rdflib import OWL, Graph, URIRef
from rdflib.plugins.sparql import prepareQuery

from ontoagents.data.catalog import QUERY_SUFFIXES, CatalogEntry
from ontoagents.domain import Signature, extract_signature

#: Namespaces of vocabularies the dataset's ontologies import or embed. Terms in these
#: namespaces are reported separately from the ontology's own terms.
IMPORTED_NAMESPACE_PREFIXES: tuple[str, ...] = (
    "http://erlangen-crm.org/",
    "http://www.cidoc-crm.org/",
    "http://www.w3.org/2006/time",
    "http://edamontology.org/",
    "http://purl.obolibrary.org/obo/",
    "https://w3id.org/arco/",
    "http://www.w3.org/ns/prov",
    "http://xmlns.com/foaf/",
    "http://www.w3.org/2004/02/skos/",
    "http://purl.org/dc/",
    "http://www.ontologydesignpatterns.org/",
    "http://www.w3.org/2003/01/geo/",
    "http://www.geonames.org/",
)
TIME_ONTOLOGY = "http://www.w3.org/2006/time"

TermOrigin = Literal["own", "imported"]


def term_origin(iri: str) -> TermOrigin:
    """Whether a reference term belongs to the ontology itself or to an imported vocabulary."""
    return "imported" if iri.startswith(IMPORTED_NAMESPACE_PREFIXES) else "own"


class GoldQuery(BaseModel):
    """A reference SPARQL query and the CQs that link to it."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str
    cq_ids: tuple[str, ...] = ()


def _parse_any(graph: Graph, path: Path) -> None:
    formats = {".ttl": "turtle", ".n3": "n3", ".nt": "nt", ".jsonld": "json-ld"}
    first = formats.get(path.suffix, "xml")
    errors = []
    for fmt in dict.fromkeys([first, "xml", "turtle"]):
        try:
            graph.parse(path, format=fmt)
            return
        except Exception as exc:  # rdflib raises many exception types
            errors.append(f"{fmt}: {exc}")
    raise ValueError(f"cannot parse {path}: {'; '.join(errors)}")


class ReferenceOntology:
    """All ``dm/`` files of a project merged into one graph, with import tagging."""

    def __init__(
        self,
        entry: CatalogEntry,
        graph: Graph,
        loaded_files: tuple[Path, ...],
        parse_errors: tuple[str, ...],
    ) -> None:
        self.entry = entry
        self.graph = graph
        self.loaded_files = loaded_files
        self.parse_errors = parse_errors
        self._signature: Signature | None = None

    @classmethod
    def load(cls, entry: CatalogEntry, shared_files: Iterable[Path] = ()) -> ReferenceOntology:
        """Merge every ``dm/`` file; add ``data/shared`` files that are imported (e.g. Time)."""
        graph = Graph()
        loaded: list[Path] = []
        errors: list[str] = []
        for path in entry.dm_files:
            try:
                _parse_any(graph, path)
                loaded.append(path)
            except ValueError as exc:
                errors.append(str(exc))
        imports = {str(o).rstrip("#/") for o in graph.objects(None, OWL.imports)}
        for path in shared_files:
            if path.stem == "time" and TIME_ONTOLOGY in imports:
                try:
                    _parse_any(graph, path)
                    loaded.append(path)
                except ValueError as exc:
                    errors.append(str(exc))
        return cls(entry, graph, tuple(loaded), tuple(errors))

    def signature(self) -> Signature:
        """All named classes and properties outside the standard vocabularies."""
        if self._signature is None:
            self._signature = extract_signature(self.graph)
        return self._signature

    def class_iris(self, origin: TermOrigin | None = None) -> frozenset[str]:
        """Class IRIs, optionally filtered by origin."""
        iris = self.signature().class_iris()
        return frozenset(i for i in iris if origin is None or term_origin(i) == origin)

    def property_iris(self, origin: TermOrigin | None = None) -> frozenset[str]:
        """Object/datatype property IRIs (annotation properties excluded)."""
        iris = {p.iri for p in self.signature().properties if p.kind != "annotation"}
        return frozenset(i for i in iris if origin is None or term_origin(i) == origin)

    def gold_queries(self) -> list[GoldQuery]:
        """Reference queries under ``rq/`` with the CQ ids that reference them."""
        rq_dir = self.entry.directory / "rq"
        if not rq_dir.exists():
            return []
        links = self._query_links()
        return [
            GoldQuery(
                name=p.name,
                text=p.read_text(encoding="utf-8", errors="replace"),
                cq_ids=tuple(links.get(p.name, ())),
            )
            for p in sorted(rq_dir.iterdir())
            if p.suffix in QUERY_SUFFIXES
        ]

    def _query_links(self) -> dict[str, list[str]]:
        # The rq field is read here, on the evaluation side, and nowhere else.
        from ontoagents.data.loader import YamlCQLoader

        path = self.entry.directory / "cqs" / "cqs.yml"
        if not path.exists():
            return {}
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records = (payload.get("ontology") or {}).get("cqs") or []
        cqs = YamlCQLoader.from_records(self.entry.name, records)
        by_question = {cq.question: cq.id for cq in cqs}
        from ontoagents.data.loader import normalise_question

        links: dict[str, list[str]] = {}
        for record in records:
            cq_id = by_question.get(normalise_question(str(record.get("question") or "")))
            for name in record.get("rq") or []:
                if cq_id:
                    links.setdefault(str(name), []).append(cq_id)
        return links

    def cq_relevant_terms(self) -> frozenset[str]:
        """Reference classes/properties mentioned by at least one gold query."""
        known = self.class_iris() | self.property_iris()
        found: set[str] = set()
        for query in self.gold_queries():
            found.update(i for i in query_iris(query.text) if i in known)
        return frozenset(found)


_IRI_RE = re.compile(r"<([^<>\s]+)>")
_TEMPLATE_SLOT_RE = re.compile(r"\$\??(\w+)\$")
_SELECT_CLAUSE_RE = re.compile(r"(?is)(SELECT\s+)(.*?)(\s*(?:FROM|WHERE)\b|\s*\{)")
_BARE_AGGREGATE_RE = re.compile(
    r"(?i)(?<!\()\b((?:SAMPLE|COUNT|MIN|MAX|SUM|AVG|GROUP_CONCAT)\s*\([^()]*\)\s+AS\s+\?\w+)"
)
_STANDARD_SPARQL_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "schema": "http://schema.org/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


def sanitise_gold_query(text: str) -> str:
    """Make dataset queries parseable: ``$slot$`` template variables become ``?slot`` and
    missing declarations of the standard prefixes are added."""
    text = _TEMPLATE_SLOT_RE.sub(lambda m: f"?{m.group(1)}", text)
    # Virtuoso dialect: comma-separated projections and unparenthesised aggregates.
    text = _SELECT_CLAUSE_RE.sub(
        lambda m: (
            m.group(1) + _BARE_AGGREGATE_RE.sub(r"(\1)", m.group(2).replace(",", " ")) + m.group(3)
        ),
        text,
        count=1,
    )
    declared = {m.lower() for m in re.findall(r"(?im)^\s*prefix\s+(\w*):", text)}
    header = "".join(
        f"PREFIX {p}: <{ns}>\n" for p, ns in _STANDARD_SPARQL_PREFIXES.items() if p not in declared
    )
    return header + text


def query_iris(text: str) -> set[str]:
    """IRIs used by a SPARQL query (prefixes expanded). Falls back to regex on parse errors."""
    try:
        query = prepareQuery(sanitise_gold_query(text))
    except Exception:  # unparseable vendor-specific SPARQL
        return set(_IRI_RE.findall(text))
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, URIRef):
            found.add(str(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list | tuple | set):
            for value in node:
                walk(value)
        elif hasattr(node, "__dict__"):
            for value in vars(node).values():
                walk(value)

    walk(query.algebra)
    return found
