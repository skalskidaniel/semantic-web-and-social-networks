"""Ontology drafts stored as canonical Turtle text, plus their vocabulary signature."""

from collections.abc import Iterable
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rdflib import OWL, RDF, RDFS, BNode, Graph, URIRef

from ontoagents.domain.vocab import STANDARD_PREFIXES, is_whitelisted, prefix_header

PropertyKind = Literal["object", "datatype", "annotation", "rdf"]

_PROPERTY_TYPES: dict[URIRef, PropertyKind] = {
    OWL.ObjectProperty: "object",
    OWL.DatatypeProperty: "datatype",
    OWL.AnnotationProperty: "annotation",
    RDF.Property: "rdf",
    OWL.TransitiveProperty: "object",
    OWL.SymmetricProperty: "object",
    OWL.FunctionalProperty: "rdf",
    OWL.InverseFunctionalProperty: "object",
    OWL.AsymmetricProperty: "object",
    OWL.ReflexiveProperty: "object",
    OWL.IrreflexiveProperty: "object",
}
_KIND_PRIORITY: dict[PropertyKind, int] = {"object": 3, "datatype": 2, "annotation": 1, "rdf": 0}


class TurtleParseError(ValueError):
    """Raised when Turtle text cannot be parsed."""


@lru_cache(maxsize=256)
def _parse_cached(turtle: str) -> Graph:
    graph = Graph()
    try:
        graph.parse(data=turtle, format="turtle")
    except Exception as exc:  # rdflib raises many exception types
        raise TurtleParseError(str(exc)) from exc
    return graph


def parse_turtle(turtle: str) -> Graph:
    """Parse Turtle into a graph. The result is cached and must be treated as read-only."""
    return _parse_cached(turtle)


def local_name(iri: str) -> str:
    """The fragment after the last ``#`` or ``/``."""
    for sep in ("#", "/"):
        if sep in iri:
            tail = iri.rsplit(sep, 1)[1]
            if tail:
                return tail
    return iri


class ClassInfo(BaseModel):
    """A named class in a signature."""

    model_config = ConfigDict(frozen=True)

    iri: str
    label: str | None = None
    parents: tuple[str, ...] = ()


class PropertyInfo(BaseModel):
    """A named property in a signature, with its declared domains and ranges."""

    model_config = ConfigDict(frozen=True)

    iri: str
    kind: PropertyKind
    label: str | None = None
    domains: tuple[str, ...] = ()
    ranges: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()


class Signature(BaseModel):
    """The vocabulary of an ontology: what a CQ-to-SPARQL translator is allowed to see.

    It deliberately omits axioms other than the class/property hierarchy, labels and
    domain/range declarations.
    """

    model_config = ConfigDict(frozen=True)

    classes: tuple[ClassInfo, ...] = ()
    properties: tuple[PropertyInfo, ...] = ()

    def class_iris(self) -> frozenset[str]:
        """IRIs of all classes."""
        return frozenset(c.iri for c in self.classes)

    def property_iris(self) -> frozenset[str]:
        """IRIs of all properties."""
        return frozenset(p.iri for p in self.properties)

    def get_property(self, iri: str) -> PropertyInfo | None:
        """Look up a property by IRI."""
        return self._property_index().get(iri)

    def superclasses(self, iri: str) -> frozenset[str]:
        """Reflexive-transitive superclasses of ``iri`` (including ``iri`` itself)."""
        parents = {c.iri: c.parents for c in self.classes}
        seen: set[str] = set()
        stack = [iri]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(parents.get(cur, ()))
        return frozenset(seen)

    def _property_index(self) -> dict[str, PropertyInfo]:
        return {p.iri: p for p in self.properties}

    def to_prompt(self, base_iri: str | None = None) -> str:
        """Compact, deterministic text rendering for prompts."""

        def short(iri: str) -> str:
            if base_iri and iri.startswith(base_iri):
                return ":" + iri[len(base_iri) :]
            for prefix, ns in STANDARD_PREFIXES.items():
                if iri.startswith(ns):
                    return f"{prefix}:{iri[len(ns) :]}"
            return f"<{iri}>"

        if not self.classes and not self.properties:
            return "(empty ontology: no classes or properties yet)"
        lines = ["Classes:"]
        for c in sorted(self.classes, key=lambda c: c.iri):
            parent = f" subClassOf {', '.join(short(p) for p in c.parents)}" if c.parents else ""
            label = f' "{c.label}"' if c.label else ""
            lines.append(f"  {short(c.iri)}{label}{parent}")
        lines.append("Properties:")
        for p in sorted(self.properties, key=lambda p: p.iri):
            label = f' "{p.label}"' if p.label else ""
            dom = ", ".join(short(d) for d in p.domains) or "?"
            rng = ", ".join(short(r) for r in p.ranges) or "?"
            lines.append(f"  {short(p.iri)} [{p.kind}]{label} domain={dom} range={rng}")
        return "\n".join(lines)


def _named(nodes: Iterable[object]) -> list[str]:
    return sorted({str(n) for n in nodes if isinstance(n, URIRef)})


def extract_signature(graph: Graph, include: str | None = None) -> Signature:
    """Build a :class:`Signature` from a graph.

    Args:
        graph: The ontology graph.
        include: If given, only terms whose IRI starts with this namespace are kept as
            signature entries (references to other terms may still appear in parents,
            domains and ranges).

    """

    def keep(iri: str) -> bool:
        if include is not None:
            return iri.startswith(include)
        return not is_whitelisted(iri)

    class_nodes: set[str] = set()
    for t in (OWL.Class, RDFS.Class):
        class_nodes.update(_named(graph.subjects(RDF.type, t)))
    for s, o in graph.subject_objects(RDFS.subClassOf):
        class_nodes.update(_named([s, o]))
    prop_kinds: dict[str, PropertyKind] = {}
    for rdf_type, kind in _PROPERTY_TYPES.items():
        for s in _named(graph.subjects(RDF.type, rdf_type)):
            current = prop_kinds.get(s)
            if current is None or _KIND_PRIORITY[kind] > _KIND_PRIORITY[current]:
                prop_kinds[s] = kind
    for pred in (RDFS.domain, RDFS.range, RDFS.subPropertyOf):
        for s in _named(graph.subjects(pred, None)):
            prop_kinds.setdefault(s, "rdf")
    for pred in (RDFS.domain, RDFS.range):
        for o in graph.objects(None, pred):
            if isinstance(o, URIRef) and not str(o).startswith(str(RDFS.Literal)):
                if not str(o).startswith("http://www.w3.org/2001/XMLSchema#"):
                    class_nodes.add(str(o))
    class_nodes -= set(prop_kinds)
    class_nodes.discard(str(OWL.Thing))
    class_nodes.discard(str(OWL.Nothing))

    def label(iri: str) -> str | None:
        value = graph.value(URIRef(iri), RDFS.label)
        return str(value) if value is not None else None

    def union_members(node: object) -> list[str]:
        if isinstance(node, URIRef):
            return [str(node)]
        if isinstance(node, BNode):
            union = graph.value(node, OWL.unionOf)
            if union is not None:
                return _named(graph.items(union))
        return []

    classes = tuple(
        ClassInfo(
            iri=c,
            label=label(c),
            parents=tuple(p for p in _named(graph.objects(URIRef(c), RDFS.subClassOf)) if p != c),
        )
        for c in sorted(class_nodes)
        if keep(c)
    )
    properties = []
    for p, kind in sorted(prop_kinds.items()):
        if not keep(p):
            continue
        node = URIRef(p)
        domains = sorted({m for d in graph.objects(node, RDFS.domain) for m in union_members(d)})
        ranges = sorted({m for r in graph.objects(node, RDFS.range) for m in union_members(r)})
        if kind == "rdf" and any(r.startswith("http://www.w3.org/2001/XMLSchema#") for r in ranges):
            kind = "datatype"
        properties.append(
            PropertyInfo(
                iri=p,
                kind=kind,
                label=label(p),
                domains=tuple(domains),
                ranges=tuple(ranges),
                parents=tuple(_named(graph.objects(node, RDFS.subPropertyOf))),
            )
        )
    return Signature(classes=classes, properties=tuple(properties))


class OntologyDiff(BaseModel):
    """Difference between two drafts."""

    model_config = ConfigDict(frozen=True)

    added_classes: tuple[str, ...] = ()
    removed_classes: tuple[str, ...] = ()
    added_properties: tuple[str, ...] = ()
    removed_properties: tuple[str, ...] = ()
    added_triples: int = 0
    removed_triples: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the two drafts are triple-identical."""
        return self.added_triples == 0 and self.removed_triples == 0


class OntologyDraft(BaseModel):
    """An ontology under construction.

    Canonical Turtle text is the single source of truth; graphs are derived on demand and
    never stored, so drafts are cheap to serialise and checkpoint.
    """

    model_config = ConfigDict(frozen=True)

    base_iri: str = Field(description="Namespace for all terms minted by the generator.")
    turtle: str = Field(description="Canonical Turtle serialisation.")

    @classmethod
    def empty(cls, base_iri: str) -> OntologyDraft:
        """A draft containing only the ontology header."""
        ontology_iri = base_iri.rstrip("#/")
        text = prefix_header(base_iri) + f"<{ontology_iri}> a owl:Ontology .\n"
        return cls.from_turtle(base_iri, text)

    @classmethod
    def from_turtle(cls, base_iri: str, turtle: str) -> OntologyDraft:
        """Parse and canonicalise arbitrary Turtle (prefix header is added if missing)."""
        graph = parse_turtle(prefix_header(base_iri) + turtle)
        return cls(base_iri=base_iri, turtle=_canonical(graph, base_iri))

    def graph(self) -> Graph:
        """The parsed graph (cached, read-only)."""
        return parse_turtle(self.turtle)

    def merge(self, fragment: str) -> OntologyDraft:
        """Return a new draft with ``fragment`` (Turtle, prefixes optional) added.

        Raises:
            TurtleParseError: if the fragment is not valid Turtle.

        """
        fragment_graph = parse_turtle(prefix_header(self.base_iri) + fragment)
        merged = Graph()
        for triple in self.graph():
            merged.add(triple)
        for triple in fragment_graph:
            merged.add(triple)
        return OntologyDraft(base_iri=self.base_iri, turtle=_canonical(merged, self.base_iri))

    def signature(self) -> Signature:
        """Vocabulary of the terms minted in the draft's base namespace."""
        return extract_signature(self.graph(), include=self.base_iri)

    def diff(self, other: OntologyDraft) -> OntologyDiff:
        """What ``other`` adds to / removes from ``self``."""
        mine, theirs = self.graph(), other.graph()
        added = sum(1 for t in theirs if t not in mine)
        removed = sum(1 for t in mine if t not in theirs)
        a, b = self.signature(), other.signature()
        return OntologyDiff(
            added_classes=tuple(sorted(b.class_iris() - a.class_iris())),
            removed_classes=tuple(sorted(a.class_iris() - b.class_iris())),
            added_properties=tuple(sorted(b.property_iris() - a.property_iris())),
            removed_properties=tuple(sorted(a.property_iris() - b.property_iris())),
            added_triples=added,
            removed_triples=removed,
        )

    def __len__(self) -> int:
        return len(self.graph())


def _canonical(graph: Graph, base_iri: str) -> str:
    out = Graph()
    for prefix, ns in STANDARD_PREFIXES.items():
        out.bind(prefix, ns, override=True)
    out.bind("", base_iri, override=True)
    for triple in graph:
        out.add(triple)
    return out.serialize(format="turtle")
