"""Turtle syntax and namespace hygiene."""

import time

from rdflib import RDFS, Graph, URIRef
from rdflib.namespace import DCTERMS, SKOS

from ontoagents.domain import Issue, TurtleParseError, ValidationReport, parse_turtle
from ontoagents.domain.vocab import is_whitelisted, prefix_header
from ontoagents.validation.base import ValidationContext, Validator

#: Predicates whose objects may point anywhere (documentation links).
_FREE_OBJECT_PREDICATES = frozenset(
    str(p)
    for p in (
        RDFS.seeAlso,
        RDFS.isDefinedBy,
        RDFS.comment,
        RDFS.label,
        SKOS.definition,
        SKOS.exactMatch,
        SKOS.closeMatch,
        DCTERMS.source,
        DCTERMS.creator,
        DCTERMS.license,
    )
)


class SyntaxValidator(Validator):
    """rdflib must parse the fragment, and every structural IRI must be in the run's base
    namespace or a whitelisted vocabulary (owl, rdf, rdfs, xsd, skos, dcterms, ODP)."""

    name = "syntax"

    def __init__(self, extra_allowed_namespaces: tuple[str, ...] = ()) -> None:
        self._extra = extra_allowed_namespaces

    def _allowed(self, iri: str, base_iri: str) -> bool:
        return (
            iri.startswith(base_iri)
            or iri == base_iri.rstrip("#/")
            or is_whitelisted(iri)
            or iri.startswith(self._extra)
        )

    def validate(self, context: ValidationContext) -> ValidationReport:
        """Parse the fragment (or the candidate if no fragment) and check namespaces."""
        start = time.perf_counter()
        issues: list[Issue] = []
        graph: Graph | None = None
        if context.fragment is not None:
            try:
                graph = parse_turtle(prefix_header(context.base_iri) + context.fragment)
            except TurtleParseError as exc:
                issues.append(Issue(severity="error", message=f"Turtle does not parse: {exc}"))
        elif context.candidate is not None:
            graph = context.candidate.graph()
        else:
            issues.append(Issue(severity="error", message="nothing to validate"))
        if graph is not None:
            if context.fragment is not None and len(graph) == 0:
                issues.append(Issue(severity="warning", message="fragment contains no triples"))
            foreign: set[str] = set()
            for s, p, o in graph:
                for node in (s, p) if str(p) in _FREE_OBJECT_PREDICATES else (s, p, o):
                    if isinstance(node, URIRef) and not self._allowed(str(node), context.base_iri):
                        foreign.add(str(node))
            for iri in sorted(foreign)[:20]:
                issues.append(
                    Issue(
                        severity="error",
                        message=(
                            f"IRI {iri} is outside the base namespace {context.base_iri} and "
                            "the whitelisted vocabularies; mint it under the base namespace"
                        ),
                        term=iri,
                    )
                )
        passed = not any(i.severity == "error" for i in issues)
        return ValidationReport(
            validator=self.name,
            passed=passed,
            issues=tuple(issues),
            duration_s=time.perf_counter() - start,
        )
