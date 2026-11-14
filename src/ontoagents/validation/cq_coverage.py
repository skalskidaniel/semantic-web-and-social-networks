"""Structural check of CQ SPARQL against a draft's signature (queries are never executed)."""

import time
from collections.abc import Iterable, Iterator
from typing import Literal as TypingLiteral

from rdflib import OWL, RDF, RDFS, BNode, Literal, URIRef, Variable
from rdflib.paths import Path
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.parserutils import CompValue

from ontoagents.domain import CQCheck, CQStatus, Issue, Signature, ValidationReport
from ontoagents.domain.sparql import UNTRANSLATABLE, ensure_prefixes, is_untranslatable
from ontoagents.domain.vocab import is_whitelisted
from ontoagents.validation.base import ValidationContext, Validator

__all__ = [
    "UNTRANSLATABLE",
    "CQCoverageValidator",
    "StructuralQueryChecker",
    "ensure_prefixes",
    "is_untranslatable",
    "query_triples",
]


_CLASS_LINK = {str(RDFS.subClassOf), str(OWL.equivalentClass), str(OWL.disjointWith)}
_CLASS_OBJECT = {str(OWL.someValuesFrom), str(OWL.allValuesFrom), str(OWL.onClass)}
_PROPERTY_LINK = {str(RDFS.subPropertyOf), str(OWL.inverseOf), str(OWL.equivalentProperty)}
_DOMAIN_RANGE = {str(RDFS.domain), str(RDFS.range)}


def _triples(node: object) -> Iterator[tuple[object, object, object]]:
    if isinstance(node, CompValue):
        if node.name == "BGP":
            yield from node.get("triples") or []
        for value in node.values():
            yield from _triples(value)
    elif isinstance(node, list | tuple):
        for value in node:
            yield from _triples(value)


def query_triples(algebra: object) -> list[tuple[object, object, object]]:
    """All triple patterns of every BGP in a query algebra."""
    return list(_triples(algebra))


def _path_iris(path: object) -> Iterator[URIRef]:
    if isinstance(path, URIRef):
        yield path
    elif isinstance(path, Path):
        for attr in ("args", "path"):
            value = getattr(path, attr, None)
            if isinstance(value, list | tuple):
                for item in value:
                    yield from _path_iris(item)
            elif value is not None:
                yield from _path_iris(value)


class StructuralQueryChecker:
    """Checks one SPARQL query against a signature.

    * every class/property IRI must exist in the signature (and be used in the right role);
    * declared domains/ranges must be compatible with variable types inferred from
      ``rdf:type`` patterns; missing domains/ranges only produce warnings;
    * IRIs in subject/object position that are not used as classes are treated as
      individuals and may be unknown.
    """

    def check(self, cq_id: str, sparql: str, signature: Signature, base_iri: str) -> CQCheck:
        """Return the per-CQ verdict."""
        if is_untranslatable(sparql):
            return CQCheck(cq_id=cq_id, status=CQStatus.UNTRANSLATABLE, sparql=sparql)
        try:
            query = prepareQuery(ensure_prefixes(sparql, base_iri))
        except Exception as exc:  # pyparsing / rdflib errors
            issue = Issue(severity="error", message=f"SPARQL does not parse: {exc}", cq_id=cq_id)
            return CQCheck(cq_id=cq_id, status=CQStatus.FAIL, issues=(issue,), sparql=sparql)
        return self.check_triples(cq_id, query_triples(query.algebra), signature, sparql)

    def check_triples(
        self,
        cq_id: str,
        triples: list[tuple[object, object, object]],
        signature: Signature,
        sparql: str | None = None,
    ) -> CQCheck:
        """Check already-extracted triple patterns (used to check rewritten gold queries)."""
        issues = list(self._check_triples(cq_id, triples, signature))
        if any(i.severity == "error" for i in issues):
            status = CQStatus.FAIL
        elif issues:
            status = CQStatus.WARN
        else:
            status = CQStatus.PASS
        return CQCheck(cq_id=cq_id, status=status, issues=tuple(issues), sparql=sparql)

    def _check_triples(
        self, cq_id: str, triples: list[tuple[object, object, object]], sig: Signature
    ) -> Iterable[Issue]:
        classes = sig.class_iris()
        properties = sig.property_iris()
        seen: set[tuple[str, str]] = set()

        def emit(
            severity: TypingLiteral["error", "warning"], message: str, term: str | None = None
        ) -> Iterator[Issue]:
            key = (severity, message)
            if key not in seen:
                seen.add(key)
                yield Issue(severity=severity, message=message, cq_id=cq_id, term=term)

        def need_class(node: object) -> Iterator[Issue]:
            if not isinstance(node, URIRef) or is_whitelisted(str(node)):
                return
            iri = str(node)
            if iri in classes:
                return
            if iri in properties:
                yield from emit("error", f"{iri} is a property but is used as a class", iri)
            else:
                yield from emit("error", f"class {iri} is not defined in the ontology", iri)

        def need_property(node: object) -> Iterator[Issue]:
            if not isinstance(node, URIRef) or is_whitelisted(str(node)):
                return
            iri = str(node)
            if iri in properties:
                return
            if iri in classes:
                yield from emit("error", f"{iri} is a class but is used as a property", iri)
            else:
                yield from emit("error", f"property {iri} is not defined in the ontology", iri)

        var_types: dict[object, set[str]] = {}
        for s, p, o in triples:
            if p == RDF.type and isinstance(o, URIRef) and str(o) in classes:
                var_types.setdefault(s, set()).add(str(o))

        def compatible(declared: tuple[str, ...], actual: set[str]) -> bool:
            for a in actual:
                ancestors = sig.superclasses(a)
                for d in declared:
                    if d in ancestors or a in sig.superclasses(d):
                        return True
            return False

        for s, p, o in triples:
            if isinstance(p, Path):
                for iri in _path_iris(p):
                    if iri != RDF.type:
                        yield from need_property(iri)
                continue
            if isinstance(p, Variable | BNode) or not isinstance(p, URIRef):
                continue
            pred = str(p)
            if p == RDF.type:
                yield from need_class(o)
            elif pred in _CLASS_LINK:
                yield from need_class(s)
                yield from need_class(o)
            elif pred in _CLASS_OBJECT:
                yield from need_class(o)
            elif p == OWL.onProperty:
                yield from need_property(o)
            elif pred in _DOMAIN_RANGE:
                yield from need_property(s)
                yield from need_class(o)
            elif pred in _PROPERTY_LINK:
                yield from need_property(s)
                yield from need_property(o)
            elif is_whitelisted(pred):
                continue
            else:
                yield from need_property(p)
                info = sig.get_property(pred)
                if info is None or info.kind == "annotation":
                    continue
                name = pred
                if info.domains:
                    types = var_types.get(s, set())
                    if types and not compatible(info.domains, types):
                        yield from emit(
                            "error",
                            f"subject of {name} is typed {sorted(types)} but its domain is "
                            f"{list(info.domains)}",
                            name,
                        )
                else:
                    yield from emit("warning", f"property {name} has no rdfs:domain", name)
                if info.kind == "datatype":
                    if o in var_types:
                        yield from emit(
                            "error",
                            f"datatype property {name} is used with a class-typed object",
                            name,
                        )
                    continue
                if isinstance(o, Literal) and info.kind == "object":
                    yield from emit(
                        "error", f"object property {name} is used with a literal value", name
                    )
                    continue
                if info.ranges:
                    types = var_types.get(o, set())
                    if types and not compatible(info.ranges, types):
                        yield from emit(
                            "error",
                            f"object of {name} is typed {sorted(types)} but its range is "
                            f"{list(info.ranges)}",
                            name,
                        )
                elif info.kind == "object":
                    yield from emit("warning", f"property {name} has no rdfs:range", name)


class CQCoverageValidator(Validator):
    """Structural coverage of the current batch's SPARQL; fails if any CQ fails."""

    name = "cq_coverage"

    def __init__(self, checker: StructuralQueryChecker | None = None) -> None:
        self._checker = checker or StructuralQueryChecker()

    def validate(self, context: ValidationContext) -> ValidationReport:
        """Check every query of the current batch against the candidate signature."""
        start = time.perf_counter()
        if context.candidate is None:
            return ValidationReport(
                validator=self.name,
                passed=False,
                issues=(Issue(severity="error", message="no candidate draft"),),
            )
        signature = context.candidate.signature()
        checks = tuple(
            self._checker.check(cq_id, sparql, signature, context.base_iri)
            for cq_id, sparql in sorted(context.batch_sparql.items())
        )
        issues = tuple(i for c in checks for i in c.issues)
        return ValidationReport(
            validator=self.name,
            passed=not any(c.status is CQStatus.FAIL for c in checks),
            issues=issues,
            cq_checks=checks,
            duration_s=time.perf_counter() - start,
        )
