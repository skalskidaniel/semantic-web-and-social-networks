"""Namespaces that every component agrees on."""

from rdflib import OWL, RDF, RDFS, SKOS, XSD
from rdflib.namespace import DCTERMS

#: Vocabularies a generated ontology may use besides its own base namespace.
WHITELISTED_NAMESPACES: tuple[str, ...] = (
    str(RDF),
    str(RDFS),
    str(OWL),
    str(XSD),
    str(SKOS),
    str(DCTERMS),
    "http://purl.org/dc/elements/1.1/",
    "http://www.w3.org/XML/1998/namespace",
    "http://www.ontologydesignpatterns.org/",
)

#: Prefix header prepended to every fragment so that LLM output may omit declarations.
STANDARD_PREFIXES: dict[str, str] = {
    "rdf": str(RDF),
    "rdfs": str(RDFS),
    "owl": str(OWL),
    "xsd": str(XSD),
    "skos": str(SKOS),
    "dcterms": str(DCTERMS),
}


def is_whitelisted(iri: str) -> bool:
    """Return whether ``iri`` belongs to one of the shared vocabularies."""
    return any(iri.startswith(ns) for ns in WHITELISTED_NAMESPACES)


def prefix_header(base_iri: str) -> str:
    """Turtle prefix declarations for the standard vocabularies and ``:`` = ``base_iri``."""
    lines = [f"@prefix {p}: <{ns}> ." for p, ns in STANDARD_PREFIXES.items()]
    lines.append(f"@prefix : <{base_iri}> .")
    return "\n".join(lines) + "\n"
