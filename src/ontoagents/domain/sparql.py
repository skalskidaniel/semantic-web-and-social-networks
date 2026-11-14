"""SPARQL conventions shared by the translator and the validators."""

import re

from ontoagents.domain.vocab import STANDARD_PREFIXES

#: Marker the translator emits for CQs no SPARQL query could answer.
UNTRANSLATABLE = "UNTRANSLATABLE"

_PREFIX_DECL_RE = re.compile(r"(?im)^\s*prefix\s+(\w*):")


def ensure_prefixes(query: str, base_iri: str) -> str:
    """Add declarations for the standard prefixes and ``:`` when a query omits them."""
    declared = {m.lower() for m in _PREFIX_DECL_RE.findall(query)}
    header = [f"PREFIX {p}: <{ns}>" for p, ns in STANDARD_PREFIXES.items() if p not in declared]
    if "" not in declared:
        header.append(f"PREFIX : <{base_iri}>")
    return "\n".join(header) + "\n" + query if header else query


def is_untranslatable(sparql: str) -> bool:
    """Whether the translator declined the CQ."""
    return sparql.strip().upper().startswith(UNTRANSLATABLE)
