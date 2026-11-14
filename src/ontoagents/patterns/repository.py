"""Ontology design pattern catalogue with naive keyword retrieval."""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from ontoagents.domain import CompetencyQuestion, DesignPattern

_TOKEN_RE = re.compile(r"[a-z]+")


class PatternRepository(ABC):
    """Source of design patterns for a batch of CQs."""

    @abstractmethod
    def all(self) -> list[DesignPattern]:
        """Every pattern in the repository."""

    @abstractmethod
    def retrieve(self, cqs: Sequence[CompetencyQuestion], k: int = 3) -> list[DesignPattern]:
        """The ``k`` patterns most relevant to ``cqs``."""


def _tokens(text: str) -> set[str]:
    words = _TOKEN_RE.findall(text.lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}


class StaticPatternRepository(PatternRepository):
    """A fixed catalogue; relevance is keyword overlap with the CQ text."""

    def __init__(self, patterns: Sequence[DesignPattern] | None = None) -> None:
        self._patterns = list(patterns) if patterns is not None else list(DEFAULT_PATTERNS)

    def all(self) -> list[DesignPattern]:
        """Every pattern."""
        return list(self._patterns)

    def retrieve(self, cqs: Sequence[CompetencyQuestion], k: int = 3) -> list[DesignPattern]:
        """Patterns ranked by keyword hits (ties: catalogue order); zero-hit ones dropped."""
        text = _tokens(" ".join(cq.question for cq in cqs))
        scored = []
        for order, pattern in enumerate(self._patterns):
            hits = len(text & {kw.lower() for kw in pattern.keywords})
            if hits:
                scored.append((-hits, order, pattern))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [p for _, _, p in scored[:k]]


_ODP = "http://www.ontologydesignpatterns.org/cp/owl/"

DEFAULT_PATTERNS: tuple[DesignPattern, ...] = (
    DesignPattern(
        name="PartOf",
        intent="Represent part-whole relations between entities.",
        keywords=("part", "component", "consist", "contain", "compose", "made", "member", "whole"),
        turtle=""":hasPart a owl:ObjectProperty , owl:TransitiveProperty ; owl:inverseOf :isPartOf .
:isPartOf a owl:ObjectProperty , owl:TransitiveProperty .""",
        source=_ODP + "partof.owl",
    ),
    DesignPattern(
        name="Participation",
        intent="Relate objects to the events they take part in.",
        keywords=("event", "participate", "involve", "attend", "take", "activity", "performance"),
        turtle=""":Event a owl:Class .
:isParticipantIn a owl:ObjectProperty ; rdfs:range :Event ; owl:inverseOf :hasParticipant .""",
        source=_ODP + "participation.owl",
    ),
    DesignPattern(
        name="AgentRole",
        intent="Separate an agent from the roles it plays (composer, performer, author...).",
        keywords=(
            "role",
            "who",
            "person",
            "author",
            "composer",
            "performer",
            "agent",
            "actor",
            "creator",
            "responsible",
            "played",
        ),
        turtle=""":Agent a owl:Class . :Role a owl:Class .
:hasRole a owl:ObjectProperty ; rdfs:domain :Agent ; rdfs:range :Role .""",
        source=_ODP + "agentrole.owl",
    ),
    DesignPattern(
        name="TimeIndexedSituation",
        intent="Attach a time interval to a situation or relation (when, during, date).",
        keywords=(
            "when",
            "time",
            "date",
            "year",
            "century",
            "period",
            "during",
            "duration",
            "past",
            "start",
            "end",
        ),
        turtle=""":TimeInterval a owl:Class .
:hasTimeInterval a owl:ObjectProperty ; rdfs:range :TimeInterval .
:startDate a owl:DatatypeProperty ; rdfs:domain :TimeInterval ; rdfs:range xsd:dateTime .""",
        source=_ODP + "timeindexedsituation.owl",
    ),
    DesignPattern(
        name="Place",
        intent="Locate entities in space (where, country, city, site).",
        keywords=(
            "where",
            "place",
            "location",
            "country",
            "city",
            "site",
            "region",
            "produced",
            "located",
            "geographic",
        ),
        turtle=""":Place a owl:Class .
:hasLocation a owl:ObjectProperty ; rdfs:range :Place .""",
        source=_ODP + "place.owl",
    ),
    DesignPattern(
        name="Classification",
        intent="Classify entities with concepts from a controlled vocabulary (type, kind, genre).",
        keywords=("type", "kind", "category", "genre", "classify", "sort", "class"),
        turtle=""":Concept a owl:Class .
:isClassifiedBy a owl:ObjectProperty ; rdfs:range :Concept .""",
        source=_ODP + "classification.owl",
    ),
    DesignPattern(
        name="InformationRealization",
        intent="Distinguish information objects (a work) from their physical realisations.",
        keywords=(
            "document",
            "text",
            "record",
            "manuscript",
            "book",
            "score",
            "edition",
            "version",
            "realization",
            "copy",
            "work",
            "expression",
            "publication",
        ),
        turtle=""":InformationObject a owl:Class . :InformationRealization a owl:Class .
:realizes a owl:ObjectProperty ; rdfs:domain :InformationRealization ;
    rdfs:range :InformationObject .""",
        source=_ODP + "informationrealization.owl",
    ),
    DesignPattern(
        name="Collection",
        intent="Group members into collections.",
        keywords=("collection", "set", "group", "list", "corpus", "catalogue", "member"),
        turtle=""":Collection a owl:Class .
:hasMember a owl:ObjectProperty ; rdfs:domain :Collection .""",
        source=_ODP + "collectionentity.owl",
    ),
    DesignPattern(
        name="Description-Measurement",
        intent="Attach measured qualities or values with units to entities.",
        keywords=(
            "value",
            "measure",
            "amount",
            "quantity",
            "number",
            "how",
            "many",
            "much",
            "size",
            "weight",
            "quality",
            "property",
        ),
        turtle=""":Quality a owl:Class .
:hasQuality a owl:ObjectProperty ; rdfs:range :Quality .
:hasValue a owl:DatatypeProperty ; rdfs:domain :Quality ; rdfs:range xsd:decimal .""",
        source=_ODP + "objectrole.owl",
    ),
    DesignPattern(
        name="Sequence",
        intent="Order entities (before, after, next, previous, step).",
        keywords=(
            "before",
            "after",
            "next",
            "previous",
            "order",
            "sequence",
            "step",
            "follow",
            "precede",
        ),
        turtle=""":precedes a owl:ObjectProperty , owl:TransitiveProperty ;
    owl:inverseOf :follows .""",
        source=_ODP + "sequence.owl",
    ),
)
