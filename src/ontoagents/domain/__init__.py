"""Domain models shared by every layer."""

from ontoagents.domain.competency_question import CompetencyQuestion, CQBatch
from ontoagents.domain.generator import (
    GenerationInputs,
    GenerationResult,
    OntologyGenerator,
    TraceEvent,
    Usage,
)
from ontoagents.domain.ontology import (
    ClassInfo,
    OntologyDiff,
    OntologyDraft,
    PropertyInfo,
    Signature,
    TurtleParseError,
    extract_signature,
    local_name,
    parse_turtle,
)
from ontoagents.domain.pattern import DesignPattern
from ontoagents.domain.report import CQCheck, CQStatus, Issue, PipelineReport, ValidationReport

__all__ = [
    "CQBatch",
    "CQCheck",
    "CQStatus",
    "ClassInfo",
    "CompetencyQuestion",
    "DesignPattern",
    "GenerationInputs",
    "GenerationResult",
    "Issue",
    "OntologyDiff",
    "OntologyDraft",
    "OntologyGenerator",
    "PipelineReport",
    "PropertyInfo",
    "Signature",
    "TraceEvent",
    "TurtleParseError",
    "Usage",
    "ValidationReport",
    "extract_signature",
    "local_name",
    "parse_turtle",
]
