"""Validator abstraction and the context every validator receives."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from ontoagents.domain import CQStatus, OntologyDraft, ValidationReport


class ValidationContext(BaseModel):
    """What a validator may inspect for one generation attempt."""

    model_config = ConfigDict(frozen=True)

    base_iri: str
    candidate: OntologyDraft | None = Field(
        default=None, description="Committed draft merged with the fragment (None if merge failed)."
    )
    fragment: str | None = Field(default=None, description="Raw Turtle emitted by the generator.")
    batch_sparql: dict[str, str] = Field(
        default_factory=dict, description="CQ id -> SPARQL for the current batch."
    )
    cached_sparql: dict[str, str] = Field(
        default_factory=dict, description="CQ id -> SPARQL of earlier committed batches."
    )
    previous_status: dict[str, CQStatus] = Field(default_factory=dict)


class Validator(ABC):
    """A deterministic check on a candidate draft."""

    name: str = "validator"

    @abstractmethod
    def validate(self, context: ValidationContext) -> ValidationReport:
        """Check ``context`` and report."""
