"""Competency questions: the only requirement input the generation loop may see."""

import re

from pydantic import BaseModel, ConfigDict, Field, computed_field

PLACEHOLDER_RE = re.compile(r"\[[^\]]+\]")


class CompetencyQuestion(BaseModel):
    """A single competency question.

    Deliberately has no field for gold SPARQL: the reference queries are evaluation-only
    data and must never travel with the question into the generation loop.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Stable identifier, unique within an ontology.")
    ontology: str = Field(description="Name of the ontology this CQ belongs to.")
    question: str = Field(description="Natural-language question, placeholders kept verbatim.")
    group: str | None = Field(default=None, description="Thematic group from the dataset.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_placeholders(self) -> bool:
        """Whether the question contains ``[slot]``-style placeholders."""
        return bool(PLACEHOLDER_RE.search(self.question))


class CQBatch(BaseModel):
    """An ordered group of CQs processed together in one generation step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    cqs: tuple[CompetencyQuestion, ...]
    rationale: str | None = None

    @property
    def cq_ids(self) -> tuple[str, ...]:
        """Identifiers of the CQs in the batch, in order."""
        return tuple(cq.id for cq in self.cqs)
