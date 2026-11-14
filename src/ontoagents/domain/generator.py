"""The interface shared by the agent system and every baseline."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ontoagents.domain.competency_question import CompetencyQuestion
from ontoagents.domain.ontology import OntologyDraft
from ontoagents.domain.report import CQStatus


class Usage(BaseModel):
    """LLM usage accounting."""

    model_config = ConfigDict(frozen=True)

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            calls=self.calls + other.calls,
            cached_calls=self.cached_calls + other.cached_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


class GenerationInputs(BaseModel):
    """Everything a generator is allowed to see. Identical for the system and baselines."""

    model_config = ConfigDict(frozen=True)

    ontology: str
    base_iri: str
    cqs: tuple[CompetencyQuestion, ...]
    description: str | None = Field(
        default=None, description="Only set when the experiment enables use_description."
    )


class TraceEvent(BaseModel):
    """One step of a generation run."""

    model_config = ConfigDict(frozen=True)

    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Output of a generator: the final draft plus a trace for analysis."""

    model_config = ConfigDict(frozen=True)

    draft: OntologyDraft
    usage: Usage = Usage()
    wall_time_s: float = 0.0
    trace: tuple[TraceEvent, ...] = ()
    sparql: dict[str, str] = Field(
        default_factory=dict, description="Committed CQ id -> SPARQL (agent system only)."
    )
    cq_status: dict[str, CQStatus] = Field(default_factory=dict)
    failed_cqs: tuple[str, ...] = ()
    stop_reason: str | None = None


@runtime_checkable
class OntologyGenerator(Protocol):
    """Anything that turns competency questions into an ontology."""

    @property
    def name(self) -> str:
        """Short identifier used in result files."""
        ...

    def generate(self, inputs: GenerationInputs) -> GenerationResult:
        """Generate an ontology from ``inputs``."""
        ...
