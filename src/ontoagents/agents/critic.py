"""Critic/Repair agent: turns a validator report into concrete repair instructions."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ontoagents.agents.base import LLMAgent
from ontoagents.domain import CompetencyQuestion, PipelineReport, Signature, Usage
from ontoagents.llm.client import system, user


class CriticOutput(BaseModel):
    """Repair instructions for the generator."""

    model_config = ConfigDict(frozen=True)

    instructions: str
    usage: Usage


class CriticRepairAgent(LLMAgent):
    """Diagnoses validator failures; the Generator applies the fixes."""

    def critique(
        self,
        cqs: Sequence[CompetencyQuestion],
        signature: Signature,
        base_iri: str,
        fragment: str,
        report: PipelineReport,
    ) -> CriticOutput:
        """Return numbered repair instructions for ``fragment``."""
        messages = [
            system(self._renderer.render("critic_system.j2", base_iri=base_iri)),
            user(
                self._renderer.render(
                    "critic_user.j2",
                    cqs=list(cqs),
                    signature=signature.to_prompt(base_iri),
                    fragment=fragment,
                    report=report.to_feedback(),
                )
            ),
        ]
        response, usage = self._call(messages)
        return CriticOutput(instructions=response.text.strip(), usage=usage)


def raw_feedback(report: PipelineReport) -> str:
    """Feedback used when the Critic is ablated: the validator report verbatim."""
    return report.to_feedback()
