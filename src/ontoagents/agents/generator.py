"""Generator agent: emits an incremental Turtle fragment for a batch of CQs."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ontoagents.agents.base import LLMAgent, extract_block
from ontoagents.domain import CompetencyQuestion, DesignPattern, Signature, Usage
from ontoagents.llm.client import system, user


class GeneratorOutput(BaseModel):
    """A fragment plus the raw reply, for tracing."""

    model_config = ConfigDict(frozen=True)

    fragment: str
    raw: str
    usage: Usage


class GeneratorAgent(LLMAgent):
    """Sees the current signature and the batch; never the committed axioms or the reference."""

    def generate(
        self,
        cqs: Sequence[CompetencyQuestion],
        signature: Signature,
        base_iri: str,
        *,
        patterns: Sequence[DesignPattern] = (),
        description: str | None = None,
        feedback: str | None = None,
        previous_fragment: str | None = None,
    ) -> GeneratorOutput:
        """Produce the Turtle fragment for ``cqs`` (optionally repairing a rejected attempt)."""
        messages = [
            system(self._renderer.render("generator_system.j2", base_iri=base_iri)),
            user(
                self._renderer.render(
                    "generator_user.j2",
                    description=description,
                    signature=signature.to_prompt(base_iri),
                    patterns=[p.to_prompt() for p in patterns],
                    cqs=list(cqs),
                    feedback=feedback,
                    previous_fragment=previous_fragment or "",
                )
            ),
        ]
        response, usage = self._call(messages)
        fragment = extract_block(response.text, ("turtle", "ttl", "owl", "rdf"))
        return GeneratorOutput(
            fragment=fragment if fragment is not None else response.text.strip(),
            raw=response.text,
            usage=usage,
        )
