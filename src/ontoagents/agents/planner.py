"""Planners decide batch composition and order."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ontoagents.agents.base import LLMAgent, PromptRenderer, extract_json
from ontoagents.domain import CompetencyQuestion, CQBatch, Usage
from ontoagents.llm.client import LLMClient, system, user


class Plan(BaseModel):
    """Ordered batches plus planning cost and notes."""

    model_config = ConfigDict(frozen=True)

    batches: tuple[CQBatch, ...]
    usage: Usage = Usage()
    notes: str | None = None


class Planner(ABC):
    """Splits the CQs into ordered batches."""

    name: str = "planner"

    @abstractmethod
    def plan(self, cqs: Sequence[CompetencyQuestion], description: str | None = None) -> Plan:
        """Return the batches to process, covering every CQ exactly once."""


def _chunk(cqs: Sequence[CompetencyQuestion], size: int) -> list[list[CompetencyQuestion]]:
    return [list(cqs[i : i + size]) for i in range(0, len(cqs), size)]


class GroupPlanner(Planner):
    """Deterministic: keeps dataset order, never mixes CQ groups, chunks to ``batch_size``."""

    name = "group"

    def __init__(self, batch_size: int = 5) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size

    def plan(self, cqs: Sequence[CompetencyQuestion], description: str | None = None) -> Plan:
        """Group by the ``group`` field (first-appearance order), then chunk."""
        groups: dict[str | None, list[CompetencyQuestion]] = {}
        for cq in cqs:
            groups.setdefault(cq.group, []).append(cq)
        batches = []
        for group, members in groups.items():
            for chunk in _chunk(members, self._batch_size):
                batches.append(
                    CQBatch(id=f"b{len(batches) + 1:03d}", cqs=tuple(chunk), rationale=group)
                )
        return Plan(batches=tuple(batches))


class LLMPlanner(LLMAgent, Planner):
    """Asks the LLM for thematic batches; falls back to :class:`GroupPlanner` on bad output."""

    name = "llm"

    def __init__(
        self,
        llm: LLMClient,
        renderer: PromptRenderer | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        batch_size: int = 5,
    ) -> None:
        super().__init__(llm, renderer, temperature=temperature, max_tokens=max_tokens)
        self._batch_size = batch_size
        self._fallback = GroupPlanner(batch_size)

    def plan(self, cqs: Sequence[CompetencyQuestion], description: str | None = None) -> Plan:
        """Validate that the LLM plan is a partition of ``cqs`` with small batches."""
        messages = [
            system(self._renderer.render("planner_system.j2", batch_size=self._batch_size)),
            user(self._renderer.render("planner_user.j2", cqs=list(cqs), description=description)),
        ]
        response, usage = self._call(messages)
        by_id = {cq.id: cq for cq in cqs}
        try:
            payload = extract_json(response.text)
            raw_batches = payload["batches"]
            batches: list[CQBatch] = []
            seen: set[str] = set()
            for item in raw_batches:
                ids = [str(i) for i in item["cq_ids"]]
                ids = [i for i in ids if i in by_id and i not in seen]
                seen.update(ids)
                for chunk in _chunk([by_id[i] for i in ids], self._batch_size):
                    batches.append(
                        CQBatch(
                            id=f"b{len(batches) + 1:03d}",
                            cqs=tuple(chunk),
                            rationale=str(item.get("rationale") or "") or None,
                        )
                    )
            missing = [cq for cq in cqs if cq.id not in seen]
            notes = None
            if missing:
                notes = f"LLM plan omitted {len(missing)} CQs; appended in dataset order"
                for chunk in _chunk(missing, self._batch_size):
                    batches.append(CQBatch(id=f"b{len(batches) + 1:03d}", cqs=tuple(chunk)))
            if not batches:
                raise ValueError("empty plan")
            return Plan(batches=tuple(batches), usage=usage, notes=notes)
        except (ValueError, KeyError, TypeError) as exc:
            fallback = self._fallback.plan(cqs, description)
            return Plan(
                batches=fallback.batches,
                usage=usage,
                notes=f"LLM plan unusable ({exc}); fell back to group planner",
            )
