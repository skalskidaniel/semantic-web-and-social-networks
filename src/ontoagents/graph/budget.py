"""Stateless budget limits checked by the routing functions."""

from pydantic import BaseModel, ConfigDict, Field

from ontoagents.graph.state import GraphState


class BudgetPolicy(BaseModel):
    """Limits for one run. ``None`` means unlimited."""

    model_config = ConfigDict(frozen=True)

    max_repair: int = Field(default=3, ge=0, description="Repair attempts per batch.")
    max_llm_calls: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(
        default=None, ge=1, description="Generator invocations in total."
    )

    def exhausted(self, state: GraphState) -> str | None:
        """Reason the global budget is used up, or None."""
        if self.max_llm_calls is not None and state.get("llm_calls", 0) >= self.max_llm_calls:
            return "budget: llm_calls"
        tokens = state.get("input_tokens", 0) + state.get("output_tokens", 0)
        if self.max_tokens is not None and tokens >= self.max_tokens:
            return "budget: tokens"
        if self.max_iterations is not None and state.get("iterations", 0) >= self.max_iterations:
            return "budget: iterations"
        return None

    def can_repair(self, state: GraphState) -> bool:
        """Whether the current batch may be sent back for another repair."""
        return state.get("repair_count", 0) < self.max_repair
