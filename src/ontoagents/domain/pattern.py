"""Ontology design patterns (ODPs) offered to generators as modelling guidance."""

from pydantic import BaseModel, ConfigDict


class DesignPattern(BaseModel):
    """A content ontology design pattern."""

    model_config = ConfigDict(frozen=True)

    name: str
    intent: str
    keywords: tuple[str, ...]
    turtle: str
    source: str | None = None

    def to_prompt(self) -> str:
        """Compact rendering for prompts."""
        return f"ODP {self.name}: {self.intent}\n{self.turtle.strip()}"
