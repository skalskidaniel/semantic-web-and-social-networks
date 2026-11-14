"""Experiment configuration (YAML files under ``experiments/``)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LLMConfig(BaseModel):
    """Which model to call and how."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["claude-code", "anthropic", "openai", "fake"] = "claude-code"
    model: str = "claude-sonnet-5-5"
    temperature: float | None = Field(
        default=0.0, description="Ignored by the claude-code provider (CLI does not expose it)."
    )
    max_tokens: int | None = 16000
    seed: int | None = Field(
        default=42, description="Forwarded only where the provider supports it."
    )
    cache_dir: Path | None = Path(".cache/llm")
    timeout_s: float = 600.0


class SystemConfig(BaseModel):
    """Generator under test plus its ablation switches."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["agents", "ontogenia"] = "agents"
    planner: Literal["group", "llm"] = "group"
    batch_size: int = Field(default=5, ge=1)
    use_translator: bool = True
    use_critic: bool = True
    use_consistency: bool = True
    use_regression: bool = True
    use_patterns: bool = True
    pattern_k: int = 3
    retry_failed: bool = True
    max_repair: int = Field(default=3, ge=0)
    max_llm_calls: int | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None

    def label(self) -> str:
        """Short identifier encoding the ablation."""
        if self.kind == "ontogenia":
            return "ontogenia"
        parts = ["agents", self.planner]
        for flag, tag in (
            (self.use_translator, "notrans"),
            (self.use_critic, "nocritic"),
            (self.use_regression, "noregr"),
            (self.use_consistency, "nocons"),
            (self.use_patterns, "noodp"),
        ):
            if not flag:
                parts.append(tag)
        if self.max_repair != 3:
            parts.append(f"rep{self.max_repair}")
        return "-".join(parts)


class EvaluationConfig(BaseModel):
    """Evaluation-only settings."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    posthoc_coverage: bool = True
    posthoc_batch_size: int = 10
    final_consistency: bool = True


class ExperimentConfig(BaseModel):
    """One experiment: a system configuration applied to a selection of ontologies."""

    model_config = ConfigDict(extra="forbid")

    name: str
    data_dir: Path = Path("data")
    results_dir: Path = Path("experiments/results")
    ontologies: list[str] | Literal["all", "with_reference"] = "all"
    max_cqs: int | None = Field(default=None, description="Evenly spaced CQ subset per ontology.")
    use_description: bool = Field(
        default=False, description="Give description.txt to the system AND the baseline."
    )
    llm: LLMConfig = LLMConfig()
    system: SystemConfig = SystemConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load a YAML config."""
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    @property
    def run_dir(self) -> Path:
        """Where this experiment writes its results."""
        return self.results_dir / self.name
