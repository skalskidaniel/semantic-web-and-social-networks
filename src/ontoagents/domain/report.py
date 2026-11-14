"""Validation reports produced by the deterministic validators."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CQStatus(StrEnum):
    """Outcome of the structural check of one CQ's SPARQL against a draft."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNTRANSLATABLE = "untranslatable"
    UNCHECKED = "unchecked"  # translator ablated: nothing was checked

    @property
    def is_ok(self) -> bool:
        """PASS and WARN count as covered; UNTRANSLATABLE/UNCHECKED are neither pass nor fail."""
        return self in (CQStatus.PASS, CQStatus.WARN)


class Issue(BaseModel):
    """A single finding of a validator."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["error", "warning"]
    message: str
    cq_id: str | None = None
    term: str | None = None


class CQCheck(BaseModel):
    """Per-CQ result of the coverage or regression check."""

    model_config = ConfigDict(frozen=True)

    cq_id: str
    status: CQStatus
    issues: tuple[Issue, ...] = ()
    sparql: str | None = None


class ValidationReport(BaseModel):
    """Result of one validator."""

    model_config = ConfigDict(frozen=True)

    validator: str
    passed: bool
    issues: tuple[Issue, ...] = ()
    cq_checks: tuple[CQCheck, ...] = ()
    skipped: bool = False
    duration_s: float = 0.0

    @property
    def errors(self) -> tuple[Issue, ...]:
        """Issues with severity ``error``."""
        return tuple(i for i in self.issues if i.severity == "error")


class PipelineReport(BaseModel):
    """Result of a :class:`ValidatorPipeline` run (stops at the first failure)."""

    model_config = ConfigDict(frozen=True)

    reports: tuple[ValidationReport, ...] = ()

    @property
    def passed(self) -> bool:
        """True iff every validator that ran passed."""
        return all(r.passed for r in self.reports)

    @property
    def failed_validator(self) -> str | None:
        """Name of the first failing validator, if any."""
        return next((r.validator for r in self.reports if not r.passed), None)

    def cq_checks(self, validator: str) -> tuple[CQCheck, ...]:
        """CQ checks of the named validator (empty if it did not run)."""
        for r in self.reports:
            if r.validator == validator:
                return r.cq_checks
        return ()

    def to_feedback(self, max_issues: int = 25) -> str:
        """Human/LLM-readable summary of what went wrong."""
        lines: list[str] = []
        for r in self.reports:
            status = "PASSED" if r.passed else "FAILED"
            if r.skipped:
                status = "SKIPPED"
            lines.append(f"[{r.validator}] {status}")
            shown = 0
            for issue in r.issues:
                if shown >= max_issues:
                    lines.append("  ... (more issues truncated)")
                    break
                where = f" (CQ {issue.cq_id})" if issue.cq_id else ""
                lines.append(f"  - {issue.severity}{where}: {issue.message}")
                shown += 1
            for check in r.cq_checks:
                if check.status is CQStatus.FAIL and check.sparql:
                    lines.append(f"  query for CQ {check.cq_id}:\n{_indent(check.sparql, 6)}")
        return "\n".join(lines)


def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in text.strip().splitlines())
