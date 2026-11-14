"""Run validators cheapest-first, stopping at the first failure."""

from collections.abc import Sequence

from ontoagents.domain import PipelineReport
from ontoagents.validation.base import ValidationContext, Validator


class ValidatorPipeline:
    """Ordered validators; by default stops at the first failing one."""

    def __init__(self, validators: Sequence[Validator], stop_on_failure: bool = True) -> None:
        self._validators = list(validators)
        self._stop = stop_on_failure

    @property
    def names(self) -> list[str]:
        """Validator names in execution order."""
        return [v.name for v in self._validators]

    def run(self, context: ValidationContext) -> PipelineReport:
        """Validate ``context``."""
        reports = []
        for validator in self._validators:
            report = validator.validate(context)
            reports.append(report)
            if self._stop and not report.passed:
                break
        return PipelineReport(reports=tuple(reports))
