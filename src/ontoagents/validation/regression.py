"""Re-check cached SPARQL of earlier batches against the new candidate."""

import time

from ontoagents.domain import CQCheck, CQStatus, Issue, ValidationReport
from ontoagents.validation.base import ValidationContext, Validator
from ontoagents.validation.cq_coverage import StructuralQueryChecker


class RegressionValidator(Validator):
    """Fails when a CQ that passed in an earlier committed batch now fails."""

    name = "regression"

    def __init__(self, checker: StructuralQueryChecker | None = None) -> None:
        self._checker = checker or StructuralQueryChecker()

    def validate(self, context: ValidationContext) -> ValidationReport:
        """Run the structural check on every cached query."""
        start = time.perf_counter()
        if context.candidate is None or not context.cached_sparql:
            return ValidationReport(validator=self.name, passed=True, skipped=True)
        signature = context.candidate.signature()
        broken: list[CQCheck] = []
        issues: list[Issue] = []
        for cq_id, sparql in sorted(context.cached_sparql.items()):
            before = context.previous_status.get(cq_id, CQStatus.PASS)
            if not before.is_ok:
                continue
            check = self._checker.check(cq_id, sparql, signature, context.base_iri)
            if check.status is CQStatus.FAIL:
                broken.append(check)
                reasons = "; ".join(i.message for i in check.issues if i.severity == "error")
                issues.append(
                    Issue(
                        severity="error",
                        cq_id=cq_id,
                        message=f"regression: CQ {cq_id} passed earlier but now fails: {reasons}",
                    )
                )
        return ValidationReport(
            validator=self.name,
            passed=not broken,
            issues=tuple(issues),
            cq_checks=tuple(broken),
            duration_s=time.perf_counter() - start,
        )
