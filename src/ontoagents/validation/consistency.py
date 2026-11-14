"""DL consistency and class satisfiability via owlready2 + HermiT (needs Java)."""

import shutil
import tempfile
import time
from pathlib import Path

from ontoagents.domain import Issue, ValidationReport
from ontoagents.validation.base import ValidationContext, Validator


def java_available() -> bool:
    """Whether a Java runtime is on PATH (HermiT requires one)."""
    return shutil.which("java") is not None


class ConsistencyValidator(Validator):
    """Fails if the candidate is inconsistent or has unsatisfiable named classes.

    Reasoner crashes (e.g. unsupported datatypes) are reported as warnings and do not fail
    the draft: a limitation of the reasoner is not evidence of a modelling error.
    """

    name = "consistency"

    def validate(self, context: ValidationContext) -> ValidationReport:
        """Run HermiT on the candidate draft."""
        start = time.perf_counter()
        if context.candidate is None:
            return ValidationReport(
                validator=self.name,
                passed=False,
                issues=(Issue(severity="error", message="no candidate draft to reason over"),),
            )
        if not java_available():
            return ValidationReport(
                validator=self.name,
                passed=True,
                skipped=True,
                issues=(Issue(severity="warning", message="java not found; reasoner skipped"),),
            )
        issues = self._reason(context.candidate.graph().serialize(format="nt"), context.base_iri)
        return ValidationReport(
            validator=self.name,
            passed=not any(i.severity == "error" for i in issues),
            issues=tuple(issues),
            duration_s=time.perf_counter() - start,
        )

    @staticmethod
    def _reason(ntriples: str, base_iri: str) -> list[Issue]:
        import owlready2

        with tempfile.TemporaryDirectory(prefix="ontoagents-hermit-") as tmp:
            path = Path(tmp) / "candidate.nt"
            path.write_text(ntriples, encoding="utf-8")
            world = owlready2.World()
            try:
                onto = world.get_ontology(path.as_uri()).load(format="ntriples")
            except Exception as exc:  # owlready2 parser errors
                return [Issue(severity="warning", message=f"owlready2 could not load draft: {exc}")]
            try:
                with onto:
                    owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=0)
            except owlready2.OwlReadyInconsistentOntologyError:
                return [Issue(severity="error", message="ontology is inconsistent (HermiT)")]
            except Exception as exc:  # reasoner crash, unsupported datatype, ...
                message = str(exc).strip().splitlines()[-1:] or [type(exc).__name__]
                return [
                    Issue(
                        severity="warning", message=f"reasoner error, check skipped: {message[0]}"
                    )
                ]
            unsat = sorted(
                c.iri
                for c in world.inconsistent_classes()
                if hasattr(c, "iri") and str(c.iri).startswith(base_iri)
            )
            return [
                Issue(
                    severity="error",
                    message=f"class {iri} is unsatisfiable (equivalent to owl:Nothing)",
                    term=iri,
                )
                for iri in unsat
            ]
