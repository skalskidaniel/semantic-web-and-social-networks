"""Runs a generator over a catalog selection and writes one result JSON per ontology."""

import json
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ontoagents.data.catalog import CatalogEntry, OntologyCatalog
from ontoagents.data.loader import CQLoader, subsample
from ontoagents.data.reference import ReferenceOntology
from ontoagents.domain import GenerationInputs, OntologyGenerator
from ontoagents.evaluation.alignment import Aligner, LexicalAligner
from ontoagents.evaluation.metrics import EvaluationContext, Metric, MetricValues


class RunRecord(BaseModel):
    """Summary of one (generator, ontology) run."""

    model_config = ConfigDict(frozen=True)

    ontology: str
    system: str
    ok: bool
    metrics: MetricValues = {}
    error: str | None = None
    result_path: Path | None = None


class ExperimentRunner:
    """Generation first, evaluation strictly afterwards (the reference is loaded only then)."""

    def __init__(
        self,
        generator: OntologyGenerator,
        catalog: OntologyCatalog,
        loader: CQLoader,
        metrics: Sequence[Metric],
        run_dir: Path,
        *,
        use_description: bool = False,
        max_cqs: int | None = None,
        aligner: Aligner | None = None,
        threshold: float = 0.85,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._generator = generator
        self._catalog = catalog
        self._loader = loader
        self._metrics = list(metrics)
        self._run_dir = run_dir
        self._use_description = use_description
        self._max_cqs = max_cqs
        self._aligner = aligner or LexicalAligner()
        self._threshold = threshold
        self._config = config or {}

    def run(self, entries: Sequence[CatalogEntry]) -> list[RunRecord]:
        """Run every entry; failures are recorded, not raised."""
        return [self.run_one(e) for e in entries]

    def run_one(self, entry: CatalogEntry) -> RunRecord:
        """Generate, evaluate, write ``<run_dir>/<ontology>.json`` and ``.ttl``."""
        self._run_dir.mkdir(parents=True, exist_ok=True)
        out_json = self._run_dir / f"{entry.name}.json"
        started = time.time()
        try:
            cqs = subsample(self._loader.load(entry.name), self._max_cqs)
            inputs = GenerationInputs(
                ontology=entry.name,
                base_iri=entry.base_iri,
                cqs=tuple(cqs),
                description=entry.description() if self._use_description else None,
            )
            result = self._generator.generate(inputs)
            # --- evaluation boundary: nothing below feeds back into generation ---
            reference = (
                ReferenceOntology.load(entry, self._catalog.shared_files())
                if entry.has_reference
                else None
            )
            ctx = EvaluationContext(
                ontology=entry.name,
                cqs=cqs,
                result=result,
                reference=reference,
                aligner=self._aligner,
                threshold=self._threshold,
            )
            values: MetricValues = {"n_cqs": len(cqs), "has_reference": int(reference is not None)}
            metric_errors: dict[str, str] = {}
            for metric in self._metrics:
                if metric.needs_reference and reference is None:
                    continue
                try:
                    values.update(metric.compute(ctx))
                except Exception as exc:  # one broken metric must not lose the run
                    metric_errors[metric.name] = f"{type(exc).__name__}: {exc}"
            alignment = (
                {
                    kind: [
                        {"generated": m.generated, "reference": m.reference, "score": m.score}
                        for m in ctx.alignment(kind).matches
                    ]
                    for kind in ("class", "property")
                }
                if reference is not None
                else {}
            )
            (self._run_dir / f"{entry.name}.ttl").write_text(result.draft.turtle, encoding="utf-8")
            payload = {
                "ontology": entry.name,
                "system": self._generator.name,
                "config": self._config,
                "started_at": started,
                "metrics": values,
                "metric_errors": metric_errors,
                "stop_reason": result.stop_reason,
                "failed_cqs": list(result.failed_cqs),
                "cq_status": {k: v.value for k, v in result.cq_status.items()},
                "sparql": result.sparql,
                "alignment": alignment,
                "trace": [e.model_dump(mode="json") for e in result.trace],
            }
            out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return RunRecord(
                ontology=entry.name,
                system=self._generator.name,
                ok=True,
                metrics=values,
                result_path=out_json,
            )
        except Exception as exc:  # record and continue with the next ontology
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            out_json.write_text(
                json.dumps(
                    {
                        "ontology": entry.name,
                        "system": self._generator.name,
                        "config": self._config,
                        "error": error,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
            return RunRecord(
                ontology=entry.name, system=self._generator.name, ok=False, error=error
            )
