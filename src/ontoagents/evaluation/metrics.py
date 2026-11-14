"""Metrics. Each returns a flat ``{name: number | None}`` dict; ``None`` means not applicable."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from rdflib import RDFS, URIRef
from rdflib.plugins.sparql import prepareQuery

from ontoagents.agents.cq_translator import CQToSparqlAgent
from ontoagents.data.reference import ReferenceOntology, sanitise_gold_query, term_origin
from ontoagents.domain import (
    CompetencyQuestion,
    CQStatus,
    GenerationResult,
    Signature,
    Usage,
)
from ontoagents.evaluation.alignment import Aligner, Alignment, LexicalAligner, Term
from ontoagents.validation.base import ValidationContext
from ontoagents.validation.consistency import ConsistencyValidator
from ontoagents.validation.cq_coverage import StructuralQueryChecker, query_triples

MetricValues = dict[str, float | int | None]
TermKind = Literal["class", "property"]
SENSITIVITY_THRESHOLDS: tuple[float, ...] = (0.7, 0.8, 0.85, 0.9, 1.0)


def _prf(tp_gen: int, n_gen: int, tp_ref: int, n_ref: int) -> tuple[float, float, float]:
    precision = tp_gen / n_gen if n_gen else 0.0
    recall = tp_ref / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


@dataclass
class EvaluationContext:
    """Everything metrics may look at. Built by the runner *after* generation."""

    ontology: str
    cqs: Sequence[CompetencyQuestion]
    result: GenerationResult
    reference: ReferenceOntology | None
    aligner: Aligner = field(default_factory=LexicalAligner)
    threshold: float = 0.85
    _alignments: dict[TermKind, Alignment] = field(default_factory=dict)

    @property
    def generated_signature(self) -> Signature:
        """Signature of the final draft."""
        return self.result.draft.signature()

    def terms(self, kind: TermKind, side: Literal["generated", "reference"]) -> list[Term]:
        """Alignable terms of one side."""
        if side == "generated":
            sig = self.generated_signature
        else:
            assert self.reference is not None
            sig = self.reference.signature()
        if kind == "class":
            return [Term(c.iri, c.label) for c in sig.classes]
        return [Term(p.iri, p.label) for p in sig.properties if p.kind != "annotation"]

    def alignment(self, kind: TermKind) -> Alignment:
        """Cached alignment of generated to reference terms."""
        if kind not in self._alignments:
            self._alignments[kind] = self.aligner.align(
                self.terms(kind, "generated"),
                self.terms(kind, "reference"),
                is_property=kind == "property",
            )
        return self._alignments[kind]


class Metric(ABC):
    """A named family of numbers computed from an :class:`EvaluationContext`."""

    name: str = "metric"
    needs_reference: bool = False

    @abstractmethod
    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Compute the metric values."""


class TermAlignmentMetric(Metric):
    """Precision/recall/F1 of classes or properties after one-to-one alignment.

    * precision: aligned generated terms / generated terms;
    * recall: aligned reference terms / reference terms, also split into own vs imported
      and restricted to the terms that the gold SPARQL actually uses (``cq_relevant``);
    * ``f1@t``: sensitivity of F1 to the alignment threshold.
    """

    needs_reference = True

    def __init__(self, kind: TermKind) -> None:
        self.kind = kind
        self.name = f"{kind}_alignment"

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Compute the alignment-based scores."""
        assert ctx.reference is not None
        k = self.kind
        gen = ctx.terms(k, "generated")
        ref = ctx.terms(k, "reference")
        ref_iris = {t.iri for t in ref}
        own = {i for i in ref_iris if term_origin(i) == "own"}
        imported = ref_iris - own
        relevant = ctx.reference.cq_relevant_terms() & ref_iris
        alignment = ctx.alignment(k)
        matched_ref = {m.reference for m in alignment.at(ctx.threshold)}
        n_matched = len(matched_ref)
        precision, recall, f1 = _prf(n_matched, len(gen), n_matched, len(ref))
        values: MetricValues = {
            f"{k}_n_generated": len(gen),
            f"{k}_n_reference": len(ref),
            f"{k}_n_reference_own": len(own),
            f"{k}_n_reference_imported": len(imported),
            f"{k}_n_matched": n_matched,
            f"{k}_precision": precision,
            f"{k}_recall": recall,
            f"{k}_f1": f1,
            f"{k}_recall_own": _rate(len(matched_ref & own), len(own)),
            f"{k}_recall_imported": _rate(len(matched_ref & imported), len(imported)),
            f"{k}_n_cq_relevant": len(relevant),
            f"{k}_recall_cq_relevant": _rate(len(matched_ref & relevant), len(relevant)),
        }
        for t in SENSITIVITY_THRESHOLDS:
            n = len(alignment.at(t))
            values[f"{k}_f1@{t:g}"] = _prf(n, len(gen), n, len(ref))[2]
        return values


class StructuralAgreementMetric(Metric):
    """Share of aligned subclass edges and domain/range pairs that also hold in the reference."""

    name = "structural_agreement"
    needs_reference = True

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Compare hierarchy and domain/range after alignment."""
        assert ctx.reference is not None
        cmap = ctx.alignment("class").mapping(ctx.threshold)
        pmap = ctx.alignment("property").mapping(ctx.threshold)
        gen = ctx.generated_signature
        ref = ctx.reference.signature()

        def related(a: str, b: str) -> bool:
            return b in ref.superclasses(a) or a in ref.superclasses(b)

        edges = held = 0
        for c in gen.classes:
            for parent in c.parents:
                if c.iri in cmap and parent in cmap:
                    edges += 1
                    held += int(cmap[parent] in ref.superclasses(cmap[c.iri]))
        pairs = pairs_held = 0
        for p in gen.properties:
            rp = ref.get_property(pmap[p.iri]) if p.iri in pmap else None
            if rp is None:
                continue
            for declared, ref_side in ((p.domains, rp.domains), (p.ranges, rp.ranges)):
                for d in declared:
                    if d in cmap and ref_side:
                        pairs += 1
                        pairs_held += int(any(related(cmap[d], r) for r in ref_side))
        return {
            "subclass_edges_aligned": edges,
            "subclass_agreement": _rate(held, edges),
            "domain_range_pairs_aligned": pairs,
            "domain_range_agreement": _rate(pairs_held, pairs),
        }


class InternalCQCoverageMetric(Metric):
    """The loop's own verdicts (agent system only; ``None`` for single-pass baselines)."""

    name = "internal_cq_coverage"

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Rates of pass/warn/fail/untranslatable/unchecked over all CQs."""
        status = ctx.result.cq_status
        n = len(ctx.cqs)
        if not status:
            return {f"internal_{s.value}_rate": None for s in CQStatus} | {
                "internal_covered_rate": None,
                "internal_failed_cqs": None,
            }
        counts = {s: sum(1 for cq in ctx.cqs if status.get(cq.id) is s) for s in CQStatus}
        values: MetricValues = {f"internal_{s.value}_rate": _rate(counts[s], n) for s in CQStatus}
        values["internal_covered_rate"] = _rate(counts[CQStatus.PASS] + counts[CQStatus.WARN], n)
        values["internal_failed_cqs"] = len(ctx.result.failed_cqs)
        return values


class PostHocCQCoverageMetric(Metric):
    """Translate every CQ against the *final* signature and check it structurally.

    Identical procedure for every generator, so the agent system and the baselines are
    compared on equal terms. Its own LLM usage is reported as ``posthoc_*`` cost.
    """

    name = "posthoc_cq_coverage"

    def __init__(
        self,
        translator: CQToSparqlAgent,
        checker: StructuralQueryChecker | None = None,
        batch_size: int = 10,
    ) -> None:
        self._translator = translator
        self._checker = checker or StructuralQueryChecker()
        self._batch_size = batch_size

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Rates over all CQs."""
        base = ctx.result.draft.base_iri
        sig = ctx.generated_signature
        usage = Usage()
        counts = dict.fromkeys(CQStatus, 0)
        cqs = list(ctx.cqs)
        for i in range(0, len(cqs), self._batch_size):
            batch = cqs[i : i + self._batch_size]
            out = self._translator.translate(batch, sig, base)
            usage = usage + out.usage
            for cq in batch:
                counts[self._checker.check(cq.id, out.queries[cq.id], sig, base).status] += 1
        n = len(cqs)
        translated = n - counts[CQStatus.UNTRANSLATABLE]
        return {
            "posthoc_pass_rate": _rate(counts[CQStatus.PASS], n),
            "posthoc_warn_rate": _rate(counts[CQStatus.WARN], n),
            "posthoc_fail_rate": _rate(counts[CQStatus.FAIL], n),
            "posthoc_untranslatable_rate": _rate(counts[CQStatus.UNTRANSLATABLE], n),
            "posthoc_covered_rate": _rate(counts[CQStatus.PASS] + counts[CQStatus.WARN], n),
            "posthoc_covered_rate_of_translatable": _rate(
                counts[CQStatus.PASS] + counts[CQStatus.WARN], translated
            ),
            "posthoc_llm_calls": usage.calls,
            "posthoc_tokens": usage.total_tokens,
        }


class ReferenceSPARQLMetric(Metric):
    """Gold SPARQL rewritten through the alignment, then structurally checked.

    Secondary metric: it depends on alignment quality. Unparseable gold queries (vendor
    dialects, broken files) are counted and excluded.
    """

    name = "reference_sparql"
    needs_reference = True

    def __init__(self, checker: StructuralQueryChecker | None = None) -> None:
        self._checker = checker or StructuralQueryChecker()

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Share of gold queries that remain structurally valid after rewriting."""
        assert ctx.reference is not None
        mapping = {
            **ctx.alignment("class").inverse(ctx.threshold),
            **ctx.alignment("property").inverse(ctx.threshold),
        }
        known = ctx.reference.class_iris() | ctx.reference.property_iris()
        sig = ctx.generated_signature
        queries = ctx.reference.gold_queries()
        parseable = ok = term_total = term_mapped = 0
        for q in queries:
            try:
                algebra = prepareQuery(sanitise_gold_query(q.text)).algebra
            except Exception:  # vendor dialect or broken file
                continue
            parseable += 1
            triples: list[tuple[object, object, object]] = []
            for triple in query_triples(algebra):
                rewritten = []
                for node in triple:
                    if isinstance(node, URIRef) and str(node) in known:
                        term_total += 1
                        if str(node) in mapping:
                            term_mapped += 1
                            node = URIRef(mapping[str(node)])
                    rewritten.append(node)
                triples.append((rewritten[0], rewritten[1], rewritten[2]))
            check = self._checker.check_triples(q.name, triples, sig)
            ok += int(check.status.is_ok)
        return {
            "gold_queries": len(queries),
            "gold_queries_parseable": parseable,
            "gold_query_pass_rate": _rate(ok, parseable),
            "gold_term_mapped_rate": _rate(term_mapped, term_total),
        }


class OntologyStatsMetric(Metric):
    """Size and hygiene of the generated ontology, plus final consistency."""

    name = "ontology_stats"

    def __init__(self, consistency: ConsistencyValidator | None = None) -> None:
        self._consistency = consistency

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Counts and shares."""
        draft = ctx.result.draft
        sig = ctx.generated_signature
        graph = draft.graph()
        props = [p for p in sig.properties if p.kind != "annotation"]
        labelled = sum(1 for c in sig.classes if c.label) + sum(1 for p in props if p.label)
        commented = sum(
            1 for t in [*sig.classes, *props] if graph.value(URIRef(t.iri), RDFS.comment)
        )
        n_terms = len(sig.classes) + len(props)
        values: MetricValues = {
            "n_triples": len(graph),
            "n_classes": len(sig.classes),
            "n_properties": len(props),
            "n_object_properties": sum(1 for p in props if p.kind == "object"),
            "n_datatype_properties": sum(1 for p in props if p.kind == "datatype"),
            "property_domain_rate": _rate(sum(1 for p in props if p.domains), len(props)),
            "property_range_rate": _rate(sum(1 for p in props if p.ranges), len(props)),
            "label_rate": _rate(labelled, n_terms),
            "comment_rate": _rate(commented, n_terms),
            "root_class_rate": _rate(
                sum(1 for c in sig.classes if not c.parents), len(sig.classes)
            ),
        }
        if self._consistency is not None:
            report = self._consistency.validate(
                ValidationContext(base_iri=draft.base_iri, candidate=draft)
            )
            values["final_consistent"] = None if report.skipped else int(report.passed)
        return values


class CostMetric(Metric):
    """LLM usage and wall time of generation (evaluation cost excluded)."""

    name = "cost"

    def compute(self, ctx: EvaluationContext) -> MetricValues:
        """Usage counters."""
        u = ctx.result.usage
        return {
            "llm_calls": u.calls,
            "cached_llm_calls": u.cached_calls,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "wall_time_s": ctx.result.wall_time_s,
            "generator_iterations": sum(1 for e in ctx.result.trace if e.kind == "generate"),
            "repairs": sum(1 for e in ctx.result.trace if e.kind == "repair"),
            "rollbacks": sum(1 for e in ctx.result.trace if e.kind == "rollback"),
        }
