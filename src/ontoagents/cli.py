"""Command line interface and composition root: the only place concrete classes are wired."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ontoagents.agents.cq_translator import CQToSparqlAgent
from ontoagents.agents.critic import CriticRepairAgent
from ontoagents.agents.generator import GeneratorAgent
from ontoagents.agents.planner import GroupPlanner, LLMPlanner, Planner
from ontoagents.baselines.ontogenia import OntogeniaBaseline
from ontoagents.config import ExperimentConfig, LLMConfig, SystemConfig
from ontoagents.data.catalog import OntologyCatalog
from ontoagents.data.loader import YamlCQLoader
from ontoagents.domain import OntologyGenerator
from ontoagents.evaluation.metrics import (
    CostMetric,
    InternalCQCoverageMetric,
    Metric,
    OntologyStatsMetric,
    PostHocCQCoverageMetric,
    ReferenceSPARQLMetric,
    StructuralAgreementMetric,
    TermAlignmentMetric,
)
from ontoagents.evaluation.report import load_results, render_markdown
from ontoagents.evaluation.runner import ExperimentRunner, RunRecord
from ontoagents.graph.budget import BudgetPolicy
from ontoagents.graph.builder import GraphNodes, OntologyGraphBuilder, SystemFlags
from ontoagents.graph.system import AgentSystem
from ontoagents.llm.cache import CachingLLMClient
from ontoagents.llm.client import LLMClient
from ontoagents.llm.heuristic import HeuristicLLMClient
from ontoagents.patterns.repository import StaticPatternRepository
from ontoagents.validation.base import Validator
from ontoagents.validation.consistency import ConsistencyValidator
from ontoagents.validation.cq_coverage import CQCoverageValidator
from ontoagents.validation.pipeline import ValidatorPipeline
from ontoagents.validation.regression import RegressionValidator
from ontoagents.validation.syntax import SyntaxValidator


def build_llm(cfg: LLMConfig) -> LLMClient:
    """Concrete LLM client, optionally behind the disk cache."""
    client: LLMClient
    if cfg.provider == "fake":
        client = HeuristicLLMClient()
    elif cfg.provider == "claude-code":
        from ontoagents.llm.claude_code import ClaudeCodeClient

        client = ClaudeCodeClient(cfg.model, timeout_s=cfg.timeout_s)
    else:
        from ontoagents.llm.langchain_client import build_langchain_client

        client = build_langchain_client(
            cfg.provider,
            cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )
    if cfg.cache_dir is not None and cfg.provider != "fake":
        client = CachingLLMClient(client, cfg.cache_dir, namespace=f"seed={cfg.seed}")
    return client


def build_pipeline(cfg: SystemConfig) -> ValidatorPipeline:
    """Validators cheapest first, per the ablation flags."""
    validators: list[Validator] = [SyntaxValidator()]
    if cfg.use_consistency:
        validators.append(ConsistencyValidator())
    if cfg.use_translator:
        validators.append(CQCoverageValidator())
        if cfg.use_regression:
            validators.append(RegressionValidator())
    return ValidatorPipeline(validators)


def build_generator(cfg: ExperimentConfig, llm: LLMClient) -> OntologyGenerator:
    """The system under test."""
    s = cfg.system
    temperature, max_tokens = cfg.llm.temperature, cfg.llm.max_tokens
    patterns = StaticPatternRepository() if s.use_patterns else None
    if s.kind == "ontogenia":
        return OntogeniaBaseline(
            llm,
            patterns=patterns,
            pattern_k=s.pattern_k,
            mode=s.baseline_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    planner: Planner = (
        LLMPlanner(llm, batch_size=s.batch_size, temperature=temperature, max_tokens=max_tokens)
        if s.planner == "llm"
        else GroupPlanner(s.batch_size)
    )
    flags = SystemFlags(
        use_translator=s.use_translator,
        use_critic=s.use_critic,
        use_patterns=s.use_patterns,
        pattern_k=s.pattern_k,
        retry_failed=s.retry_failed,
    )
    nodes = GraphNodes(
        planner=planner,
        generator=GeneratorAgent(llm, temperature=temperature, max_tokens=max_tokens),
        pipeline=build_pipeline(s),
        flags=flags,
        translator=CQToSparqlAgent(llm, temperature=temperature, max_tokens=max_tokens)
        if s.use_translator
        else None,
        critic=CriticRepairAgent(llm, temperature=temperature, max_tokens=max_tokens)
        if s.use_critic
        else None,
        patterns=patterns,
    )
    policy = BudgetPolicy(
        max_repair=s.max_repair,
        max_llm_calls=s.max_llm_calls,
        max_tokens=s.max_tokens,
        max_iterations=s.max_iterations,
    )
    return AgentSystem(OntologyGraphBuilder(nodes, policy, flags), name=s.label())


def build_metrics(cfg: ExperimentConfig, llm: LLMClient) -> list[Metric]:
    """Metric suite. The post-hoc translator is the same for every system."""
    metrics: list[Metric] = [
        TermAlignmentMetric("class"),
        TermAlignmentMetric("property"),
        StructuralAgreementMetric(),
        ReferenceSPARQLMetric(),
        InternalCQCoverageMetric(),
        OntologyStatsMetric(ConsistencyValidator() if cfg.evaluation.final_consistency else None),
        CostMetric(),
    ]
    if cfg.evaluation.posthoc_coverage:
        translator = CQToSparqlAgent(
            llm, temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens
        )
        metrics.append(
            PostHocCQCoverageMetric(translator, batch_size=cfg.evaluation.posthoc_batch_size)
        )
    return metrics


def run_ontology(config_json: str, ontology: str) -> RunRecord:
    """Run one ontology (top-level so it can execute in a worker process)."""
    cfg = ExperimentConfig.model_validate_json(config_json)
    llm = build_llm(cfg.llm)
    catalog = OntologyCatalog(cfg.data_dir)
    runner = ExperimentRunner(
        generator=build_generator(cfg, llm),
        catalog=catalog,
        loader=YamlCQLoader(cfg.data_dir),
        metrics=build_metrics(cfg, llm),
        run_dir=cfg.run_dir,
        use_description=cfg.use_description,
        max_cqs=cfg.max_cqs,
        threshold=cfg.evaluation.threshold,
        config=json.loads(config_json),
    )
    return runner.run_one(catalog.get(ontology))


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    update: dict[str, Any] = {}
    if args.name:
        update["name"] = args.name
    if args.ontology:
        update["ontologies"] = args.ontology
    if args.max_cqs is not None:
        update["max_cqs"] = args.max_cqs
    if args.results_dir:
        update["results_dir"] = Path(args.results_dir)
    llm = cfg.llm
    if args.provider:
        llm = llm.model_copy(update={"provider": args.provider})
    if args.model:
        llm = llm.model_copy(update={"model": args.model})
    update["llm"] = llm
    if args.system:
        update["system"] = cfg.system.model_copy(update={"kind": args.system})
    return ExperimentConfig.model_validate({**cfg.model_dump(), **update})


def cmd_run(args: argparse.Namespace) -> int:
    """``ontoagents run``."""
    cfg = (
        ExperimentConfig.from_yaml(Path(args.config))
        if args.config
        else ExperimentConfig(name="adhoc")
    )
    cfg = _apply_overrides(cfg, args)
    catalog = OntologyCatalog(cfg.data_dir)
    names = [e.name for e in catalog.select(cfg.ontologies)]
    config_json = cfg.model_dump_json()
    print(
        f"[ontoagents] experiment={cfg.name} system={cfg.system.label()} "
        f"llm={cfg.llm.provider}:{cfg.llm.model} ontologies={names}",
        file=sys.stderr,
    )
    records: list[RunRecord] = []
    if args.workers > 1 and len(names) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_ontology, config_json, n): n for n in names}
            for future in as_completed(futures):
                records.append(future.result())
                _print_record(records[-1])
    else:
        for name in names:
            records.append(run_ontology(config_json, name))
            _print_record(records[-1])
    (cfg.run_dir / "REPORT.md").write_text(
        render_markdown([r for r in load_results(cfg.results_dir) if r["experiment"] == cfg.name]),
        encoding="utf-8",
    )
    return 0 if all(r.ok for r in records) else 1


def _print_record(r: RunRecord) -> None:
    if not r.ok:
        print(f"  {r.ontology:10s} ERROR {(r.error or '').splitlines()[0]}", file=sys.stderr)
        return
    m = r.metrics
    keys = ("class_f1", "property_f1", "posthoc_covered_rate", "llm_calls")
    summary = " ".join(
        f"{k}={m[k]:.3f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}" for k in keys
    )
    print(f"  {r.ontology:10s} {summary}", file=sys.stderr)


def cmd_report(args: argparse.Namespace) -> int:
    """``ontoagents report``."""
    results_dir = Path(args.results_dir)
    text = render_markdown(load_results(results_dir))
    out = Path(args.out) if args.out else results_dir / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    """``ontoagents catalog``: dataset inventory."""
    from ontoagents.data.reference import ReferenceOntology

    data_dir = Path(args.data_dir)
    catalog = OntologyCatalog(data_dir)
    loader = YamlCQLoader(data_dir)
    print(
        f"{'ontology':12s} {'CQs':>5s} {'ref':>4s} {'cls own/imp':>12s} {'props own/imp':>14s} "
        f"{'gold rq':>8s}"
    )
    for entry in catalog.select("all"):
        n = len(loader.load(entry.name))
        if entry.has_reference:
            ref = ReferenceOntology.load(entry, catalog.shared_files())
            cls = f"{len(ref.class_iris('own'))}/{len(ref.class_iris('imported'))}"
            props = f"{len(ref.property_iris('own'))}/{len(ref.property_iris('imported'))}"
            gold = str(len(ref.gold_queries()))
        else:
            cls = props = gold = "-"
        print(
            f"{entry.name:12s} {n:5d} {'yes' if entry.has_reference else 'no':>4s} "
            f"{cls:>12s} {props:>14s} {gold:>8s}"
        )
    return 0


def make_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    parser = argparse.ArgumentParser(prog="ontoagents", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate + evaluate")
    run.add_argument("--config", help="experiment YAML (default: built-in defaults)")
    run.add_argument("--name", help="override experiment name")
    run.add_argument("--ontology", action="append", help="restrict to ontology (repeatable)")
    run.add_argument("--system", choices=["agents", "ontogenia"], help="override system.kind")
    run.add_argument("--provider", choices=["claude-code", "anthropic", "openai", "fake"])
    run.add_argument("--model", help="override llm.model")
    run.add_argument("--max-cqs", type=int, dest="max_cqs")
    run.add_argument("--results-dir", dest="results_dir")
    run.add_argument("--workers", type=int, default=1, help="ontologies run in parallel")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="aggregate result JSONs into Markdown")
    report.add_argument("--results-dir", default="experiments/results", dest="results_dir")
    report.add_argument("--out")
    report.set_defaults(func=cmd_report)

    cat = sub.add_parser("catalog", help="dataset inventory")
    cat.add_argument("--data-dir", default="data", dest="data_dir")
    cat.set_defaults(func=cmd_catalog)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = make_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
