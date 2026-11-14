"""CLI-level runs with the offline heuristic LLM, plus the leakage guarantee."""

import json

from ontoagents.baselines.ontogenia import OntogeniaBaseline
from ontoagents.cli import build_generator, main
from ontoagents.config import ExperimentConfig, LLMConfig, SystemConfig
from ontoagents.data.catalog import OntologyCatalog
from ontoagents.data.loader import YamlCQLoader
from ontoagents.domain import CompetencyQuestion, GenerationInputs
from ontoagents.evaluation.runner import ExperimentRunner
from ontoagents.llm.fake import FakeLLMClient
from ontoagents.llm.heuristic import HeuristicLLMClient


def test_cli_run_with_fake_llm_writes_results(tmp_path, data_dir):
    code = main(
        [
            "run",
            "--provider", "fake",
            "--name", "e2e",
            "--ontology", "awo",
            "--ontology", "SILKNOW",
            "--max-cqs", "6",
            "--results-dir", str(tmp_path),
        ]
    )  # fmt: skip
    assert code == 0
    awo = json.loads((tmp_path / "e2e" / "awo.json").read_text())
    m = awo["metrics"]
    assert m["n_cqs"] == 6 and m["has_reference"] == 1
    for key in (
        "class_f1",
        "property_f1",
        "posthoc_covered_rate",
        "gold_query_pass_rate",
        "subclass_agreement",
        "llm_calls",
        "final_consistent",
    ):
        assert key in m, key
    assert not awo["metric_errors"]
    silk = json.loads((tmp_path / "e2e" / "SILKNOW.json").read_text())
    assert silk["metrics"]["has_reference"] == 0 and "class_f1" not in silk["metrics"]
    assert "posthoc_covered_rate" in silk["metrics"]
    assert (tmp_path / "e2e" / "awo.ttl").exists()
    assert "## e2e" in (tmp_path / "e2e" / "REPORT.md").read_text()


def test_prompts_never_contain_reference_or_gold_queries(tmp_path, data_dir):
    """Leakage: no reference IRI, gold query text or rq file name reaches any prompt."""
    recorded: list[str] = []
    heuristic = HeuristicLLMClient()

    def handler(messages):
        recorded.extend(m.content for m in messages)
        return heuristic.complete(messages).text

    llm = FakeLLMClient(handler=handler)
    cfg = ExperimentConfig(
        name="leak",
        data_dir=data_dir,
        results_dir=tmp_path,
        llm=LLMConfig(provider="fake"),
        system=SystemConfig(planner="llm", use_consistency=False),
    )
    catalog = OntologyCatalog(data_dir)
    for name in ("awo", "stuff"):
        entry = catalog.get(name)
        runner = ExperimentRunner(
            generator=build_generator(cfg, llm),
            catalog=catalog,
            loader=YamlCQLoader(data_dir),
            metrics=[],
            run_dir=tmp_path / "leak",
        )
        assert runner.run_one(entry).ok
        blob = "\n".join(recorded)
        assert "meteck.org" not in blob  # reference namespace of both ontologies
        for path in (entry.directory / "rq").iterdir():
            assert path.name not in blob
            body = [ln for ln in path.read_text().splitlines() if "where" in ln.lower()]
            assert not any(len(ln) > 20 and ln in blob for ln in body)


def test_ontogenia_single_pass_and_parse_failure(base_iri):
    cqs = (CompetencyQuestion(id="c1", ontology="t", question="Which animals eat plants?"),)
    inputs = GenerationInputs(ontology="t", base_iri=base_iri, cqs=cqs)
    ok = OntogeniaBaseline(
        FakeLLMClient(["```turtle\n:Animal a owl:Class .\n```"]), mode="single_pass"
    )
    result = ok.generate(inputs)
    assert result.usage.calls == 1 and len(result.draft.signature().classes) == 1
    broken = OntogeniaBaseline(FakeLLMClient(["```turtle\n:Animal a owl:Class\n```"]))
    result = broken.generate(inputs)
    assert result.stop_reason == "unparseable output"
    assert result.draft.signature().classes == ()


def test_ontogenia_cq_by_cq_carries_ontology_forward(base_iri):
    cqs = tuple(
        CompetencyQuestion(id=f"c{i}", ontology="t", question=f"Question {i}?") for i in (1, 2, 3)
    )
    replies = [
        "```turtle\n:Animal a owl:Class .\n```",
        "```turtle\n:Animal a owl:Class . :Plant a owl:Class\n```",  # broken: kept previous
        "```turtle\n:Animal a owl:Class . :Plant a owl:Class . :Lion a owl:Class .\n```",
    ]
    llm = FakeLLMClient(replies)
    result = OntogeniaBaseline(llm).generate(
        GenerationInputs(ontology="t", base_iri=base_iri, cqs=cqs)
    )
    assert result.usage.calls == 3
    assert len(result.draft.signature().classes) == 3
    assert "1/3 steps unparseable" in (result.stop_reason or "")
    second_prompt = llm.calls[1][-1].content
    assert "Ontology built so far" in second_prompt and ":Animal" in second_prompt
    assert "[c2]" in second_prompt and "[c1]" not in second_prompt


def test_description_flag_reaches_both_systems(tmp_path, data_dir):
    seen: dict[str, bool] = {}

    def handler(messages):
        text = "\n".join(m.content for m in messages)
        role = "baseline" if "Ontogenia" in messages[0].content else "agents"
        marker = "Domain description:" in text or "Ontology story:" in text
        seen[role] = seen.get(role, False) or marker
        return HeuristicLLMClient().complete(messages).text

    llm = FakeLLMClient(handler=handler)
    catalog = OntologyCatalog(data_dir)
    for kind in ("agents", "ontogenia"):
        cfg = ExperimentConfig(
            name=kind,
            data_dir=data_dir,
            results_dir=tmp_path,
            use_description=True,
            max_cqs=3,
            llm=LLMConfig(provider="fake"),
            system=SystemConfig(kind=kind, use_consistency=False),  # type: ignore[arg-type]
        )
        runner = ExperimentRunner(
            generator=build_generator(cfg, llm),
            catalog=catalog,
            loader=YamlCQLoader(data_dir),
            metrics=[],
            run_dir=tmp_path / kind,
            use_description=True,
            max_cqs=3,
        )
        assert runner.run_one(catalog.get("noria-o")).ok
    assert seen == {"agents": True, "baseline": True}


def test_config_selects_baseline_mode(data_dir):
    from ontoagents.llm.heuristic import HeuristicLLMClient as H

    for mode, name in (("cq_by_cq", "ontogenia"), ("single_pass", "ontogenia-single")):
        cfg = ExperimentConfig(
            name="x",
            data_dir=data_dir,
            llm=LLMConfig(provider="fake"),
            system=SystemConfig(kind="ontogenia", baseline_mode=mode),  # type: ignore[arg-type]
        )
        generator = build_generator(cfg, H())
        assert generator.name == name == cfg.system.label()
