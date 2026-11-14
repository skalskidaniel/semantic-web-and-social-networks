# Ontoagents scaffold: implementation status

This file tracks [the scaffold plan](ontoagents-scaffold.md), which lives on the author's machine and is not in this repository.

| Todo | Status | Where |
|---|---|---|
| tooling | done | `pyproject.toml` (module `ontoagents`; `dev` and `notebooks` groups; ruff with `D` scoped to domain + public ABCs/Protocols; pytest; 3 import-linter contracts), `.pre-commit-config.yaml` (ruff, ruff-format, ty, lint-imports) |
| domain | done | `src/ontoagents/domain/` + `tests/unit/test_domain.py` |
| data | done | `data/loader.py`, `data/catalog.py`, `data/reference.py` + `tests/unit/test_data.py` (leakage, stable IDs, multi-file merge, `time.owl`, SILKNOW without a reference) |
| llm | done | `LLMClient` Protocol, `LangChainClient` (temperature, seed where supported, `usage_metadata`), `ClaudeCodeClient` (headless `claude -p`), `CachingLLMClient`, `FakeLLMClient`, `HeuristicLLMClient` |
| baseline-slice | done, except numbers | `baselines/ontogenia.py`, `evaluation/{alignment,metrics,runner,report}.py`, `ontoagents run --system ontogenia --ontology awo`. **No real-model numbers yet**: the benchmark was deliberately left to the author |
| validation | done | `validation/{syntax,consistency,cq_coverage,regression,pipeline}.py` + `tests/unit/test_validation.py` (HermiT tests marked `reasoner`) |
| agents-graph | done | `agents/*`, `graph/{state,budget,routing,builder,system}.py` |
| integration-test | done | `tests/integration/test_graph_loop.py`: happy path, translator sees only its batch, repair limit then rollback, rollback isolation, regression triggering repair, budget stop, no-Critic, no-translator, retry round, HermiT in the loop |
| ablations | configs done, not run | `experiments/configs/*.yaml`, `experiments/run_all.sh` |

## Deviations from the plan (and why)

1. **The Ontogenia baseline defaults to CQ-by-CQ, not single-pass.** The paper (Sect. "Ontogenia technique") says the model "processes one CQ at a time … models and merges the generated ontology at each step and provides it in the context". A single-pass baseline would be a weaker opponent than the method the comparison names. Single-pass is still available as `baseline_mode: single_pass` (`baseline-single-pass.yaml`).
2. **SPARQL helpers moved to `domain/sparql.py`.** `ensure_prefixes`, `is_untranslatable` and `UNTRANSLATABLE` moved there because `agents` and `validation` are independent siblings in the layer contract, and the translator needs them.
3. **Import tagging uses an explicit namespace list** (`IMPORTED_NAMESPACE_PREFIXES`), not the `owl:imports` graph. DemCare's `exchangemodel.owl` imports its sibling modules, so graph-based tagging would wrongly classify DemCare's own terms as imported.
4. **Extra metrics beyond the plan:**
   - `*_recall_cq_relevant`: recall over the reference terms that the gold SPARQL uses. Without it, recall is dominated by reference size (SWO has 4,000+ classes).
   - `posthoc_*` CQ coverage: the same translator and check run on every system's final ontology, so the baseline gets a coverage number too.
   - `f1@t` threshold sensitivity, ontology hygiene stats, and final HermiT consistency.
5. **`ClaudeCodeClient` was added** so experiments can run without a provider API key. It cannot set temperature or seed; determinism comes from the response cache.
