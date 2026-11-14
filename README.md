# ontoagents

A multi-agent ontology generator built with LangGraph. It generates an OWL ontology from **competency questions (CQs)** and measures it against an
**Ontogenia-style single-pass baseline** using alignment-based metrics on the
[llm4ke](https://github.com/D2KLab/llm4ke) dataset (`data/`: 10 ontologies, 831 CQs).

The feedback signal is **executable validation**, not LLM self-critique alone:

1. rdflib parse
2. HermiT DL consistency
3. structural CQ coverage of translator-written SPARQL
4. regression on earlier CQs

```
CQs ──► Planner ──► Generator ──► merge ──► CQ→SPARQL ──► Validators ──green──► commit ──► next batch
                        ▲                                    │red
                        └──────── Critic/Repair ◄────────────┘ (≤ max_repair, else rollback)
```

## Install

```bash
uv sync                      # Python 3.14, dev tools included
uv run pre-commit install    # ruff, ruff-format, ty, import-linter
uv sync --group notebooks    # pandas/matplotlib for notebooks/data_analysis.ipynb
```

HermiT needs a **Java runtime** (`java` on `PATH`). Without one, the consistency validator
reports itself as *skipped*, and tests marked `@pytest.mark.reasoner` are skipped.

## Quick start

```bash
uv run ontoagents catalog                                        # dataset inventory
uv run ontoagents run --config experiments/configs/dry-run.yaml  # offline, fake LLM
uv run pytest                                                    # 89 tests, no network
```

A real run:

```bash
# Option A: Claude Code CLI (no API key needed if `claude` is logged in)
uv run ontoagents run --config experiments/configs/agents-full.yaml --ontology awo
# Option B: LangChain provider
uv sync --extra anthropic && export ANTHROPIC_API_KEY=...
uv run ontoagents run --config experiments/configs/agents-full.yaml --provider anthropic
```

The full benchmark (both baselines, the full system and every ablation, then aggregation). It is expensive: `agents-full` makes about 2–4 calls per batch of 5 CQs, and Polifonia alone has 361 CQs. Start with `--max-cqs 30`, or select ontologies with `--ontology`:

```bash
WORKERS=4 experiments/run_all.sh            # optionally: --max-cqs 30 for a cheaper pass
uv run ontoagents report                    # -> experiments/results/REPORT.md
```

Each run writes the following to `experiments/results/<experiment>/`:

- `<ontology>.json`, which holds:
  - the full config: model, seed, temperature, flags;
  - every metric;
  - the per-CQ status and the committed SPARQL;
  - the term alignment (for manual inspection);
  - the complete trace (plan, every generate / validate / repair / commit / rollback).
- `<ontology>.ttl`, the generated ontology.
- `REPORT.md`.

LLM responses are cached on disk (`.cache/llm`, keyed by transcript, model and sampling settings), so reruns and new metrics cost nothing. The cache keeps the original token counts, so the cost metrics stay truthful.

CLI overrides: `--ontology X` (repeatable), `--system agents|ontogenia`, `--provider`, `--model`, `--max-cqs N` (evenly spaced, deterministic subset), `--workers N` (ontologies run in parallel processes), `--name`, `--results-dir`.

## Package layout

```
src/ontoagents/
  domain/      CompetencyQuestion, CQBatch, OntologyDraft (canonical Turtle text), Signature,
               OntologyDiff, ValidationReport/PipelineReport, DesignPattern, Usage,
               OntologyGenerator Protocol + GenerationInputs/Result, SPARQL conventions
  data/        YamlCQLoader (stable IDs, drops `rq`), OntologyCatalog, ReferenceOntology (eval only)
  llm/         LLMClient Protocol, LangChainClient, ClaudeCodeClient, CachingLLMClient,
               FakeLLMClient (scripted), HeuristicLLMClient (offline dry runs)
  patterns/    PatternRepository ABC, StaticPatternRepository (10 content ODPs, keyword retrieval)
  agents/      GeneratorAgent, CQToSparqlAgent, CriticRepairAgent, GroupPlanner, LLMPlanner,
               prompts/*.j2
  validation/  SyntaxValidator, ConsistencyValidator (owlready2+HermiT), CQCoverageValidator,
               RegressionValidator, ValidatorPipeline
  graph/       GraphState (TypedDict + reducers), BudgetPolicy, pure routing, OntologyGraphBuilder,
               AgentSystem (implements OntologyGenerator)
  baselines/   OntogeniaBaseline (implements OntologyGenerator)
  evaluation/  LexicalAligner (Hungarian), Metric classes, ExperimentRunner, Markdown report
  config.py    Experiment/LLM/System/Evaluation configs (YAML)
  cli.py       composition root: the only place concrete classes are wired together
```

### Design rules (enforced)

- **Leakage boundary.** Inside the loop, the only visible inputs are the CQs and, when `use_description: true`, `description.txt`.
  - `CompetencyQuestion` has no field for gold SPARQL (`extra="forbid"`).
  - The loader drops `rq:`.
  - `data.reference` (the reference ontology and gold queries) is loaded by `ExperimentRunner` only after `generate()` returns.
  - An integration test records every prompt of a run and asserts that no reference namespace, `rq` file name or gold query line appears in them.
- **Same inputs for system and baseline.** Both receive the same `GenerationInputs`, with the same description flag and the same ODP repository. A test checks the description flag.
- **Import contracts** (`uv run lint-imports`, transitive):
  - layers `cli → evaluation | baselines | graph → agents | validation → llm | patterns | data → domain`;
  - the generation side (`agents`, `graph`, `validation`, `llm`, `patterns`) must not import `evaluation`, `baselines` or `data.reference`;
  - `baselines` must not import `evaluation` or `data.reference`.
- **Dependency injection.** Agents receive their LLM client, renderer, validators and pattern repository through their constructors. The graph is wiring only; node logic lives in `GraphNodes`, and routing functions are pure `(state, policy) -> next`.
- **Serialisable state.** `GraphState` holds plain JSON values: Turtle strings, dicts and counters. It never holds an `rdflib.Graph`, so LangGraph checkpointing works.

## The loop

1. **Planning.**
   - `GroupPlanner` (deterministic) keeps dataset order, never mixes CQ `group`s, and chunks into batches of `batch_size`.
   - `LLMPlanner` asks the model for thematic batches. Its output is validated: it must be a partition with small batches; missing CQs are appended. On unusable output it falls back to the group planner.
2. **Generation.**
   - The Generator sees the committed draft's **signature** (class/property IRIs, labels, parents, domain/range), the batch, the top-k ODPs and any repair feedback.
   - It emits an **incremental Turtle fragment**. The fragment is merged into a **candidate** draft; the committed draft changes only on green.
3. **Translation.**
   - The translator sees only the candidate's signature, never its axioms or the generator's reasoning.
   - It writes one SPARQL query per CQ of the current batch. When vocabulary is missing it is told to write the query anyway, with the term the ontology *should* contain, so the gap becomes a validation error. `UNTRANSLATABLE` is reserved for questions no SPARQL could answer.
   - Queries that do not parse are re-asked once.
4. **Validation** runs cheapest first and stops at the first failure:
   - `syntax`: the fragment parses, and every structural IRI is in the base namespace or in owl/rdf/rdfs/xsd/skos/dcterms/ODP.
   - `consistency`: HermiT reports the candidate consistent with no unsatisfiable named class. A reasoner crash (for example an unsupported datatype) is a warning, not a failure.
   - `cq_coverage`: a structural check of each query; queries are never executed, because a TBox has no data.
     - Every class/property IRI must exist and be used in the right role.
     - Declared domains/ranges must be compatible with the variable types inferred from `rdf:type` patterns (sub- or super-class either way).
     - A missing domain/range gives a **warning**, not a failure.
     - Unknown individuals are allowed. Property paths and TBox-style queries (`rdfs:subClassOf [owl:onProperty …]`) are supported.
   - `regression`: re-checks the **cached** SPARQL of every CQ committed earlier. A CQ that passed before and fails now makes the batch red.
5. **Routing.**
   - Green: commit the candidate and cache the batch's SPARQL.
   - Red with repairs below `max_repair`: the Critic turns the report into numbered fix instructions (or, in the no-Critic ablation, the raw report is passed on), then generate again.
   - Otherwise: **rollback**. The candidate is discarded and the batch is marked failed.
   - With `retry_failed`, failed CQs get one more round at the end.
6. **Budget.** `BudgetPolicy` caps repairs per batch, plus global LLM calls, tokens and generator iterations. It is checked before each new batch and before each repair. When it runs out, `stop_reason` is set and the remaining CQs are reported as failed.

## Metrics

Generated IRIs never match the reference, so every comparison with the reference goes through a **one-to-one lexical alignment**. The `LexicalAligner`:

1. normalises names: splits camelCase and snake_case, strips CIDOC/FRBRoo/DOREMUS codes such as `E21_`/`P14_`/`U13_`, lowercases, and lemmatises plurals;
2. compares variants (local name, `rdfs:label`, and for properties a form without `has`/`is`/`of`) with rapidfuzz;
3. runs a Hungarian assignment and applies the threshold (default 0.85) afterwards.

| Key | Meaning |
|---|---|
| `class_precision/recall/f1`, `property_*` | Aligned generated terms / generated terms; aligned reference terms / reference terms (annotation properties excluded) |
| `*_recall_own`, `*_recall_imported` | Recall over the ontology's own terms vs. imported vocabularies (CRM, FRBRoo, OWL-Time, EDAM, OBO, ArCo …), reported separately |
| `*_recall_cq_relevant` | Recall over the reference terms that the gold SPARQL actually uses. **This is the fairest recall**: a 14-CQ input cannot be expected to rebuild SWO's 4,000 classes |
| `*_f1@{0.7,0.8,0.85,0.9,1}` | Sensitivity of F1 to the alignment threshold |
| `subclass_agreement` | Share of generated subclass edges (both ends aligned) that hold, transitively, in the reference |
| `domain_range_agreement` | Share of aligned domain/range declarations compatible with the reference |
| `posthoc_*` | **Identical for every system.** Every CQ is translated against the *final* signature and checked structurally: pass/warn/fail/untranslatable rates, `covered_rate` = pass + warn. Its own LLM cost is reported apart |
| `internal_*` | The loop's own per-CQ verdicts (agent system only) |
| `gold_query_pass_rate` | Gold `rq/` queries, with reference IRIs rewritten through the alignment, checked structurally against the generated ontology. Secondary metric, because it depends on alignment quality. `gold_queries_parseable` counts queries that survive dialect sanitising (template slots `$PPx1$`, Virtuoso comma projections, bare aggregates) |
| `gold_term_mapped_rate` | Share of reference-term occurrences in gold queries that have an aligned generated term |
| `n_classes`, `n_properties`, `property_domain_rate`, `label_rate`, `comment_rate`, `root_class_rate`, `final_consistent` | Size, hygiene and HermiT verdict on the final ontology |
| `llm_calls`, `cached_llm_calls`, `input/output/total_tokens`, `wall_time_s`, `repairs`, `rollbacks` | Generation cost |

Reference metrics are skipped for SILKNOW, which has no `dm/`. Coverage and stats metrics still run for it.

### Experiments (`experiments/configs/`)

| Config | What changes |
|---|---|
| `baseline-ontogenia` | **Ontogenia as in the paper**: one CQ per call, the complete ontology so far in context, the model returns the complete updated ontology. Five Metacognitive Prompting steps (comprehension, decomposition, reflection with ODP reuse, decision confirmation, evaluation with test individuals). No validation or repair |
| `baseline-single-pass` | All CQs in one call, from the original scaffold plan. Cheaper, but weaker than what the paper describes |
| `agents-full` | Group planner, batch 5, translator, critic, regression, HermiT, `max_repair=3`, retry round |
| `ablation-no-critic` | Raw validator report goes back to the generator |
| `ablation-llm-planner` | `LLMPlanner` instead of `GroupPlanner` |
| `ablation-no-regression` | No regression check |
| `ablation-no-translator` | No CQ→SPARQL feedback (syntax + consistency only) |
| `ablation-no-repair` | `max_repair=0` |
| `dry-run` | Offline heuristic LLM on awo/stuff/SILKNOW; numbers are meaningless |

## Dataset quirks handled

| Quirk | Handling |
|---|---|
| `ID: <CQId>` (DOREMUS, Odeuropa, SILKNOW), duplicate IDs | Stable `<onto>-<index>` IDs |
| `rq:` per CQ | Dropped by the loader; read only by `ReferenceOntology._query_links` |
| `[this software]` slots | Kept verbatim, whitespace normalised; `has_placeholders` flag |
| Free-text "In the Wild" CQs | Translator may say `UNTRANSLATABLE`, which counts as neither pass nor fail and is never sent to repair |
| CQs about individuals ("works by Mozart") | Prompts tell the agents to model the class; the coverage check allows unknown individuals |
| Empty `dm/` (SILKNOW) | `has_reference=False`; reference metrics are skipped |
| Multi-module references and imports | All `dm/` files merged; `data/shared/time.owl` added when OWL-Time is imported; imported namespaces are tagged and reported apart |
| `dm-rdf.yml` says `NORIA-O`, the folder is `noria-o` | Case-insensitive lookup; folder names are canonical |
| Gold SPARQL dialects | Sanitiser for template slots, missing standard prefixes and Virtuoso projections/aggregates. 18 DOREMUS, 45 Odeuropa and 2 NORIA-O queries still do not parse: sub-selects, an undeclared `ldp:` prefix, genuinely broken files. They are counted, not hidden |

## Known limitations and open items

- **Fragments are add-only.** A repair can add axioms but cannot delete committed ones. That is safe, because only green drafts are committed, but it means early modelling mistakes persist.
- **Domain/range compatibility is lenient.** It accepts any sub- or super-class relation, and multiple domains are treated as alternatives, not as an intersection. This avoids punishing under-specified drafts.
- **`ClaudeCodeClient` ignores temperature and seed**, because the CLI does not expose them. Determinism comes from the response cache. LangChain providers forward `temperature`; `seed` is forwarded only to OpenAI-compatible models.
- **The Ontogenia reproduction follows the paper's description; it does not reuse the authors' exact prompts.** Their per-step prompt mapping lives in their GitHub repository and should be diffed against `agents/prompts/ontogenia_*.j2` before reporting numbers.
- **Open items:**
  - `EmbeddingAligner` and LLM-as-judge (secondary metrics).
  - Calibrating the alignment threshold on one ontology by hand; the `f1@t` curve is already recorded.
  - ODP retrieval instead of a static catalogue.
  - `use_description` defaults to `false`. The paper's Ontogenia uses a user story, so report both settings, or turn it on for the comparison (only DOREMUS, DemCare, Odeuropa, Polifonia and NORIA-O have a `description.txt`).
  - Java in CI.

## Development

```bash
uv run pytest                 # unit + integration (fake LLMs); HermiT tests need java
uv run ruff check src tests && uv run ruff format --check src tests
uv run ty check src
uv run lint-imports
```
