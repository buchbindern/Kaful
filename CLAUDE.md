# CLAUDE.md — Kaful revamp (doc-first pipeline)

Kaful turns equipment documentation into physics-based prognostic digital twins that
produce probabilistic remaining-useful-life (RUL) forecasts. This revamp turns a research
prototype into a production-grade system. The priorities are:

- evaluation of every LLM output;
- provenance for every value;
- structured logging and API cost control;
- a clean phase architecture with typed contracts;
- interfaces that work the same for batch files and live streaming data.

`OVERVIEW.md` is the audit of the current code, and `PLAN.md` holds the milestones. Read both
at the start of every session. `CLAUDE_legacy.md` (repo root, local only, gitignored) is the
old description of the system, for reference only; where it conflicts with this file, this
file wins.

---

## How we work

- **One milestone at a time, in PLAN.md order.** No "while I'm here" changes outside the
  current milestone.
- **Before writing code**, post a short plan: the files to touch, interface changes, the test
  approach, the expected API cost (if any), and anything you think PLAN.md got wrong.
  Wait for Natalie's "go".
- **Branch per milestone** (`m4-typed-artifacts`). Never commit to `main`. Small commits,
  one logical change each, with messages like `M4: add Spec artifact model`.
- **Merging** (on Natalie's "merge"): merge the milestone branch into `main` with `--no-ff`,
  then push. Before merging, confirm `git status` is clean: no uncommitted or untracked
  changes in the milestone's files. After merging, run `pytest -q` on the merged result
  and push only if it is green.
- **Never switch branches in the main working tree while a live run is in progress; use a
  separate git worktree.**
- **Done means:** acceptance criteria met, `pytest -q` green offline, and a short report
  covering what changed, before/after measurements, API spend, what was deferred, and any
  surprises. Tick the box in PLAN.md and add a one-line note under it.
- **If the code disagrees with PLAN.md or OVERVIEW.md, stop and say so.** Do not work around
  it quietly.
- **Ask before:** deleting files outside the milestone's scope, changing an artifact format,
  adding a dependency, any paid API call, re-indexing a document, or regenerating a
  machine's outputs.

## Migration strategy

The new pipeline is built as a package, `kaful/`, **alongside** the legacy code
(`phases/`, `run.py`). The legacy pipeline stays runnable until the new one beats the M1
baseline on the evaluation metrics (PLAN M15); matching legacy outputs is not the goal.
Legacy code is deleted only in an approved milestone. The new code does not import from
`phases/`; anything it needs is reimplemented cleanly.

**Legacy code is frozen.** Don't fix legacy bugs; log them under the milestone that replaces
that code.

---

## Architecture

### Phases (each has one job, a typed input, and a typed output)

| # | Phase | Job | Main artifact |
|---|---|---|---|
| 1 | ingest | documents → raw page text, tables, page labels, index | `DocumentIndex` |
| 2 | understand | components, flow paths, fault modes, **sensor contract** | `MachineModel`, `SensorContract` |
| 3 | scope | decide which components get full models (before any expensive per-component work) | `Scope` |
| 4 | parameterize | every rate, threshold, capacity and nominal input as a **prior** with provenance | `ParamSet` |
| 5 | build | pick a degradation template per component, bind params, wire the composite | `TwinDefinition` |
| 6 | verify | static and dynamic checks; failure stops the run | `VerificationReport` |
| 7 | synthesize | *optional, no-data case only*: generate observations + hidden truth from params | `SyntheticDataset` |
| 8 | estimate | state (+ parameter) estimation and RUL forecast from a `DataSource` | `Forecast` |
| 9 | evaluate | grounding, twin quality, forecast calibration, sensitivity | `EvaluationReport` |

**Naming (decided 2026-11-06; enforced by `tests/unit/test_kaful_phase_layout.py`):**
- Phase folders carry their order and match `PHASE_ORDER`: `kaful/phases/p1_ingest` …
  `p9_evaluate`.
- **Steps:** only true pipeline steps get a letter, in strict execution order with no gaps,
  plus what they do (`p1_ingest/a_check_documents.py`, `b_read_pages.py`, …). `phase.py`
  calls each step directly, in that order, and opens with a map: each step, one line on
  what it does, and the helpers it uses.
- **Helpers:** code the steps use that is not a step goes in the phase's `helpers/`, named
  for what it does; each helper's docstring lists the steps that use it.
- **Tools:** code not used during a run (CLI commands, eval and labelling tools) goes in the
  phase's `tools/`, unprefixed.
- **Shared code:** code that several phases use goes outside `phases/`, e.g.
  `kaful/retrieval/` (index loading and search).
- Prompt ids, artifact kinds, CLI phase names and run directories stay unprefixed
  (`ingest`, `ingest.extract_page`), so saved runs, caches and prompt hashes never depend
  on file layout.

### Core interfaces (implementation-agnostic)

- **`SensorContract`**: channels, units, ranges, sample rates. It is the single definition of
  "what the machine reports", and the synthesizer, the observation models, and streaming
  ingest all read it. If an artifact isn't used downstream, it shouldn't exist.
- **`ParamSet`**: each parameter is `{name, component, unit, prior, posterior?, source, ref,
  confidence}`. `prior` is a distribution (e.g. lognormal with a median and spread), not a
  single number. `posterior` is filled later by data. Nothing in the twin, simulator or
  forecast uses a literal constant; everything reads from the `ParamSet`.
- **Degradation templates**: pure-math parametric forms (in plain numpy, with no library
  dependency) of the shape **rate = k × f(operating conditions)**. Physics or standards give
  f, and k is calibrated so that nominal conditions reach the limit at the documented life.
- **`TwinBackend`**: `build(twin_definition)`, `simulate(...)`, `estimate(...)`,
  `predict(...)`. ProgPy is the first adapter, and **no ProgPy import may appear outside
  `kaful/backends/progpy/`**. Swapping libraries means writing one new adapter.
- **`DataSource`**: yields timestamped observations that conform to the `SensorContract`.
  `FileSource` (CSV/parquet) and, later, a stream source implement the same interface.
  Estimators process one observation at a time, so batch and streaming share the same code
  path.
- **Operating profile**: forecasts accept a future operating profile (load, speed, temp
  over time), so "what-if" questions such as "run at 80% speed" are possible later.

### Cross-cutting services

- **LLM gateway** (`kaful/llm/`): the only way to call a model. It does the following:
  - logs every call to the **run ledger**: run id, phase/subphase, prompt id + version, model,
    input/output/cached tokens, cost, latency, cache hit, and output artifact;
  - provides a content-hash response cache, prompt caching for repeated context, and Batch
    API support for bulk work;
  - enforces a **per-run budget**, so the run stops cleanly when the budget is exceeded;
  - takes model choice from config per prompt, never from a hardcoded string.
- **Prompt registry** (`kaful/prompts/`): one versioned file per prompt, plus its output
  model, plus its eval set. Each rule is stated once per prompt. Prompt changes bump the
  version and must pass that prompt's evals.
- **Run manifest**: inputs, code version, prompt versions, index hash, config, overrides,
  total cost, and pass/fail per phase. Every artifact records the run id that produced it.
- **Structured logging**: `logging` with JSON output; no `print` in pipeline code.

### Proposed layout

```
kaful/
  cli.py            # kaful run <machine> --from/--only/--budget/--allow-failures
  config.py         # pydantic-settings; models, budgets, tunables
  artifacts/        # Pydantic models for every artifact (schema_version on each)
  llm/              # gateway, ledger, cache, pricing table
  prompts/          # versioned prompts + output models
  phases/           # p1_ingest/ p2_understand/ p3_scope/ p4_parameterize/ p5_build/
                    # p6_verify/ p7_synthesize/ p8_estimate/ p9_evaluate/
                    # each: phase.py, lettered steps, helpers/, tools/
  retrieval/        # loading a built document index; vector search
  templates/        # degradation templates (pure math) + tests
  backends/progpy/  # the only place ProgPy is imported
  data/             # DataSource implementations
  sandbox/          # subprocess runner for any generated code
  evals/            # eval runners, metrics, report builders
machines/<id>/      # machine.toml + documents + outputs/runs/<run_id>/... (gitignored)
tests/              # unit/ integration/ evals/ fixtures/
```

---

## Hard rules

1. **No network in tests.** The LLM gateway and embedder are replaced with recorded
   fixtures, and a missing fixture is a test error that names the prompt.
2. **Gates gate.** A verification failure stops the run, or explicitly quarantines that
   item, and the run reports it. Overrides require `--allow-failures` and are recorded in
   the manifest.
3. **Every number has provenance.** Sources are `manual`, `standard`, `handbook`,
   `literature` (secondary sources such as theses and papers), `calibrated` (derived from a
   documented life or interval), `data`, `deployment` (declared by the user in
   machine.toml, e.g. added instrumentation), `llm_judgement` or `default`. If a parameter
   has no source, verification fails.
4. **Evidence is checked by code.** A value counts as `manual` only if its quote is found in
   the raw page text of a cited page. Otherwise it is downgraded, and the downgrade shows up
   in the report. A reading shown in a screenshot or photo is evidence that the channel
   exists (it keeps its source, weighted as `in_layer`), but the value shown is only an
   example and never sets a range or a prior.
5. **Be honest about rates.** Manuals rarely state wear rates. Use the hierarchy in PLAN M9;
   service intervals are conservative lower bounds on life, not failure times;
   `llm_judgement` priors are wide. Reports must never imply accuracy we cannot validate
   without failure data.
6. **Conventions are enforced in code:** health states lie in [0, 1] and start at 1.0, there
   is one declared `time_unit` per machine, and every rate is expressed per that unit.
7. **Generated code runs in the sandbox subprocess with a timeout**, never imported into
   the pipeline's interpreter. No `sys.path` / `sys.modules` hacks.
8. **Typed artifacts only.** Load and save through Pydantic models, and validate LLM
   output against the prompt's output model. On failure, retry once with the validation
   error in the prompt; after that, fail.
9. **Don't regenerate outputs to make a test pass.** Report the violation instead.
   Regeneration is a separate, approved step.
10. **Never weaken a check** (threshold, tolerance, classification) to get green. Propose
    the change and explain why.
11. **Cost is a feature.** For every milestone that touches LLM calls, report the cost per
    run and per phase against the ledger baseline. Use the cheapest model that passes the
    prompt's evals.
12. **No absolute paths in generated code or artifacts.** Resolve paths relative to the
    machine directory at load time.
13. **Artifacts are only produced by the CLI from committed code and config.** No
    notebook-produced outputs.
14. **Output must not depend on set or dict iteration order; sort explicitly.** Determinism
    tests run under different `PYTHONHASHSEED` values.
15. **Confidence scores are computed by code from measurable signals, never asserted by an
    LLM.**

## Environment

- Conda env `pipeline311`, Python 3.11 (ProgPy 1.7.1 doesn't support 3.13). Use `python -B`.
- Secrets live in `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). Never print or commit them.
- Legacy run: `python run.py <machine_id> --from phase2` (use `--from phase2` for CNC
  until the index is rebuilt).
- Reference machine: `eversys_coffee_machine`. Development target: `cnc_spindle`.

## Testing

- `pytest -q` covers unit and offline integration tests, and must stay green.
- `pytest -m evals` runs prompt evals on recorded outputs.
- `pytest -m slow` runs the golden-run regression.
- `pytest -m live` makes real API calls and is **never run without approval**.
- New logic ships with tests. Bugs get a failing test first. Templates get
  property-based tests (`hypothesis`).
- **Before locking any prompt, its eval cases run at least twice** (`kaful eval run
  --sample 1` repeats with fresh calls), every run of the locked model or route passes, and
  the lock records the per-run scores. A prompt with model routing is locked on a routed
  run (`--routed`).

## Code style

- Type hints everywhere; `mypy` on all non-generated code.
- Functions under ~60 lines.
- No magic numbers: put them in config or settings models.
- Validation results are saved data, not console output.