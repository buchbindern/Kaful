# PLAN.md — Kaful revamp milestones

Ordering principle: first **measure** (cost, behaviour), then build the **foundation**
(contracts, evals, gateway), then rebuild the **phases in pipeline order**, and finish with
evaluation and delivery.

Workflow for each milestone: plan → Natalie approves → small commits → report → tick.
Section references (§) point to OVERVIEW.md.

---

## Stage A — Safety net and baseline (legacy code, no behaviour change)

### [x] M0. Freeze the baseline
- Ask Natalie whether to commit or stash the uncommitted eversys spec/validation changes and
  the `utils/rag.py` changes. Then tag `pre-revamp`.
- Pin dependencies in `requirements.txt` (lowercase) to `pipeline311`'s current versions,
  and add `.env.example`.
- Add `pyproject.toml` with pytest config and the `slow`, `evals` and `live` markers.
- **Accept:** a fresh install from the pinned file works, and `pytest` runs.
- *Done 2026-11-02:* tag `pre-revamp` = `3347610`; repo moved out of iCloud to `~/code/Kaful`;
  gitignored baseline in `~/Kaful-baselines/pre-revamp-2026-11-02.tar.gz` (240 MB, sha256
  `50e9cea1…93cfd9`, manifest sha256 `317d99e1…7678ec`).

### [x] M1. Instrument the legacy pipeline (cost and logging baseline)
- Wrap `utils/llm.py`, the PDF extraction calls and the embedding calls with a minimal
  ledger. Log phase/step, model, tokens in/out, cost, latency and cache hit to
  `machines/<id>/outputs/ledger.jsonl`. Pricing comes from a config table.
- Work out the cost of a full eversys run per phase from existing data where possible. If
  that isn't possible, **estimate** it from prompt sizes. Any real run needs approval.
- **Accept:** a baseline cost table per phase and per step. It names the top 3 cost drivers
  (expected: phase 1 re-extraction, phase 2's 5× extraction, per-component physics/codegen).
- *Done 2026-11-02:* ledger in `utils/ledger.py`, prices in `pricing.toml`; baseline in
  `docs/m1_cost_baseline.md`. eversys $5.96 cold / $3.78 `--from phase2`; CNC $22.06 / $6.21.
  Top drivers: phase 1, 6b codegen (ProgPy context), 2c (CNC). $0 spent (free `count_tokens` only).

### [x] M2. Offline harness and golden run
- Fake LLM gateway and fake embedder that replay recorded responses by prompt hash.
- A **tiny fixture machine** (2 components, a 3-page synthetic manual) that runs the legacy
  pipeline end-to-end offline in under ~60 s.
- Golden snapshot of the eversys phase 6–7 outputs. A `slow` test diffs against it using
  reviewed tolerances.
- **Accept:** `pytest -q` is green with sockets blocked.
- *Done 2026-11-02, closed early (legacy frozen):*
  - **Built:** network block, replay/recording clients keyed by `prompt_hash`, a hashing
    embedder, frozen ProgPy hits (NOSA notice), and the eversys golden snapshot. The
    snapshot is pinned to `PYTHONHASHSEED=0` + seed 0, compares exactly, and takes ~9 min.
  - **tiny_pump:** replays phases 1, 2, 4 and 5 offline in about 1 s. Phase 3 can't replay
    because of the 3a hash-order bug (logged under M7). Phases 6–7 were not recorded.
  - **Not met:** the full end-to-end tiny-machine run.
  - **Spend:** $1.65 recording (cap $2.00).

### [x] M3. Remove what's dead
- Delete `data_pipeline/` (this work now lives in a separate production repo). Before
  deleting, keep a copy of `reconcile.py` outside the repo (it is also preserved at tag
  `pre-revamp`); M4 brings it into `kaful/` as a reference implementation.
- The rest of the legacy cleanup is dropped: legacy code is frozen until M15 deletes it.
- **Accept:** `data_pipeline/` is gone, the `reconcile.py` copy exists outside the repo, and
  the tests are green.
- *Done 2026-11-02:* 21 files removed; `reconcile.py` kept at
  `~/Kaful-baselines/m3/reconcile.py` (sha256 `fb5d4e9f…4806`, same as tag `pre-revamp`).
  The PHM `raw/c1` symlink was already dangling (`~/kaful_raw/` doesn't exist). $0 spent.

---

## Stage B — Foundation (new `kaful/` package)

### [x] M4. Package skeleton and typed artifacts
- Create the `kaful/` layout from CLAUDE.md, with a CLI stub
  (`kaful run <machine> --from --only --budget --allow-failures`).
- Write Pydantic models for every artifact in the phase table, each with a
  `schema_version`. Include `ParamSet` with prior/posterior slots and `SensorContract`.
- Write the run manifest and run-scoped output dirs (`outputs/<run_id>/<phase>/`).
- Write the sandbox runner: run generated code in a subprocess with a timeout and return a
  structured result. It loads generated code by machine-relative path.
- Define the interfaces only, with no implementations yet: `TwinBackend`, `DataSource`,
  and the degradation-template protocol.
- Add the copy of `reconcile.py` kept in M3 to `kaful/` as a reference implementation.
- **Accept:** models round-trip, the interfaces have docstrings, and the tests are green.
- *Done 2026-11-03:* 10 artifact models + manifest round-trip, with byte-identical JSON
  under 3 hash seeds. Templates track physical damage D (health = 1 − D/D_limit), with
  `apply_maintenance` reserved. Runs live in `outputs/runs/<run_id>/` (gitignored), and
  `--from` auto-reuses a base run only when the inputs match. The sandbox runs code in a
  subprocess with a scrubbed env and process-group kill on timeout. pytest 280 passed,
  mypy strict clean. Deps: pydantic, pydantic-settings pinned; mypy 2.4.0 (dev). $0 spent.

### [x] M5. LLM gateway, prompt registry, eval framework
- The gateway provides the ledger, a content-hash cache, prompt caching, a Batch API path,
  a per-run budget stop, model choice per prompt from config, and retry-on-validation-error
  (once).
- Prompt registry: versioned prompt files, each paired with an output model.
- Eval framework, where each prompt has an eval set made of:
  - **deterministic checks**: output parses, the schema is valid, quotes are found in the
    source, units parse, and values are in range;
  - **golden examples**: hand-labeled input → expected output for key fields;
  - an **LLM judge**, used only where deterministic checks can't decide (e.g. whether a
    paraphrase is faithful). The judge has its own small validation set.
- Eval results are stored per prompt version, so prompt changes can be compared.
- **Accept:** a sample prompt runs through the gateway into the ledger and is scored by its
  evals offline.
- *Done 2026-11-03:* gateway in `kaful/llm/` (one ledger schema with M1, v2 rows per run;
  pre-call worst-case budget check; response cache in M2's fixture format; Batch API with
  resumable state); registry in `kaful/prompts/` with `versions.lock.json`; evals in
  `kaful/evals/` with results in `evals/results/`. Sample prompt and judge both pass on
  Haiku 5.5 and Sonnet 5.5 with identical scores, so both run on Haiku (~20x cheaper).
  `unit_parses` is declared but deferred to M7 (pint). pytest 334 passed, evals 4 passed,
  mypy strict clean. Spend $0.0386 (cap $0.25).

---

## Stage C — Rebuild the phases (in pipeline order)

For each phase:
1. Build it in `kaful/phases/<phase>/`.
2. Give it typed input/output and evals for its prompts.
3. Report its cost against the M1 baseline.
4. Show parity or a documented improvement on eversys.

### [x] M6. Ingest
*Done 2026-11-06: eversys $0.21 (≤ $0.65), CNC $3.71 (≤ $4.75); gate passed; see docs/milestones/M6.md.*

- Extract raw page text with PyMuPDF and keep the printed page labels. Use **LLM extraction
  only for pages that need it** (tables, figures, garbled text), chosen by a cheap heuristic.
  This is expected to be the biggest cost cut.
- Chunk IDs are deterministic (a hash of file sha + page range + chunker version). An
  **index manifest hash** is recorded in every downstream artifact; a mismatch triggers a
  warning, or an error with `--strict`.
- The machine config's document list is validated against the files on disk. This fixes the
  CNC list (§5).
- **Accept:** CNC index rebuilt (with approval), each chunk stores raw text + labels, and
  the M6 gate below passes.
- **M1 baseline → target:** phase-1 cost cut by ≥70%. eversys $2.18 → ≤$0.65; CNC
  $15.85 → ≤$4.75 (CNC baseline partly estimated, range $14.46–$17.61; the target uses
  the point value).
- **M6 gate** (decided 2026-11-03; the old index is reference only, not a target: it was
  partly hand-tweaked and its quotes came from its own LLM transcription, so matching it
  is not a quality measure):
  - page-classifier needs-LLM recall ≥ 0.90 on Natalie's hand labels
    (`tests/fixtures/ingest/page_class_labels.toml`);
  - `ingest.extract_page` passes its evals on the golden pages;
  - coverage: every page with text is in exactly one chunk (chunker-v2, 2026-11-05: pages
    classed blank are chunked when they have text, e.g. Fanuc's part-divider pages);
  - a hand-review table of *old-only* quotes (supported by the old index, not by the new
    one): is the information present in the new index (yes / no / partial), regardless of
    wording.
- **Reported for context, not gated:** a fixed set of evidence quotes taken from today's
  saved checkpoints (frozen in `tests/fixtures/ingest/grounding_quotes/`), scored against
  the old index (labelled *partial* for CNC: live Chroma + saved phase-2 context), the new
  index with all layers, and the new index's raw text only (what rule 4 accepts), plus
  *page coverage* of old and new indexes.

- **Extraction routing** (decided 2026-11-05; `ingest.extract_page` v2 locked): Haiku 5.5 on
  every LLM page; a page whose Haiku output has any table is re-run on Sonnet 5.5 and Sonnet's
  output is used. The rule is `[llm.prompts."ingest.extract_page".escalate]` in kaful.toml;
  every call's ledger row carries its `route`, each `llm` layer its model, and the index
  fingerprint the rule. Why: Haiku v2 passes at table_cells 0.90 but collapses a merged cell
  to one row (NSK p28, 0.5); Sonnet scored 1.0, and class routing cannot pick those pages
  (most table pages are classed `figure`).

### [ ] M7. Understand (+ sensor contract)
- Extract components, flow paths and fault modes, with evidence quotes and pages.
- **Replace the old field schema with `SensorContract`** (today: eversys 27 fields,
  CNC 29, air_compressor 19, CDV 34) (channels, units, ranges,
  sample rates). It is used downstream by the synthesizer, the observation models and the
  data sources. The DDL is gone.
- Self-consistency: N extractions. Field agreement is kept as `confidence` instead of being
  discarded. N is tuned against eval quality vs cost.
- Units are parsed with `pint` and never pass through name normalization. Ranges keep their
  sign.
- Evidence verifier: fuzzy-match each quote against the raw page text and classify it as
  `verbatim`, `near` or `unsupported` (plus `in_layer`: found only in an LLM transcription
  layer of the cited page, which rule 4 does not accept as `manual`).
- **Decisions (2026-11-05):** `MachineModel.time_unit` removed (machine.toml declares it, M9
  confirms it); `Component.parent` added (understand records parts, M8 decides scope);
  sampling is explicit (`periodic` with a quoted period, `event_driven` or `unknown`), so a
  guessed rate never passes for a documented one; channels the deployment adds are declared
  in machine.toml `[[instrumentation]]` with source `deployment`; `pint` added. CNC channels
  come from Fanuc's readable data, with MTConnect for naming and typing only.
- **Value roles (approved 2026-11-06; from the eversys old-only review):** every extracted
  value's evidence gets `Evidence.value_role`: `spec` (a stated property of the machine:
  rated, nominal, limit, life), `variant` (a real spec, but for one of several configurations,
  e.g. eversys p136 bean hopper 1 kg vs 0.5 kg), `default` (factory or initial setting),
  `setting_range` (adjustable min..max), or `example` (illustration: a screenshot value, a
  worked example, an "e.g.").
  - The LLM proposes the role; code may only downgrade it, never upgrade an example: a value
    found only in a figure's LLM layer (screenshot, photo) or in a worked-example context
    ("example", "e.g.", "for instance") is `example`. Agreement across samples feeds confidence.
  - Eval check `examples_not_specs`: golden example values (eversys p94 10 s countdown, p72
    100 ml, p69 Pre-Infusion 0.8 s and the other p69 screenshot settings, a tiny_pump worked
    example) must never come out as `spec` (threshold: 0); a `spec` recall check keeps the
    model from calling everything an example; a golden `variant` case (p136 bean hopper).
- **Format changes (approved 2026-11-05):** `MachineModel.boundaries`, the `UnderstandReport`
  artifact, the `in_layer` evidence class, `EvalResult.case_tag`; source kind `literature`
  for secondary sources (document kind `other` renamed `literature`; the KTH thesis uses it).
- **Accept:** grounding recomputed for eversys, CNC and tiny_pump and reported beside §5
  (today: eversys 19/27, CNC 7/29), labelled as different metrics: §5 counted evidence found
  in the retrieved LLM-transcribed chunks; M7 checks each quote against the cited page's raw
  text. air_compressor (10/19) and CDV (10/34) are reported unchanged: they have no new index.
- **M1 baseline → target:** the whole understand phase on CNC ≤ $1.08, against a baseline
  of $2.41 (2b $0.56 + 2c $1.61 + 4a $0.20 + 4c $0.04, the legacy steps it replaces); report
  eversys too (baseline $0.59).
- *Note (M2):* the tiny_pump fixture machine produced 30 schema fields from a manual that
  lists 7 channels. Use it as an eval case for sensor-contract over-extraction.
- *Eval case (M2, legacy bug, not fixed):* legacy 3a (`_organize_by_topic`) orders retrieved
  context by set iteration, so prompts change with `PYTHONHASHSEED`. The new understand
  phase must build every prompt and every artifact in a declared, sorted order (rule 14).
  Test it by building the same prompts under two hash seeds and comparing their hashes.
- **Known gaps at M7 close (2026-11-09; exit criteria and time-box in docs/proposals/M7-exit-criteria.md):**
  - CNC meets no floor with a scored run: the last scored run (20261108T192837Z-35431a) predates the
    document-roles, units, instances and declared-components fixes; its last paid iteration is unspent.
  - CNC channel over-extraction: one quantity reported from two pages (diagnosis screen p200 and trace list
    p730). sensor_channels v6 ("report each quantity once") made Haiku drop per-model variant ranges in 3 of
    4 eval runs; parked on branch m7-sensor-channels-v6. Next: a code-side merge, measured first.
  - Hold-out fault-mode recall 0.31 (air_compressor): the components step misses consumables and parts
    named only in the maintenance schedule (oil, motor bearings, flexibles, restrictors, float valve, dryer
    condenser), so their fault modes have no carrier; handbook mechanisms (insulation ageing, intake-dirt
    wear) were not retrieved. For M8/M9, or a prompt change judged on a new hold-out.
  - Hold-out channel roles 0.77: status and alarm channels typed observation (golden operating_input/event).
  - Parent accuracy has no floor and is low (eversys 0.63, hold-out 0.95 only on the parts it found): runs
    leave parts at the top level.
  - Merge misses left: eversys error_history / error_message (share only a page), ticks_per_second / flow and
    total_product_counter in run 1ec5d2; hold-out load_state / compressor_control_status.
  - Budget reservations reserve 7-8x actual for extract_page and make small budgets crawl through batch
    rounds: proposal docs/proposals/post-M7-budget-reservations.md.
  - For M9: golden rows marked documented = false (eversys grinder burrs, CNC motor overheating) are handbook
    priors. For M15: Sonnet locks and routing.
  - ruff: two unused names in tests/unit/test_harness.py and test_kaful_llm_gateway.py (not M7 code).

### [ ] M8. Scope
- Triage runs **before** any per-component physics work and decides between full model,
  connection only, and excluded. Each decision has a reason and a manual override.
- **Accept:** the cost of later phases scales with the number of in-scope components only.
- **M1 baseline → target:** no per-component LLM spend on out-of-scope components.
  eversys $0.16 (4b on 4 components triage drops) → $0; CNC $0.07 (4b on
  `housing_assembly`) → $0.
- *Eval case (M2):* tiny_pump's manual says it has exactly two main components, yet legacy
  4a found 11, and 5b kept all 11 as full models. Use it as an over-decomposition eval:
  scope should keep 2 full models, with sensors as connections or observations, not
  components.

### [ ] M9. Parameterize (rates as priors)
Build every parameter as a prior distribution with provenance. Rates use this hierarchy:
1. **Calibrated from documented life or intervals** (rated life, MTBF, service and
   replacement intervals): solve the template for k so that nominal conditions reach the
   limit at the documented life. Service intervals are conservative, so they count as
   **lower bounds** on life.
2. **Standards and physics** (ISO 281 from catalogue load ratings, Taylor tool life,
   Archard, Arrhenius, scale buildup) define how the rate scales with conditions, f(·).
3. **Reliability handbooks** for component-class failure rates. Licensing must be checked
   before use.
4. **`llm_judgement`** as a last resort, with a wide prior and a clear label.

Other parts of this milestone:
- Nominal external inputs come from the sensor contract and the documentation, each with
  provenance. No blanket 1.0.
- One declared `time_unit` per machine, with every rate expressed per that unit. This fixes
  the hours-vs-brew-cycles mismatch (§3).
- **Value roles (from M7):** only `spec`, `default` and `setting_range` evidence can set a
  prior (`setting_range` bounds it; `default` is a nominal operating input). A `variant` sets
  a prior only when machine.toml declares which configuration applies; otherwise it stays a
  labelled hint. An `example` is at most a labelled hint in the report, never a prior;
  verification fails if a prior's evidence is an example.
- **Accept:** every parameter in eversys and CNC has a source, and a provenance report shows
  the counts by source.

### [ ] M10. Build (templates + backend)
- A degradation template library in pure numpy. Each template has units and assumptions
  documented and property tests (`hypothesis`) for monotonicity, boundedness, condition
  scaling, and zero load → zero wear where that applies. The first set:
  - linear and exponential;
  - power law / Paris-type;
  - Archard;
  - an Arrhenius multiplier;
  - ISO 281 bearing life;
  - first-order tracking;
  - accumulation to capacity.
- The LLM **selects** a template per fault mode and justifies the choice. Code is
  **rendered deterministically** from template + `ParamSet`. Free-form code is allowed only
  as a flagged escape hatch.
- Composite wiring uses reconcile-style matching plus a unit-compatibility check on every
  connection. Unresolved inputs become explicit external inputs, each with provenance.
- *Eval cases (M4, reference `kaful/reference/reconcile.py`, not fixed there):*
  - its placeholder loop iterates `valid_channels`, a set, so warning and placeholder order
    change with `PYTHONHASHSEED`;
  - `difflib.get_close_matches` over a set breaks score ties in hash order.

  The reimplementation must sort its candidates and outputs (rule 14). Test it under two
  hash seeds.
- The ProgPy adapter implements `TwinBackend`. No ProgPy imports exist outside
  `kaful/backends/progpy/`.
- **Accept:** CNC rebuilt (with approval), with bearings on ISO 281 using catalogue values.
- **M1 baseline → target:** 6b-equivalent cost (template selection + any escape-hatch
  codegen) per in-scope component cut by ≥80%. eversys $0.18 → ≤$0.036; CNC $0.22 →
  ≤$0.044 (baseline excludes 6b retries).

### [ ] M11. Verify
Checks that stop the run:
- every health state starts at 1.0 and stays in [0, 1];
- all states stay within bounds, with no NaN or inf;
- degradation speed is consistent with the calibrated life, within an approved band (too
  slow fails);
- tracking states converge;
- `event_state` agrees with `threshold_met`;
- every parameter has provenance;
- every connection is unit-compatible.

Failing components are quarantined and listed in the manifest. Composite checks are saved,
not printed.
- **Baseline:** run the new verification checks on the legacy twins (all 4 machines) to
  establish the pass-rate baseline for M15.
- **Accept:** all in-scope components pass on eversys and CNC (scope as decided in M8),
  with no event triggered at t=0.

### [ ] M12. Synthesize (optional phase)
- Generates observations that follow the `SensorContract` **plus the hidden true states
  and true failure times**, using the same `ParamSet`.
- It is labeled as a pipeline sanity check, not validation. Scoring the twin on its own
  synthetic data is partly circular, and the reports say so.
- **Accept:** each dataset has a matching truth file.

### [ ] M13. Estimate and forecast
- Estimators consume a `DataSource` one observation at a time: a particle filter (UKF for
  comparison) and open-loop as the baseline.
- **Joint state and parameter estimation**: uncertain rate constants are augmented into the
  state, so a machine's own data updates its `posterior`. This is designed and tested here
  even though production streaming comes later.
- Forecasts save every time-of-event sample per event. They include explicit censoring
  (hit count, `P(event before horizon)`), percentiles (5/50/95), and an "already
  triggered" decision made per sample.
- Forecasts accept a future **operating profile** for what-if scenarios.
- **Determinism test:** the same estimator run with the same seed under at least two
  different `PYTHONHASHSEED` values must give identical output. In M2, legacy phase 7 failed
  this: ProgPy orders `CompositeModel` states, inputs and events from sets, which changes how
  noise draws are assigned to states.
- **Accept:** on synthetic data, the particle filter beats open-loop on state error, the
  posteriors move toward the true parameters, and the determinism test passes.

### [ ] M14. Evaluate
Build one evaluation report per run with:
- **Grounding:** fields by verification class, and parameters by source.
- **Prompt quality:** eval scores per prompt version.
- **Twin quality:** the verification results.
- **Forecast calibration** against synthetic truth: 90% interval coverage, α-λ accuracy,
  prognostic horizon, and RMSE of the median RUL. Calibration has no legacy baseline
  (legacy simulators don't log truth), so it uses an absolute target: 90% prediction
  intervals contain the true failure time 85–95% of the time on synthetic data. Report the
  actual coverage.
- **Sensitivity:** which parameter priors dominate RUL uncertainty, i.e. which data would
  help most.
- **Cost:** spend per phase and per run, compared with the M1 baseline.

The report states plainly that without real failure data, absolute rates are checked for
*consistency* with documentation and physics, not for *accuracy*.
- **Accept:** reports for eversys and CNC, which also fill the ASMB draft placeholders.

---

## Stage D — Delivery

### [ ] M15. CI, demo, legacy removal
- GitHub Actions running lint, `mypy` and `pytest -q` offline.
- A Dockerfile (pinned Python 3.11 image, `requirements.txt`, the package) that CI builds and
  runs the offline tests in; the same image is the deployment unit.
- The demo report is built from the new artifacts, with fan charts drawn from saved samples.
- The new pipeline is judged by evaluation metrics, not by matching legacy outputs. On
  eversys and CNC it must beat the M1 baseline numbers on:
  - grounding (fixed evidence-quote set and page coverage, as defined under M6);
  - verification pass rate (baseline from M11's run of the new checks on the legacy twins);
  - cost per run and per phase;
  - forecast calibration: no legacy baseline (legacy simulators don't log truth), so it must
    meet the absolute target from M14 (90% intervals cover the true failure time 85–95% of
    the time on synthetic data), with the actual coverage reported.
- Once that holds, delete the legacy `phases/`, `run.py`, `utils/` and `prompts/` (with
  approval).
- **Accept:** CI is green, the evaluation report shows the new pipeline beating the
  baselines on those metrics (and meeting the calibration target), and the legacy code is
  gone.

---

## Parked (designed for now, built later)
- **Adaptive twin in production:** live streaming `DataSource`, with posteriors updating
  online per machine (the interfaces exist from M4 and M13).
- **Prescriptive operation:** the twin recommends set points (e.g. slow down or speed up),
  trading throughput against RUL with the what-if forecasts from M13. It starts as
  human-approved recommendations, never direct machine control.
- **Prompt improver:** automated prompt optimization against the M5 eval sets.
- **Replace ProgPy:** write a new `TwinBackend` adapter. This is the last item.
- **Model upgrades:** swap models through config and compare eval scores and cost.
- **Container or microVM sandbox isolation:** run generated code in a container or microVM
  instead of a subprocess. Only needed if Kaful becomes multi-tenant or runs unattended in
  production; the subprocess sandbox (M4) is enough for a single trusted user.
- **Cheaper extraction escalation** (from M6): escalate only pages whose Haiku tables show
  the collapsed-merge pattern (a column whose values are separated by runs of empty cells
  while the other columns are full), instead of every page with a table. Needs labelled
  pages to check it never misses a collapsed table.
- **Merge with the streaming production repo:** the `SensorContract`, `ParamSet` and
  template library are the natural seam.