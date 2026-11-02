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

### [ ] M2. Offline harness and golden run
- Fake LLM gateway and fake embedder that replay recorded responses by prompt hash.
- A **tiny fixture machine** (2 components, a 3-page synthetic manual) that runs the legacy
  pipeline end-to-end offline in under ~60 s.
- Golden snapshot of the eversys phase 6–7 outputs. A `slow` test diffs against it using
  reviewed tolerances.
- **Accept:** `pytest -q` is green with sockets blocked.

### [ ] M3. Remove what's dead
- Delete `data_pipeline/` (this work now lives in a separate production repo). Before
  deleting, keep a copy of `reconcile.py` outside the repo (it is also preserved at tag
  `pre-revamp`); M4 brings it into `kaful/` as a reference implementation.
- Remove or guard `rag_frameworks/progpy/progpy_rag/test_retrieval.py` so that importing it
  can never make API calls.
- Delete the dead and broken code listed in §8: `helpers.py:73-110`, `step_g_ddl`, the empty
  modules, stray imports, debug prints, commented-out code and snapshot folders. (Snapshot
  folders need approval.)
- **Accept:** the golden run is unchanged and the tests are green.

---

## Stage B — Foundation (new `kaful/` package)

### [ ] M4. Package skeleton and typed artifacts
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

### [ ] M5. LLM gateway, prompt registry, eval framework
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

---

## Stage C — Rebuild the phases (in pipeline order)

For each phase:
1. Build it in `kaful/phases/<phase>/`.
2. Give it typed input/output and evals for its prompts.
3. Report its cost against the M1 baseline.
4. Show parity or a documented improvement on eversys.

### [ ] M6. Ingest
- Extract raw page text with PyMuPDF and keep the printed page labels. Use **LLM extraction
  only for pages that need it** (tables, figures, garbled text), chosen by a cheap heuristic.
  This is expected to be the biggest cost cut.
- Chunk IDs are deterministic (a hash of file sha + page range + chunker version). An
  **index manifest hash** is recorded in every downstream artifact; a mismatch triggers a
  warning, or an error with `--strict`.
- The machine config's document list is validated against the files on disk. This fixes the
  CNC list (§5).
- **Accept:** CNC index rebuilt (with approval), and each chunk stores raw text + labels.
- **M1 baseline → target:** phase-1 cost cut by ≥70% at equal or better grounding.
  eversys $2.18 → ≤$0.65; CNC $15.85 → ≤$4.75 (CNC baseline partly estimated,
  range $14.46–$17.61; the target uses the point value).
  - *Grounding* here is a fixed set of evidence quotes taken from today's saved checkpoints
    and checked against both the old index and the new one, so phase-2 changes can't move it.
  - Also report *page coverage*: the share of manual pages represented in the index.

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
  `verbatim`, `near` or `unsupported`.
- **Accept:** the grounding numbers for all 4 machines are recomputed and compared with §5
  (today: eversys 19/27, air_compressor 10/19, CDV 10/34, CNC 7/29).
- **M1 baseline → target:** CNC 2b+2c cost cut by ≥50% (retrieval + caching).
  $2.17 ($0.56 + $1.61) → ≤$1.08.
- *Note (M2):* the tiny_pump fixture machine produced 30 schema fields from a manual that
  lists 7 channels. Use it as an eval case for sensor-contract over-extraction.

### [ ] M8. Scope
- Triage runs **before** any per-component physics work and decides between full model,
  connection only, and excluded. Each decision has a reason and a manual override.
- **Accept:** the cost of later phases scales with the number of in-scope components only.
- **M1 baseline → target:** no per-component LLM spend on out-of-scope components.
  eversys $0.16 (4b on 4 components triage drops) → $0; CNC $0.07 (4b on
  `housing_assembly`) → $0.

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
  prognostic horizon, and RMSE of the median RUL.
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
- The demo report is built from the new artifacts, with fan charts drawn from saved samples.
- Once the new pipeline reaches parity on eversys and CNC, delete the legacy
  `phases/`, `run.py`, `utils/` and `prompts/` (with approval).
- **Accept:** CI is green and the legacy code is gone.

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
- **Merge with the streaming production repo:** the `SensorContract`, `ParamSet` and
  template library are the natural seam.