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

### [x] M7. Understand (+ sensor contract)
*Done 2026-11-09: eversys meets every exit floor; CNC (4132a8) and the air_compressor hold-out fail some, logged below and in M7b; docs/milestones/M7.md.*
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
  - *Updated 2026-11-09 (M7c, Natalie):* understand runs once per machine, at onboarding, so reading the
    full machine documents is a one-time cost. Per-machine targets for a full understand run (no cache),
    from M7c iteration 1's actual costs, rounded up: eversys ≤ $0.08 (actual $0.072, one whole part),
    air_compressor ≤ $0.07 ($0.066), CNC ≤ $0.70 ($0.644, 18 parts through the Batch API, run
    20261109T162051Z-e97f82), CDV ≤ $0.12 ($0.118 over its M7c runs bf8f5e and 33de55, 3 parts).
    CNC is now below the old $1.08.
- *Note (M2):* the tiny_pump fixture machine produced 30 schema fields from a manual that
  lists 7 channels. Use it as an eval case for sensor-contract over-extraction.
- *Eval case (M2, legacy bug, not fixed):* legacy 3a (`_organize_by_topic`) orders retrieved
  context by set iteration, so prompts change with `PYTHONHASHSEED`. The new understand
  phase must build every prompt and every artifact in a declared, sorted order (rule 14).
  Test it by building the same prompts under two hash seeds and comparing their hashes.
- **Known gaps at M7 close (2026-11-09; exit criteria and time-box in docs/proposals/M7-exit-criteria.md):**
  - CNC (last iteration 20261109T042103Z-4132a8, $0.036, confident aliases applied): channels R 0.73 / P 0.29,
    roles 0.44, fault modes 0.56 / 0.50, components 0.60 / 0.38; carriers met (bearings declared, position
    coder extracted). 41 disagreement rows are unsure, mostly which Fanuc data count as channels.
  - CNC channel precision: one quantity reported from several pages (diagnosis screen p200, trace list p730,
    signal list p217). sensor_channels v6 ("report each quantity once") was dropped (Natalie, 2026-11-09):
    Haiku lost per-model variant ranges in 3 of 4 eval runs; v5 stays locked. Later option: a code-side merge,
    measured offline first.
  - Hold-out fault-mode recall and channel roles: see M7b.
  - Parent accuracy has no floor and is low (eversys 0.63, hold-out 0.95 only on the parts it found): runs
    leave parts at the top level.
  - Merge misses left: eversys error_history / error_message (share only a page), ticks_per_second / flow and
    total_product_counter in run 1ec5d2; hold-out load_state / compressor_control_status.
  - Budget reservations reserve 7-8x actual for extract_page and make small budgets crawl through batch
    rounds: proposal docs/proposals/post-M7-budget-reservations.md.
  - For M9: golden rows marked documented = false (eversys grinder burrs, CNC motor overheating) are handbook
    priors. For M15: Sonnet locks and routing.
  - ruff: two unused names in tests/unit/test_harness.py and test_kaful_llm_gateway.py (not M7 code).

### [x] M7b. Understand follow-ups (first post-M7 item, logged 2026-11-09)
- **Closed under the time-box, floors missed (Natalie, 2026-11-09).** Missed: CNC channel roles (0.69 < 0.90),
  CNC fault modes (0.67 / 0.55 < 0.70 / 0.70), CNC carriers (bearings are catalogue-only); on the CDV hold-out
  every quality floor (fault modes, carriers, channel roles, channels, components) and phase status (unit gate).
  Met: eversys every floor; CDV examples stored, first attempts, cost. Record: docs/milestones/M7b.md,
  evals/results/understand_holdout/cdv/.
- **Fault-mode carriers from maintenance schedules.** On the air_compressor hold-out (run
  20261109T040027Z-19e094) fault-mode recall was 0.31: the maintenance-schedule parts (oil, motor bearings,
  flexibles, restrictors, float valve, dryer condenser) were not extracted as components, so their fault modes
  had no carrier (6 of the 9 misses). Proposed general fix, one of:
  - let `understand.fault_modes` propose a missing carrier component, with its own evidence from maintenance
    schedules and warning lists (checked by code like any component); or
  - give the components step the maintenance-schedule and warning-list pages.
- **Status and alarm channels are events, not observations.** Hold-out roles 0.77, CNC 0.44: signals such as a
  general alarm, overload status or a zero-speed signal came out as `observation` (bool).
- **A new hold-out.** air_compressor stops being a clean hold-out once these fixes are judged on it; CDV
  becomes the next hold-out (ingest, blind golden labels, one run with the locked prompts, never tuned).
- **Known gaps (logged 2026-11-09):**
  - A part named only in passing (e.g. the eversys milk system, from a residue sentence) is not proposed as a
    carrier: Haiku gave none in three eval runs, so eval case eversys_carriers expects the water filter only
    (Natalie, option A). In real runs the components step finds milk_system; the hold-out gap was
    maintenance-schedule parts, which fault_modes v4 handles.
  - CNC int alarm, warning and error codes (diagnosis 710, 712) come out as numeric observations; the
    bool/category rule cannot reach them. No sensor_channels v7 (Natalie, 2026-11-09): CNC role accuracy has
    a ceiling of about 0.75 without it.
  - Thermocouple drift: sensors are not components, so the CDV thermocouples are labelled not_a_component
    although the furnace manual checks them yearly (Natalie, 2026-11-09). A sensor that itself degrades may
    become an exception to the sensors rule later.
- **CDV hold-out result (run 20261109T143015Z-596dd5, 2026-11-09; the new-machine estimate, not tuned on):**
  fault modes 0.07 / 0.20, carriers 5 of 13, channels 0.35 / 0.50 (roles 0.67), components 0.48 / 0.53,
  $0.054; phase failed on unit `ºC`. Gaps it shows (details in docs/milestones/M7b.md):
  - retrieval spreads about 40k tokens evenly across documents, datasheets included, and misses the
    maintenance, tube-change and alarm-list pages (the air_compressor handbook misses are the same gap);
  - fault_modes v4 proposed no carrier from schedule entries it saw (CDV O-rings, vacuum line);
  - `ºC` (U+00BA) is not normalized to `°C`;
  - merge misses: dpc / digital_process_controller, dtc / digital_temperature_controller,
    inline_gas_filter / filters.

### [x] M7c. Understand reads the machine documents in full (planned and approved 2026-11-09)
*Done 2026-11-09 (closed regardless of the CDV score, Natalie): dev scores held except CNC precision
(channels 0.11, components 0.09, fault modes 0.31); CDV, seen once, raw fault-mode recall 0.07 -> 0.57 but
failed the unit gate; $1.01; docs/milestones/M7c.md.*
- **One architectural change:** machine documents are read whole, or in parts of whole TOC sections (at most
  `part_max_tokens` 80k) when they do not fit; every prompt samples every part (Batch API for several), and
  the existing merge combines them. Reference documents (datasheets, literature, catalogues) stay retrieved,
  for fault-mode mechanisms only.
- **General fixes:** unit spelling (look-alike degree signs, ℃/℉, Torr, sccm/slm as volume per time); merge
  misses (abbreviation / full name, a one-word generic carrier); a series or type of a built component
  folds into it (iteration 2).
- **Evaluation:** dev machines keep their scores; CDV is re-scored once, labelled "seen once, not a clean
  hold-out". At most 2 paid iterations; CNC cap $1.50 a run.
- **A clean new-machine estimate needs a fifth machine.** CDV has now been seen once (its M7b run and the
  table it produced), so its M7c score is not a hold-out result; a later milestone onboards a fifth machine,
  labelled blind before any run, as the next clean estimate.
- **Main gap: CNC channel precision.** Reading the full Fanuc manual extracts its interface-signal and
  parameter tables as channels (146 channels, precision 0.11, against 55 and 0.29 from retrieval). Fanuc
  interface signals can be read from the controller (PMC), so part of this is a labelling question to
  revisit: which readable signals the golden counts as channels.
- **CNC fault-mode precision varies run to run** (12 modes in run e97f82, 17 in f8692c, from fresh samples
  on the same parts). Part of it is the NSK catalogue, which lists many bearing damage types. Whether
  installation-caused or sudden types (indentation, fretting, corrosion) count as gradual fault modes is a
  labelling question to revisit (Natalie, 2026-11-09).
- **Units the CDV M7c run could not read** (its unit gate failed): `MTOR` and duration formats (`hh:mm:ss`).
  Upper-case standard flows and mTorr in any case were fixed before the merge (Natalie, 2026-11-09); the
  cache replay a2f4ea still fails the gate on these two.
- **Follow-up: product-family manuals.** A manual that covers a product line (Fanuc: motor series, amplifier
  types) yields every variant as a component. Let machine.toml declare the machine's configuration (e.g.
  motor series, amplifier type) and have understand keep only the matching variants; unmatched variants
  fold into their base. This is the general fix; the iteration-2 series/type fold covers only variants
  whose name says "series" or "type".

### [x] M8. Scope
- Triage runs **before** any per-component physics work and decides between full model,
  connection only, and excluded. Each decision has a reason and a manual override.
- **Accept:** the cost of later phases scales with the number of in-scope components only.
- **M1 baseline → target:** no per-component LLM spend on out-of-scope components.
  eversys $0.16 (4b on 4 components triage drops) → $0; CNC $0.07 (4b on
  `housing_assembly`) → $0.
- *Eval case (M2):* tiny_pump's manual says it has exactly two main components, yet legacy
  4a found 11, and 5b kept all 11 as full models. Use it as an over-decomposition eval:
  scope should keep 2 full models, with sensors as connections or observations, not
  components. *(M8, Natalie 2026-11-09):* the two are the inlet filter F1 and the impeller
  under P1 (the wear lives in the impeller; P1 is connection_only as its parent).
- *Follow-up (M8, Natalie 2026-11-09):* understand.fault_modes should fill `observable_by`
  itself. On the M7c runs it linked no observation channel on eversys or CNC, so scope's O
  rests on channels that name the component (lower confidence, flagged in the review).
- *Exit criterion 1 restated (Natalie 2026-11-09):* every extracted component with a documented
  gradual fault mode is either full_model, or connection_only with the flag "nothing to
  calibrate or observe" (overridable by declaring a channel or a life). The original wording
  (full-model recall 1.0) assumed every such component is modelable; air_compressor
  drive_motor (overheating, no channel, no stated life) shows it isn't. Recall is still reported.
- *Fixed (M8 close, Natalie 2026-11-09):* a machine-wide statement counted as L for every component
  citing its page, so air_compressor restrictors_rf1_rf2 got its L from the p29 (label 27) row "Have all
  flexibles/flexible joints inspected". Now only a statement that names the machine, or names no part at
  all, links through a cited page; a row naming another part links to that part and gives no life.
- Done 2026-11-09: rules only, $0; criteria met on eversys, air_compressor, CNC and tiny_pump
  (docs/milestones/M8.md); downstream iterates `Scope.full_model_components`.

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
- *Decided (M9, Natalie 2026-11-09):* the template protocol and three families (accumulation,
  arrhenius, bearing_fatigue) are built in M9, so calibration and sensitivity can simulate; one
  template per component, family chosen by code from the fault modes' mechanism words. Templates
  keep the M4 design: damage D with a unit and D_limit; health = 1 − D/D_limit is derived. Time
  units confirmed: eversys product, air_compressor hour, CNC hour.
- *Later:* per-mode health states as competing risks (RUL = min across modes), instead of one
  template per component.
- *Gap (M9 checkpoint 3):* understand extracts no product settings or channel ranges on the dev
  machines (no channel has a documented range), so value roles give no thresholds or nominal inputs
  yet. eversys prints its water quantity per product (p72 (label 72) "Adjustment range is 0 to 1000
  ticks. Ticks are approximately equal to ml.", p78 (label 78) 25 ml espressos), but it is not in any
  artifact, so water per product uses a flagged wide default until [[spec]] declares it or understand
  extracts settings.

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