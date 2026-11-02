Historical audit of the legacy pipeline as of 2026-11-02. Not updated as the revamp progresses; PLAN.md is the current state.

# Kaful — Codebase Overview

Read-only review of the working tree as of 2026-11-02 (branch `main`, HEAD `47cf9d7`, with
uncommitted changes to `utils/rag.py`, six eversys spec JSONs, and eversys `step_c_validation.json`).
Line numbers refer to the working tree. Where something could not be confirmed from code or
saved outputs, it is marked **(uncertain)**.

---

## 0. Findings that matter most

1. **Validation gates do not gate.** Phase 3 (`step_d`), phase 4 (`step_d`), and phase 6 (`step_c`)
   all record problems and then let the pipeline continue. Phase 6 builds the composite from every
   triaged component, including ones that failed runtime validation
   (`phases/phase6_codegen/__init__.py:46-52`).
2. **Phase 6 step C checks less than its docstring and CLAUDE.md say.** There is no out-of-bounds
   state check and no tracking-convergence check (`step_c_validate.py:178-250`, compare the
   docstring at lines 12 and 15). "Starts healthy" checks `event_state() == 1.0`, not the state
   values themselves (lines 131-139). Six eversys health states start at 100, not 1.0, and still pass.
3. **"Locked" simulator rates do not hold up in the saved specs.**
   - eversys: none of the four simulator wear rates (`grinder_wear_rate=1e-4`,
     `brewing_wear_rate=8e-5`, `boiler_scale_rate=5e-5`, `milk_contamination_rate=0.002`) appears
     in any current spec.
   - cnc_spindle: the four bearing rates are used correctly in the two bearing components, but the
     same values were also copied into `spindle_amplifier`, `position_coder`, `temperature_sensor`
     and `tool_interface` under other names. See §5.
4. **CNC spindle has two components failing and an event already triggered at t=0** in both the
   saved results and a live re-run (§6). eversys passes 8/8.
5. **The CNC vector index no longer matches the CNC checkpoints.** The live Chroma collection has
   48 chunks, all from the two MTConnect PDFs, and none from the Fanuc manual or ISO 10816. The
   checkpoints were produced on 2026-05-19, and the index was rebuilt on 2026-07-31.
6. **The RUL output cannot support a percentile fan chart.** Phase 7 saves only mean/std/min/max
   of time-to-event per event. Those statistics cover only the samples that reached the event,
   and the hit count is not saved (§4).

---

## 1. Directory map

| Path | Purpose |
|---|---|
| `run.py` | CLI entry (`python run.py <machine_id> [--from/--only phaseN] [--force]`). Runs the phases in order and reloads earlier phases from disk by calling their `run()` again (`run.py:89-136`). |
| `config.py` | `get_machine_config()` builds a dict of every checkpoint path for a machine, plus phase-7 constants. The model-name constants are defined but unused (§7). |
| `Requirements.txt` | Unpinned dependency list (capital R; README says `requirements.txt`). |
| `Readme.md` | Setup and usage notes. Partly stale: `cd pipeline`, `state.py`, `.env.example`, filename casing. |
| `docs/tweaks.md` | Two-bullet known-limits note (open-loop estimation, no failure data). |
| `utils/llm.py` | `call_claude()`: a single-turn Messages call with backoff on 429/529. |
| `utils/rag.py` | `ManualRAG`: TOC- or heading-based PDF chunking, Claude re-extraction of each slice, OpenAI embeddings, Chroma store and retrieval. |
| `utils/parsing.py` | `parse_json`, `strip_fences`, `is_valid_python`. |
| `utils/helpers.py` | `dedupe_chunks`, `chunks_to_context`, `load_domain_context`. Lines 73-110 are dead and broken (§8). |
| `utils/progpy_rag.py` | `get_framework_context()`: 10 fixed queries against the ProgPy Chroma store, plus hand-written `MANUAL_CHUNKS_*` rules. |
| `utils/audit_chunks.py` | Standalone CLI that audits indexed chunks for refusal or meta-commentary pollution. Not called by the pipeline. |
| `phases/phase1_ingest.py` | Phase 1 wrapper around `ManualRAG.index_manual()`. |
| `phases/phase2_schema/` | Steps a–g: retrieve → context → 5× extract → normalize → merge → finalize. Step g (DDL) is not wired in. |
| `phases/phase3_simulate/` | Steps a–d: retrieve → plan → simulator codegen → run and validate. |
| `phases/phase4_twin/` | Steps a–d: comprehension → per-component physics → merge and resolve units → structural validation. |
| `phases/phase5_triage/` | Steps a–c: field→component assignment → triage → priority and manual filter. |
| `phases/phase6_codegen/` | Steps a–d: spec → component code → runtime validation → composite (connections, external inputs, codegen, validation). |
| `phases/phase7_estimate/` | Steps a–c: load CSVs → open-loop ensemble propagation → ProgPy Monte Carlo RUL. |
| `prompts/` | Empty package (`schema.py` is 0 bytes). |
| `machines/<id>/queries.py` | Per-machine config: `manuals`, query lists, `domain_context`, `manual_exclusions`/`manual_inclusions`, optional noise settings. |
| `machines/<id>/manual/`, `rag/` | PDFs, chunk PDFs, and the Chroma DB (gitignored). |
| `machines/<id>/outputs/` | All checkpoints. Partly tracked: CSVs, generated components, and estimates are gitignored. |
| `machines/cnc_spindle/outputs/codegen_SNAPSHOT_*` | Manual snapshots of earlier codegen runs (one also has `specs_OLD/`). |
| `rag_frameworks/progpy/progpy_rag/` | Ingestion for the ProgPy docs RAG (GitHub and docs crawlers, chunker, `manual_chunks.py`, `ChromaDBStore`), plus `test_retrieval.py`. |
| `rag_frameworks/progpy/pipeline/` | `preprocessor.py` (CSV → DataSummary) and its test. Not imported by the main pipeline. |
| `data_pipeline/` | Separate data-first prototype on PHM 2010 milling data. Only phases 1–2 exist (§8). Contains `reconcile.py`. |
| `demo/` | `build_report.py` (offline HTML report from checkpoints), `live_validate.py` (re-runs phase 6 step C with no writes). `report.html` is gitignored. |
| `tests/unit`, `tests/eval` | Empty, untracked directories. |
| `pipeline.ipynb` | Gitignored notebook (not reviewed). `artifacts/` is empty. |

---

## 2. End-to-end data flow

Every step follows the same pattern: if the output file exists, load it and return; otherwise
compute, save, and return. "LLM" marks steps that call Claude.

| Phase / step | Entry point | Inputs | LLM | Output (under `machines/<id>/`) |
|---|---|---|---|---|
| 1 ingest | `ManualRAG.index_manual` (`utils/rag.py:80-230`) | PDFs listed in `queries.manuals` | yes, one call per chunk | `rag/vector_db`, `rag/chunks/*.pdf`, `rag/chunks/skipped_chunks.json` |
| 2a retrieve | `phase2_schema/step_a_retrieve.run` | `schema_queries` (k=4), `process_queries` (k=3) | – | `outputs/schema/step_a_chunks.json` |
| 2b context | `step_b_context.run` | chunk context | 2 calls | `step_b_context.json` |
| 2c extract | `step_c_extract.run` | context + 2b | 5 calls (`N_SCHEMA_RUNS=5`, line 18) | `step_c_runs.json` (raw strings) |
| 2d normalize | `step_d_normalize.run` | 2c | – | `step_d_normalized.json` |
| 2e merge | `step_e_merge.run` | 2d | – | `step_e_merged.json` |
| 2f finalize | `step_f_finalize.run` | 2e | – | `final_schema.json` |
| 2g DDL | `step_g_ddl.run` | – | – | **never called** (`phase2_schema/__init__.py:52-57`) |
| 3a retrieve | `phase3_simulate/step_a_retrieve.run` | `simulation_queries` (k=5), grouped by topic | – | `outputs/simulate/step_a_context.json` |
| 3b plan | `step_b_plan.run` | schema, context, `domain_context` | 1 call | `step_b_plan.json` |
| 3c codegen | `step_c_codegen.run` | schema, context, plan | 1 call; raises on a syntax error, no retry | `simulator.py` |
| 3d run + check | `step_d_validate.run` | runs `simulator.py` as a subprocess | – | `*events_*.csv`, `*maintenance_*.csv`, `step_d_validation.json` |
| 4a comprehend | `phase4_twin/step_a_comprehend.run` | `twin_comprehension_queries` (k=4) | 1 call | `outputs/twin/step_a_comprehension.json` |
| 4b physics | `step_b_physics.run` | 3 generated queries per component (k=3) | 1 call per component | `twin/components/<name>.json` |
| 4c resolve | `step_c_resolve.run` | 4a + 4b | 1 call (units) | `step_c_machine_model.json` |
| 4d validate | `step_d_validate.run` | 4c | – | `step_d_validation.json` |
| 5a assign | `phase5_triage/step_a_assign.run` | schema + components | 1 call | `outputs/triage/step_a_field_assignments.json` |
| 5b triage | `step_b_triage.run` | machine model + 5a | 1 call | `step_b_triage.json` |
| 5c filter | `step_c_filter.run` | 5b + manual overrides | – | `step_c_full_components.json` |
| 6a spec | `phase6_codegen/step_a_spec.run` | 5c, `simulator.py` constants, plan drivers | 1 call per component | `outputs/codegen/specs/<name>.json` |
| 6b code | `step_b_generate.run` | specs + ProgPy RAG context | Sonnet, up to 3 attempts per component | `codegen/components/<name>.py` |
| 6c validate | `step_c_validate.run` | specs + code | – | `step_c_validation.json` |
| 6d composite | `step_d_composite.run` | 5c, specs, machine model | 0–1 call (nominal inputs) + 1 Opus call | `step_d_connections.json`, `step_d_external_inputs.json`, `composite_model.py`, `step_d_validation.json` |
| 7a load | `phase7_estimate/step_a_load.run` | `*events_*.csv` + composite outputs | – | `outputs/estimate/<profile>/step_a_observations.json` |
| 7b propagate | `step_b_estimate.run` | 7a (only `times` is used) + composite | – | `<profile>/step_b_estimated_states.json` |
| 7c RUL | `step_c_rul.run` | 7b `final_ensemble` + external inputs | – | `<profile>/step_c_rul.json` |

Definitions: 25 steps (1+7+4+4+3+4+3); 24 are executed, because 2g is not wired in.

**Rerun caveats (relevant to "skips straight through"):**
- Without `--from`, phase 1 always runs. `index_manual` rebuilds the chunk list locally and calls
  Claude for any chunk ID that is not already in Chroma (`rag.py:146-175`). For CNC, only 48
  chunks are indexed, so a full run would re-extract the Fanuc and ISO PDFs. Chunk IDs are a
  running counter across the `manuals` list (`rag.py:147`), and CNC's list names
  `m110_03_safety.pdf`, which is not on disk. **(uncertain)** whether the existing IDs would
  line up. Use `--from phase2` for demos.
- Phase 6 step B calls `get_framework_context()` before its cache check
  (`step_b_generate.py:134`). That means 10 OpenAI embedding queries even when every component
  is cached.
- Phase 7 derives the profile name as `csv.stem.split("events_", 1)[-1]` (`step_a_load.py:53`,
  introduced in `108ebf4`). The CNC and CDV estimate folders use the full stem
  (e.g. `cnc_spindle_events_heavy_duty`), so phase 7 for those machines would recompute into new
  folders instead of loading. eversys folder names match.

---

## 3. The twin

**Shape.** A ProgPy `CompositeModel` assembled in the LLM-generated
`outputs/codegen/composite_model.py`. The module exposes `composite_model` and
`future_loading_eqn`. Each part is an LLM-generated `PrognosticsModel` subclass in
`codegen/components/<name>.py`. Saved sizes (`step_d_validation.json`):

| Machine | states | inputs | outputs | events | events < 1.0 at t=0 |
|---|---|---|---|---|---|
| eversys_coffee_machine | 22 | 18 | 12 | 17 | none |
| cnc_spindle | 39 | 14 | 19 | 29 | `cylindrical_roller_bearings.scoring_raceway_rolling_surface` |
| air_compressor | 44 | 8 | 28 | 33 | none |
| cdv | 20 | 6 | 21 | 27 | 7 |

**States.** Defined per component in the spec (`step_a_spec.py:44-60`) as one of four types:
- `degradation`: the prompt says it runs 1.0 → 0.0.
- `accumulation`: has a capacity.
- `tracking`: first-order lag, coefficient capped at 0.5.
- `static`.

The 1.0 → 0.0 convention is not enforced. eversys has six health states that start at 100
(`coffee_boiler.temperature_stability`, `steam_boiler.pressure_stability`,
`water_pump.pump_efficiency`, `powder_unit.dispensing_accuracy`,
`cleaning_system.system_cleanliness`, `cleaning_system.hygiene_safety`).

**Inputs and outputs.** Ports come from triage (`step_b_triage.py:34-60`) and are of two kinds:
- **observable**: schema fields assigned in 5a.
- **internal**: flow-path boundary signals from 4a.

Composite wiring is deterministic and uses exact name equality: an output port name must equal an
input port name on another component (`step_d_composite.py:186-208`). Any input left unconnected
becomes a constant external input in `future_loading_eqn`.

**Events.** Taken from the phase 4 fault conditions via `candidate_events`. `event_state()` returns
values in [0, 1] and `threshold_met()` returns bools (`step_b_generate.py:75-84`).

**Where degradation is defined.** It is defined in three places, which are linked only by prompts:
1. **Simulator** (phase 3): LLM-written `simulator.py` generates the synthetic data, e.g.
   eversys `simulator.py:36-39`, `603-631`, and CNC `simulator.py:158-160`.
2. **Spec text** (6a): `state_transition_logic` in plain English.
3. **Executable model** (6b): `next_state()` or `dx()` in each component.

**Observation model.** Each component's `output()`. Phase 7 never compares outputs with the CSVs
(it runs open-loop). `measurement_noise` from `queries.py` is loaded into cfg
(`config.py:113`) but nothing reads it.

**Where parameters come from.** None are fitted to data.

| Parameter | Source |
|---|---|
| Simulator rates and thresholds | Written by the LLM in phase 3 (`simulator.py`) |
| Spec parameter defaults | LLM in 6a. Prompted to copy the regex-extracted simulator constants (`step_a_spec.py:152-188`); otherwise derived from an LLM-written `degradation_timescale` |
| Code constants | LLM in 6b, from the spec. The prompt says explicit rates are "per 1000 operating hours" (`step_b_generate.py:45-49`) |
| External input values | Midpoint of the triage `typical_value`, else an LLM guess, else `1.0` (`step_d_composite.py:253-283`) |
| Units | From the manual extraction. Missing units are filled by an LLM call that picks "the most physically appropriate unit" (`step_c_resolve.py:21-46`) |
| Phase 7 noise | `process_noise_default` from `queries.py` (default 1e-4); 1% initial perturbation hardcoded; seed 42 (`step_b_estimate.py:75-87`) |

**Time base (uncertain).**
- The "one step = one operating hour, rates per 1000 h" convention in the codegen prompt matches
  the CNC and air-compressor simulators (`/ 1000.0` on hours).
- It does not match eversys, whose rates are "per coffee event" (`simulator.py:36-39`).
- Phase 7 uses one CSV row = one step with `dt=1` (`step_a_load.py:128`, `step_b_estimate.py:99-104`).

So forecast "steps" are not consistently hours or events across machines.

---

## 4. Uncertainty

| Where | What is sampled | What is saved | What is collapsed or discarded |
|---|---|---|---|
| Phase 1 | Claude extraction at default temperature | the extraction text only | – |
| Phase 2c | 5 independent extractions at `temperature=1.0` (the default in `utils/llm.py:39`) | all raw runs (`step_c_runs.json`); per-field run counts (`step_e_merged.json.field_counts`) | `final_schema.json` keeps one representative per field and no count or confidence (`step_f_finalize.py:56-79`) |
| Phase 3 | numpy `rng` inside the simulator | per-event CSVs and the maintenance log | – |
| Phase 4–6 | single LLM call per item at `temperature=0.2` | single outputs | no ensembles |
| Phase 7b | `ENSEMBLE_SIZE=100` members; initial perturbation `N(0, max(1%·|x0|, 1e-3))`; ProgPy `apply_process_noise` with scalar 1e-4 per state | mean and std of every state every `ESTIMATE_FREQ=50` steps; mean `event_state` per event; **final ensemble, all 100 members** | per-member trajectories; ensemble spread of event states |
| Phase 7c | ProgPy `MonteCarlo`, `n_samples=50` drawn from the final ensemble, `horizon=10000`, `dt=1`, `save_freq=5000` (`step_c_rul.py:101-111`) | per event: `rul_mean`, `rul_std`, `rul_min`, `rul_max`, or `already_triggered`, or `beyond_horizon` | individual time-of-event samples, percentiles, trajectories, and how many samples reached the event |

Notes:
- **RUL statistics cover only the samples that reached the event.** `valid` drops samples that
  never reached it within the horizon (`step_c_rul.py:134-143`). Some samples may never fail
  within 10,000 steps, and that cannot be seen from the saved file.
- **"Already triggered" is judged on the ensemble mean state** (`step_c_rul.py:92-99`), not per
  sample.
- **A 90% band needs a re-run.** It cannot be drawn from saved outputs. It would require re-running
  7c (compute only, no API) with samples kept. The current demo shows mean bars with min–max
  whiskers (`demo/build_report.py:356-435`).
- **Observations are unused.** 7b ignores the observations that 7a builds
  (`step_b_estimate.py:72-73`), as documented in its docstring (lines 10-12).

---

## 5. Grounding

**What the "manual" in the index actually is.** Each Chroma document is Claude's re-extraction of
a PDF slice (`rag.py:164-175`, prompt at `384-407`), not raw page text. The only check on that
extraction is length ≥ 100 characters plus a list of refusal phrases matched against the first
400 characters (`rag.py:347-374`). "Grounded in the manual" therefore means grounded in an LLM
transcription of the manual.

**Evidence fields.**
- Phase 2c, 4a, and 4b ask the LLM for `evidence` and `source_pages` (`step_c_extract.py:53-54`,
  `step_a_comprehend.py:40,56,65`, `step_b_physics.py:41,50,58,66`).
- Pipeline code never checks them.
- The demo report checks them: it normalizes the text and looks for a word-sequence substring in
  the retrieved chunks (`demo/build_report.py:174-181`). Re-computed here:

| Machine | schema fields whose evidence appears verbatim in `step_a_chunks.json` |
|---|---|
| eversys_coffee_machine | 19 / 27 |
| air_compressor | 10 / 19 |
| cdv | 10 / 34 |
| cnc_spindle | 7 / 29 |

(Non-matches may be paraphrases, which the prompt allows, rather than fabrications. This check
cannot tell the two apart.)

**Page references.**
- Chunk metadata stores physical PDF page ranges (up to 8 pages per chunk, `rag.py:80`).
- Printed page labels were added recently (`rag.py:138-141`, `189-200`), but `retrieve_chunks`
  does not return them (`rag.py:255-264`).
- The eversys index predates that change and has no `printed_page_*` metadata.
- Phase 4b does not save which chunks it retrieved, so twin physics has no stored provenance.

**Simulator "ground truth" (6a).**
- `_extract_simulator_constants` regex-scans `simulator.py` for `name = <number>` lines whose
  name contains `rate` or `threshold`, or starts with `critical_` (`step_a_spec.py:159-161`).
- `_build_ground_truth_block` puts them, together with the plan's degradation drivers, into the
  prompt.
- Enforcement is **prompt-only**: nothing compares spec defaults with the extracted constants
  after generation.

Regex limits:
- It captures only the leading literal of an expression.
- `"rate" in name` also matches non-wear constants. For eversys that includes `base_failure_rate`,
  `base_manual_stop_rate`, and `cleaning_ball_consumption_rate`.
- Duplicate names overwrite each other silently.
- The whole machine's constant list goes to every component, which explains the cross-component
  copying below.

Measured against current specs:
- **eversys:** none of the four wear rates appear. For example, `ceramic_grinders` uses
  `base_wear_rate=4.63e-05` against the simulator's `grinder_wear_rate=1e-4`.
  - The six specs modified today in the working tree restructure states and parameters.
  - Three specs (`cleaning_system`, `powder_unit`, `touch_screen`) date from 2026-05-18, before
    the ground-truth block existed (`132d587`, 2026-06-28).
- **cnc_spindle:** `angular_contact_bearings` and `cylindrical_roller_bearings` carry
  `bearing_wear_rate=0.02`, `grease_degradation_rate=0.03`, etc., matching by name. The same four
  values also appear in:
  - `spindle_amplifier`, verbatim under the bearing names.
  - `position_coder`, `temperature_sensor` and `tool_interface`, renamed (e.g.
    `signal_degradation_rate=0.02`).

The simulator constants themselves are LLM-written. Phase 3d checks only schema field presence and
`typical_range` (`step_d_validate.py:98-161`). It does not compare them against the manual, and it
does not fail on violations. Saved results show >5% out-of-range for 2 eversys fields and
10 CNC fields.

**Other LLM-judgement inputs (not manual-grounded by design).**
- Unit resolution (`step_c_resolve.py:21-46`).
- Nominal input values (`step_d_composite.py:30-45`).
- The domain context strings in `queries.py`.

**Reconciliation steps in the main pipeline.**
- Ports: `_validate_ports` warns only (`step_a_spec.py:242-247`, `264-282`).
- Composite connections: exact name matching (`step_d_composite.py:186-208`).
- Phase 7 CSV→port mapping: exact suffix matching (`step_a_load.py:115-123`).

There is no fuzzy reconciliation. `reconcile.py` (snaps LLM names back to real channel and
feature names) exists only in `data_pipeline/phases/phase2_component_map/` and is used by that
prototype's step b and step d.

**Index drift.** The CNC Chroma collection currently has 48 chunks, all from
`mtconnect_part1.pdf` and `mtconnect_part2.pdf`. Another 208 chunk PDFs sit on disk. CNC's phase
2–4 checkpoints are from 2026-05-19 and the index was rebuilt on 2026-07-31. Any retrieval step
re-run for CNC would draw from a different corpus. `queries.py` lists `m110_03_safety.pdf`
(not on disk) and omits `nsk_bearing_selection_mounting.pdf` (on disk).

---

## 6. Tests

**What exists.**
- `tests/unit/`, `tests/eval/`: empty and untracked. No pytest config.
- `rag_frameworks/progpy/pipeline/test_preprocessor.py`: a script with 6 asserts covering
  `preprocessor.py`, which the main pipeline does not use. Run it with
  `cd rag_frameworks/progpy && python pipeline/test_preprocessor.py`.
  **Ran: exit 0.**
- `rag_frameworks/progpy/progpy_rag/test_retrieval.py`: prints ProgPy retrieval results. It needs
  `OPENAI_API_KEY` and the ingested store. **Not run** (network/API).
- The runtime checks closest to tests are phase 6 step C and `demo/live_validate.py`, which
  re-runs them with no writes and no API calls.

**Results observed** (conda env `pipeline311`, Python 3.11.15, progpy 1.7.1, run with `-B`):

| Machine | Result | Detail |
|---|---|---|
| eversys | 8/8 pass, 10.3 s | – |
| cnc_spindle | 5/7, 9.2 s | `position_coder`: `output()` fails at t=0 (`could not convert string to float: np.str_('e')`). `cylindrical_roller_bearings`: `event_state()` is 0.0 at t=0 for `scoring_raceway_rolling_surface`, plus a warning that `cage_material_wear` changed only 1e-6. |
| air_compressor | 14/14 pass | saved results only, not re-run |
| cdv | 3/7 pass | saved results only, not re-run; `heating_zones`, `vacuum_pump`, `mass_flow_controllers`, `temperature_controllers` fail |

Other checks:
- Syntax: all 108 tracked `.py` files parse except `data_pipeline/store/schema.py`, which contains
  a prose sentence, not code.
- Weak "degrades" checks: passing components include states that barely move over 1,000 steps.
  Examples: `ceramic_grinders.grind_consistency` 4e-6, `spindle_amplifier.cutting_capability_degradation`
  3e-6, and `coffee_boiler.temperature_stability` 0.00022 on a 0–100 scale.
  - The "barely changed" threshold is an absolute 1e-6 and only warns (`step_c_validate.py:233-234`).
  - Inputs are all set to 1.0 regardless of unit (`step_c_validate.py:279-285`).

**Not covered by anything:** `utils/parsing.py`, the phase-2 merge logic, RAG chunking and TOC
parsing, phase 7, and composite wiring.

---

## 7. Config, environment, dependencies, hardcoding

**Environment.**
- `.env` holds `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. There is no `.env.example`.
- `EMBEDDING_MODEL` can optionally be set to override the default (`rag.py:62`).
- Python 3.11, because ProgPy does not support 3.13 (per the README).
- eversys `__pycache__` contains both cpython-311 and cpython-313 files, so it has been imported
  from two interpreters.

**Dependencies** (`Requirements.txt`, all unpinned): anthropic, openai, chromadb, pypdf, pymupdf,
python-dotenv, duckdb, pandas, numpy, progpy, beautifulsoup4, requests, scipy.
**(uncertain)** `data_pipeline` writes parquet, which needs pyarrow or fastparquet; neither is
listed.

**Model IDs.**

| Use | Model | Where set |
|---|---|---|
| Default `call_claude` | `claude-sonnet-4-6` | hardcoded, `utils/llm.py:36` |
| PDF extraction | `claude-sonnet-4-6` | hardcoded, `utils/rag.py:45` |
| Composite codegen | `claude-opus-4-6` | hardcoded, `step_d_composite.py:328` (the only Opus use) |
| Manual embeddings | `text-embedding-3-large` | `config.EMBEDDING_MODEL` |
| ProgPy store embeddings | `text-embedding-3-small` | `rag_frameworks/.../chroma.py:56` |

`config.DEFAULT_MODEL`, `CODEGEN_MODEL` and `HAIKU_MODEL` are never used. (`phase1_ingest.py:20`
imports `DEFAULT_MODEL` but does nothing with it.)

**Tunables** (`config.py:29-32`): `ENSEMBLE_SIZE=100`, `ESTIMATE_FREQ=50`, `MC_SAMPLES=50`,
`PREDICTION_HORIZON=10000`.

**Hardcoded to a machine or dataset.**
- Module-cache purge lists of coffee-machine names (`"ceramic", "brewing", "coffee", "steam",
  "water", "milk", "powder", "cleaning"`): `step_d_composite.py:351-354`, `step_a_load.py:103-106`,
  `step_b_estimate.py:148-151`, `step_c_rul.py:161-164`.
- The per-1000-operating-hours rate convention in the generic codegen prompt
  (`step_b_generate.py:45-49`). It matches the CNC and air-compressor simulators, not eversys.
- Step 2g: the `brewing_events` table and coffee example (`step_g_ddl.py:23-65`). Not wired in.
- `"/5 runs"` in a print (`step_f_finalize.py:50`); `N_SCHEMA_RUNS=5` is local, and its config
  import is commented out (`step_c_extract.py:15-18`).
- CSV discovery uses the glob `*events_*.csv` (`step_d_validate.py:69`, `step_a_load.py:37-38`).
  CDV has hand-made `*_filtered.csv` files next to the raw ones, and both would match.
- Page-label regexes tuned to particular manuals (`rag.py:496-501`); section-number limits of
  major ≤ 20, minor ≤ 30 (`rag.py:628`).
- The demo defaults to eversys, `GROUNDING_FIELD="extraction_time"` and
  `HEALTH_PROFILE="heavy_commercial"` (`demo/build_report.py:27-29`); `live_validate.py:27` also
  defaults to eversys.
- `data_pipeline/` is built around PHM 2010.

---

## 8. Smells

**Dead or broken code.**
- `utils/helpers.py:73-110`: `load_module_from_path`, `container_to_dict` and
  `find_component_class` are never called, and they use `os`, `sys`, `inspect` and `importlib`
  without importing them, so calling them would raise `NameError`.
- `phases/phase2_schema/step_g_ddl.py` is never called.
- `prompts/` is empty.
- `data_pipeline/store/{__init__,sqlite}.py` are empty; `schema.py` holds prose.
- `data_pipeline/run_data.py:128-163` imports `phase3_parameterize`, `phase4_codegen`,
  `phase5_estimate`, `phase6_rul` and `phase7_validate`, none of which exist.
- `rag_frameworks/progpy/pipeline/preprocessor.py` is not used by the main pipeline.
- `measurement_noise` (config) and 7a's `observations` are computed but never consumed.
- `step_c_validate.py:25` has a stray `from xml.parsers.expat import model`.
- `step_b_physics.py:14` imports `importlib.util` and never uses it.
- Commented-out code: `step_c_extract.py:15`, `step_a_load.py:39-40`, `step_a_load.py:64`,
  `step_d_validate.py:70`.
- Leftover debug print: `step_a_retrieve.py:50-57` ("test (delete this later)").
- The composite stability, degradation and event-progression results are printed but not saved
  (`step_d_composite.py:379-438`).

**Duplication.**
- The `queries.py` loader is re-implemented 7 times: `config.py:48-55`, `helpers.py:61-70`,
  `rag.py:293-303`, `phase2 step_a:71-99`, `phase3 step_a:66-84`, `phase4 step_a:147-163`,
  `phase5 step_c:97-111`.
- `_load_composite` appears 3 times in phase 7, plus once inline in 6d.
- `_normalize_field_name` is duplicated (`step_d_normalize.py:86`, `step_e_merge.py:145`).
- Snake→Camel class naming appears 3 times (`step_b_generate.py:174`, `step_c_validate.py:267`,
  `step_d_composite.py:166`).
- `data_pipeline/utils/{llm,parsing}.py` are near-copies of `utils/`.

**Contradictions inside prompts.**
- `step_b_generate.py:44-55` states the degradation rules twice, with a `- -` typo and two
  different formulas (`rate_per_step` vs `rate`).
- Line 100 ("rates must be derived from timescale, never picked arbitrarily") conflicts with the
  explicit-rate rule at lines 45-49.

**Gates that do not gate.**
- Phase 4 "issues — must fix" (`run.py:105-106`).
- Phase 6 failures (`phase6_codegen/__init__.py:47-52`). Status `missing` is not counted either.
- Phase 3 report (`step_d_validate.py:77-93`).

**Docstring and behaviour mismatch.** `step_c_validate.py:6-15` claims out-of-bounds and
tracking-convergence checks that the code does not perform.

**Lossy normalization and parsing.**
- Units pass through `_normalize_field_name` (`step_d_normalize.py:121`), so `°C`→`c`,
  `m/s`→`ms`, `%`→`""`.
- `_parse_range_midpoint` drops minus signs (`step_d_composite.py:296`).
- `parse_json` uses a greedy `\{.*\}` (first `{` to last `}`), not "first object"
  (`parsing.py:74-80`).

**Shared mutable state and globals.**
- Module-level Anthropic client in `utils/llm.py:25-32` (and the `data_pipeline` copy).
- `sys.path` is mutated in `phase1_ingest.py:18`, `progpy_rag.py:45-47` and the demo scripts.
- The `sys.modules` purge above.
- `_merge_physics` makes a shallow copy and so mutates the step 4a dict (`step_c_resolve.py:135`).

**Unsandboxed execution of generated code.** LLM-generated code is executed in-process in 6c, 6d
and phase 7, and the simulator runs via `subprocess` (`step_d_validate.py:50-55`). Nothing is
sandboxed.

**Functions longer than 80 lines.**

| Lines | Function |
|---|---|
| 153 | `phase4_twin/step_d_validate.py:31-183 run` |
| 151 | `utils/rag.py:80-230 index_manual` |
| 137 | `utils/rag.py:736-872 _build_section_chunks` |
| 93 | `rag_frameworks/progpy/pipeline/preprocessor.py:257-349 build_data_summary` |
| 93 | `phase6_codegen/step_d_composite.py:346-438 _validate_composite` |
| 92 | `data_pipeline/.../step_d_progpy_roles.py:109-200 run` |
| 91 | `run.py:57-147 main` |
| 90 | `data_pipeline/phases/phase1_ingest/normalize.py:75-164 run` |
| 88 | `phase6_codegen/step_c_validate.py:88-175 _validate_component` |
| 86 | `utils/rag.py:527-612 _parse_toc` |
| 86 | `data_pipeline/.../step_b_component_map.py:89-174 run` |
| 82 | `config.py:34-115 get_machine_config` |
| 81 | `phase5_triage/step_a_assign.py:64-144 run` |

**Magic numbers (selection).**
- RAG: 3 s delay between chunks, 60 s·2ⁿ rate-limit backoff, 8-page and 5-page chunk sizes,
  100-character minimum, 400-character refusal window (`rag.py`).
- Merge thresholds 0.90 / 0.62 / 0.55 and weights 0.6 / 0.4 (`step_e_merge.py:23-26, 185`).
- Validation: nominal input 1.0; thresholds 0.9 / 1e-6 / 100 (`step_c_validate.py`).
- Composite checks: 10 / 500 / 5000 steps and 1e-9 (`step_d_composite.py`).
- Phase 7: 1% perturbation, 1e-3 floor, seed 42; `save_freq=5000`.

**Repo hygiene.**
- Three manual snapshot folders under `cnc_spindle/outputs/`.
- eversys has `touch_screen` spec and code even though the component is manually excluded.
- Three eversys specs are older than the ground-truth feature.
- The working tree has uncommitted spec and validation changes for eversys.

---

## Appendix: CLAUDE.md claims checked against the code

| Claim | Status | Evidence |
|---|---|---|
| 7 phases, ~25 steps, each checkpointed and skipped if done | Mostly true | 25 defined, 24 executed (2g not wired). See the §2 rerun caveats. |
| Simulator rates injected as locked "SIMULATOR GROUND TRUTH", prompt-only | True as stated | `step_a_spec.py:96-105, 152-188, 210-231`. Measured outcome in §5: no rate matches for eversys; cross-component copying for CNC. |
| Step C: all health values start at 1.0 | **Not as stated** | It checks `event_state()==1.0` (`:131-139`). Six eversys states start at 100 and pass. |
| Step C: 1,000 steps, no NaN/inf/out-of-bounds | **Partly** | NaN/inf yes (`:200-204`). Out-of-bounds is not checked. |
| Step C: actually degrades, sane rates | True, with weak thresholds | `:216-234`. Δ > 0 fails if ≤ 0, fails if > 0.9, warns if < 1e-6. |
| "No generated code runs until it passes" (report caption) | **False** | The composite is built from every component (`__init__.py:46-52`). CNC and CDV composites include failing components. |
| Step B retries on syntax errors | True | 2 retries, 3 attempts (`step_b_generate.py:111, 181-197`). No retry on runtime failure. |
| `parse_json`: fence stripping, first object | Mostly | Greedy regex, not strictly the first object (`parsing.py:74-80`). No Pydantic. `_validate_ports` warns only. |
| `reconcile.py` is in `data_pipeline/` only | True | Used by `data_pipeline` step_b/step_d only. |
| Phase 7 is forward open-loop Monte Carlo, not a particle filter | True | `step_b_estimate.py:10-12`, `step_c_rul.py:101-111` (ProgPy `MonteCarlo`). |
| Sonnet for most steps, Opus for the hardest codegen | True, narrowly | Opus is used only for composite assembly. Component codegen uses Sonnet. |
| Triage: full / simple / excluded | Incomplete | Four buckets, including `connection_only` (`step_b_triage.py:88`). |
| Every fact traces to manual, simulator, or ProgPy | **Not enforced** | Prompt-only. Evidence verbatim match is 7/29 to 19/27 (§5). Units and nominal inputs are LLM judgement by design. |
| Demo machine: CNC spindle | Risky | 2/7 components fail; event triggered at t=0; vector index drift. eversys passes 8/8, and the demo already defaults to it. |
| Showcase page and Render API | Cannot verify here | No Dockerfile or API code in this repo. |
