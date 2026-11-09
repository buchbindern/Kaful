# M7b. Understand follow-ups

## The problem it addressed
On the air_compressor hold-out, M7's understand phase found only 0.31 of the fault modes. Parts that the
maintenance schedule says to replace (oil, bearings, flexibles, float valve) were never extracted as
components, so their fault modes had nowhere to attach. Status and alarm signals also came out as
`observation` channels instead of events. M7b fixes both with general rules, checks them on the dev
machines, and tests them once on a new blind hold-out, CDV (a Tempress LPCVD furnace).

## What was built
- **Discrete states are events** (`d_build_artifacts.channels`): a `bool` or `category` channel the
  prompt calls an observation becomes an `event`. Its secondary role and stated period are dropped,
  and the change is noted in the report.
- **`understand.fault_modes` v4 proposes carriers**: a part that the documentation says wears, clogs,
  ages or must be serviced, and that the component list lacks. Carriers are merged across samples and
  joined to a built component with another name. A carrier is kept only with a quote from a machine
  document (`kind = manual`); one quoted only from a datasheet is dropped and listed in the onboarding
  review as "datasheet-only part, not confirmed in the machine manuals", so an engineer can declare it.
  v4 is locked on Haiku 5.5 after two passing eval runs.
- **`ItemCheck.found_by`** (`components` | `fault_modes` | None) marks carriers in the report and the
  onboarding review.
- **matcher-v4** (`kaful/evidence.py`): a hyphen between letters joins on both sides, so a quote that
  copies a line-break hyphen ("cof-fee") matches the joined word. The eval quote check uses the same rule.
- **kaful.toml pins the three understand prompts to Haiku 5.5** (M7 chose it by environment override).
- **CDV**: `machines/cdv/machine.toml` (five Tempress manuals, six vendor datasheets), its ingest, and
  blind golden labels (`tests/fixtures/understand/golden/cdv.toml`), reviewed and frozen before the run.

## The most important files
- `kaful/phases/p2_understand/b_merge_samples.py` (`merge_carriers`), `d_build_artifacts.py`
  (`carriers`, `_as_event`), `phase.py`.
- `kaful/prompts/understand/fault_modes/` (v4 prompt, model, eval cases `eversys_carriers`, `tp200_carrier`).
- `kaful/phases/p2_understand/tools/onboarding_review.py` (the datasheet-only section).
- `tests/fixtures/understand/golden/cdv.toml`.

## How to see it working
- `kaful understand score eversys_coffee_machine --run 20261109T053428Z-2b935d`.
- `kaful understand review air_compressor --run 20261109T053818Z-e2b50d`: carriers are marked.
- The CDV run `20261109T143015Z-596dd5` failed its unit gate, so `kaful understand score` refuses it
  (gates gate). Its measurements were made with a scratchpad script that calls the same `score()`.

## Checks
- `pytest -q`: 601 passed offline. mypy clean.
- fault_modes v4: both eval runs passed every threshold; the lock records them.

## Results

**Dev machines** (raw scores; CNC with the confident aliases of its pre-filled table):

| Measure | Floor | eversys 2b935d | CNC d55739 |
|---|---|---|---|
| Channels R / P | M7 floors (eversys) | 1.00 / 0.88 | reported only |
| Channel roles | CNC ≥ 0.90 | 1.00 | 0.69 (was 0.44); missed, ceiling about 0.75 without int alarm codes |
| Fault modes R / P | CNC ≥ 0.70 / 0.70 | 0.86 / 0.75 (top_piston is a valid new carrier, p125) | 0.67 / 0.55 (was 0.56 / 0.50); missed |
| Components R / P | M7 floors (eversys) | 0.90 / 0.83 | reported only |
| Carriers | eversys 1.00, CNC ≥ 0.80 | met | not reachable: the bearings are named only in the NSK catalogue, and a carrier needs a machine-document quote |

No second dev iteration was run: nothing in scope would have changed between iterations.

**air_compressor before/after** (not a hold-out any more; no iterating): fault-mode recall 0.31 → 0.38
raw, about 0.54 with confident aliases (0.62 with the unsure oil-filter alias); carriers 7 → 10 of 13;
roles 0.77 → 0.92. The oil came out as three carriers (a merge miss: brand name against "compressor oil").

**CDV hold-out**, run `20261109T143015Z-596dd5`, one run with the locked prompts. Numbers use the
pre-filled aliases: the first figure counts confident ones, the bracket adds the unsure ones.

| Measure | Floor | Result |
|---|---|---|
| Fault-mode recall | ≥ 0.60 | 0.07 (0.14): **missed** |
| Fault-mode precision | ≥ 0.70 | 0.20 (up to 0.70 if the unsure MFC and heating-element rows go to the run): **missed** |
| Carriers found | ≥ 0.80 | 5 of 13 = 0.38: **missed** |
| Channel roles | ≥ 0.85 | 0.67 (0.71): **missed** |
| Channel R / P | ≥ 0.70 / 0.60 | 0.35 / 0.50 (0.41 / 0.57): **missed** |
| Component R / P | ≥ 0.60 / 0.60 | 0.48 / 0.53: **missed** |
| Examples stored, reference-only items kept | 0 | 0 / 0: met |
| First attempts valid | ≥ 8 of 9 | 9 of 9: met |
| Understand run cost | ≤ $0.12 | $0.054: met |
| Phase status | passed | **failed**: unit `ºC` (U+00BA, the ordinal sign) on spike_temp_2 |

Per the plan, these misses are the new-machine estimate; they do not reopen tuning on CDV.

**Why CDV missed.** All three prompts ran in retrieval mode (CDV's documents are 167k and 382k tokens),
with about 40k tokens of context spread almost evenly across documents:
- `fault_modes` got 28 chunks, 14 of them from the six datasheets. Of the ten Tempress pages the golden
  fault modes cite, it saw none. It did see two maintenance schedules (m320 p41-42, m330 p29-30).
- Even so, v4 proposed no carrier from them: "Replace the O-ring on the Loaddoor" (monthly) and "Clean
  complete vacuum lines" (yearly) were in its context. Its only carriers were datasheet parts (Edwards
  pump oil and seals, the Pall membrane, an MKS valve seat), which the machine-document rule correctly dropped.
- `components` saw no maintenance or tube-change page, so process tube, paddle, flange, O-rings and
  the vacuum pump were missing. That left the fault modes nowhere to attach.
- `sensor_channels` drew most of its channels from one recipe-editor screenshot (m220 p88): torch and
  valve states of an atmospheric process, not this furnace's alarm list (m210 p13, not retrieved).

## API spend for M7b: about $0.54

| Item | Cost |
|---|---|
| CDV ingest (Haiku $0.061, Sonnet $0.300, embeddings $0.031) | $0.391 |
| CDV understand | $0.054 |
| Dev runs (eversys, CNC) | $0.037 |
| fault_modes v4 evals | $0.035 |
| air_compressor before/after | $0.021 |

## Decisions you made (2026-11-09)
- Carriers come from `understand.fault_modes` (option 1) and need a machine-document quote; a
  datasheet-only part is listed in the onboarding review, not dropped silently. `found_by` defaults to None.
- Vendor sheets are `datasheet`.
- Option A: eversys_carriers expects the water filter only; passing-mention carriers are a logged gap.
- Line-break hyphens match the joined word (matcher-v4).
- CNC speed_arrival_signal is an event; no sensor_channels v7 for the CNC int alarm codes.
- The CDV review: zone_temperature and gas_flow are operating_input with secondary observation;
  controllers and electronics with no stated wear are not components; condition-triggered events
  (MFC blockage by DCS condensation) are not fault modes; thermocouples stay not_a_component.

## Open issues
- **Retrieval starves the understand prompts on large document sets.** The context budget is spread
  evenly across documents, datasheets included, and the queries miss maintenance, tube-change and
  alarm-list pages. This is the main CDV cause, and the air_compressor handbook misses are the same gap.
- **v4 proposes no carrier from a schedule entry it can see** (CDV O-rings, vacuum line). This is a Haiku
  miss on a new machine, not found on the dev machines.
- **Unit gate:** `ºC` (ordinal sign) is not normalized to `°C`.
- **Merge misses:** dpc / digital_process_controller, dtc / digital_temperature_controller(_dtc),
  inline_gas_filter / filters, and the air_compressor oil.
- The CDV disagreement table (88 rows, 17 unsure) is pre-filled. It sits in the session scratchpad
  because a failed run gets no `understand_scores` output.
- Leftover failed replay run directories from step 1 (budget-reservation and key tests) are harmless.
  The `[llm.reuse."understand.fault_modes"]` rule in kaful.toml is stale but harmless.
- Golden follow-on to check: digital_process_controller and furnace_temperature_control were set to
  not_a_component by Claude, as a consequence of your controller rule.
