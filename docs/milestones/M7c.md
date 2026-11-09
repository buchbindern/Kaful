# M7c. Understand reads the machine documents in full

## The problem it addressed
On CDV (M7b), every understand prompt ran in retrieval mode, with about 40k tokens of context spread evenly
over eleven documents. The prompts never saw the maintenance, tube-change and alarm-list pages, and
fault-mode recall was 0.07. M7c makes one architectural change: the machine documents are read in full.
It also adds the general fixes that the CDV run exposed (unit spelling, merge misses).

## What was built
- **Machine documents in parts** (`a_select_context.select_parts`, `split_parts`). If the machine documents
  fit `whole_max_tokens`, the prompt gets them whole, exactly as before. Otherwise they are packed, in
  document order, into parts of at most `part_max_tokens` (80k).
  - A part holds whole TOC sections. A section larger than a part splits between chunks, and a chunk is
    never cut.
  - Every prompt samples every part, through the Batch API when there are several parts.
  - Fault modes also get a retrieved context from the reference documents (datasheets, catalogues), for
    mechanisms only.
- **Merging across parts** (`b_merge_samples.merge`).
  - An item's agreement is the share of samples that found it in its best part.
  - `parts_found` records how many parts found it, and the onboarding review shows it as "k of n".
  - `UnderstandReport` schema 2 keeps one context per part per prompt; schema-1 reports still load.
- **Unit spelling** (`kaful/units.py`).
  - Look-alike degree signs (º, ˚), a spaced "° C" or "deg C", ℃ and ℉ are all read as degrees.
    Before, `˚C` and `° C` were silently read as coulomb.
  - Torr is recognised.
  - Standard flows (sccm, slm, slpm) are read as volume per time.
  - Any crash inside pint becomes a `UnitError`.
- **Merge fixes** (`helpers/canonical_names.py`).
  - An abbreviation joins the full name it is printed with: dpc / Digital Process Controller (DPC).
  - A full id that ends in its own initials joins the id without them: digital_temperature_controller_dtc.
  - A one-word generic carrier joins the only built component named so: filters → inline_gas_filter.
  - Iteration 2 added a fold for product lines. A component whose id names a series or type of another
    component folds into it: spindle_motor_alpha_i_series → spindle_motor, spm_type_a → spm. A quote or
    printed name must print the base's words and the word "series" or "type"; the longest base wins.
    This fold overrides "listed side by side" and also applies to carriers. The marker words are
    `merge_variant_words`.
- **understand.fault_modes v5.** max_tokens was raised from 16000 to 32000, with the text unchanged. The CDV
  run bf8f5e had stopped at 16000. The reuse rule now covers v4 → v5, so both eval runs passed on reused
  v4 replies at $0 billed, and the prompt is locked.

## The most important files
- `kaful/phases/p2_understand/a_select_context.py`, `b_merge_samples.py`, `phase.py`,
  `helpers/canonical_names.py`, `tools/onboarding_review.py`.
- `kaful/artifacts/understand_report.py` (schema 2), `kaful/units.py`, `kaful/config.py`
  (`part_max_tokens`, `batch_parts`, `merge_variant_words`).
- `kaful/prompts/understand/fault_modes/prompt.toml` (v5), `kaful.toml` (reuse rule).

## How to see it working
- `kaful understand review cnc_spindle --run 20261109T184935Z-f8692c`: 18 parts; see the "parts" column.
- `kaful understand score cnc_spindle --run 20261109T184935Z-f8692c`.
- CDV run `20261109T200742Z-33de55` failed its unit gate, so its score is in
  `evals/results/understand_holdout/cdv/`, made outside the CLI.

## Checks
- `pytest -q`: 629 passed offline. mypy is clean on `kaful/`.
- understand.fault_modes v5: both eval runs passed every threshold, and the lock records them.

## Results
All scores below use the same method as before. Channel and component scores use the confident aliases;
CNC fault modes are raw.

**eversys:** an offline replay of the cached calls (run 8fb3a8, $0) produced identical artifacts and scores.
The whole manual is still one part.

**air_compressor** (run 85f3c8, $0.021):

| Measure | Before | After |
|---|---|---|
| Channels R / P, roles | 0.81 / 0.87, 0.92 | unchanged |
| Components R / P | 0.77 / 0.82 | 0.73 / 0.85 |
| Fault modes R / P | 0.54 / 0.54 | 0.54 / 0.64 |

**CNC.** The 1.2M-token Fanuc manual is read in 18 parts.

| Measure | M7b d55739 | M7c iteration 1 (e97f82) | M7c iteration 2 (f8692c) |
|---|---|---|---|
| Channels R / P | 0.73 / 0.29 | 0.73 / **0.11** (146 channels) | same (cached) |
| Channel roles | 0.69 | 0.69 | 0.69 |
| Components R / P | 0.60 / 0.38 | 0.60 / **0.07** (3/45) | 0.60 / **0.09** (3/33) |
| Fault modes R / P | 0.67 / 0.55 | 0.67 / 0.50 | **0.56 / 0.31** (0.67 / 0.38 with the flaking alias) |

The "flaking alias" counts angular-bearing "flaking" as the golden "ball surface"; iteration 1 matched it
without help.

These are precision regressions. They come from over-extraction when the whole manual is read:
- the interface-signal and parameter tables come out as channels;
- every motor series and amplifier type comes out as its own component;
- fresh fault-mode samples list more NSK bearing damage types (12 modes in iteration 1, 17 in iteration 2).

No check was weakened. Recall held, and the three position-coder fault modes are still missed.

**CDV** (run `20261109T200742Z-33de55`). This run is **seen once, not a clean hold-out**, and it
**failed the unit gate**. Machine documents are read in 3 parts. Raw scores, with no aliases on either run:

| Measure | M7b 596dd5 | M7c 33de55 |
|---|---|---|
| Fault modes R / P | 0.07 / 0.10 | **0.57 / 0.33** |
| Components R / P (parents) | 0.45 / 0.50 (0.80) | 0.61 / 0.42 (0.70) |
| Channels R / P (roles) | 0.06 / 0.07 (0.00) | 0.35 / 0.09 (0.83) |

- **Fault modes.** Eight of the 14 golden modes are now found. The run found the tube, paddle, coldtrap,
  cassette, O-ring, heat-exchanger, dust-trap and pump deposit or wear modes from the pages retrieval
  never reached. Two misses are probably name mismatches: inline_filter_clogging (golden inline_gas_filters)
  and process_tubes_deposit_buildup (golden process_tube). They are not adjudicated.
- **Unit gate.** It failed on `SLM` and `SCCM`, because the M7c standard-flow rule matches lower case only.
  It also failed on `mTORR` / `MTOR` and on the duration format `hh:mm:ss`.
- **Datasheet-only carriers.** Pump oil, the motor fan cover and the MFC valve were proposed from datasheets
  only, so they were dropped and listed in the onboarding review, as designed.

## API spend for M7c: about $1.01

| Item | Cost |
|---|---|
| CNC iteration 1 (165 calls, Batch) | $0.644 |
| CNC iteration 2 (fault modes only) | $0.200 |
| CDV run bf8f5e (stopped at max_tokens) and run 33de55 | $0.112 + $0.005 |
| CDV cache replay after the merge fixes (unplanned: the fault-mode inputs changed) | $0.025 |
| air_compressor | $0.021 |
| fault_modes v5 evals (reused v4 replies) | $0.000 |

The per-machine cost targets for a full understand run are in PLAN.md:
- eversys ≤ $0.08;
- air_compressor ≤ $0.07;
- CNC ≤ $0.70;
- CDV ≤ $0.12. That is $0.118 for one full run's calls, which were split across bf8f5e and 33de55.

## Decisions you made (2026-11-09)
- UnderstandReport gets one context per part, with a schema bump; old reports still load.
- Agreement is the best part's share, and `parts_found` is recorded and shown in the onboarding review.
- The CNC cap is $1.50, and part_max_tokens is 80k.
- sccm and slm are volume per time, with the standard conditions kept in the printed string.
- The M1 understand target is now per machine. Understand runs once per machine, at onboarding.
- Iteration 2 is the series/type fold.
- fault_modes v5 raises max_tokens to 32000.
- CDV is scored once, as "seen once, not a clean hold-out".
- M7c closes regardless of the CDV score.

## Open issues
- **CNC channel precision is the main gap** (PLAN.md). It is partly a labelling question: Fanuc interface
  signals can be read from the PMC.
- **Product-family manuals.** machine.toml would declare the configuration, and understand would keep only
  the matching variants. This is the general fix; the series/type fold covers only names that say so.
- **CNC fault-mode precision varies run to run.** Whether installation-caused or sudden bearing damage
  types count as gradual fault modes is a labelling question.
- **Units still unread:**
  - upper-case standard flows (`SLM`, `SCCM`);
  - `mTORR` / `MTOR`;
  - duration formats (`hh:mm:ss`).
- **The disagreement tables are not pre-filled** (Natalie: skip for now): CNC e97f82 (214 rows),
  air_compressor 85f3c8 (40 rows), CDV 33de55 (148 rows).
- **A clean new-machine estimate needs a fifth machine.**
