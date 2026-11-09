# M7: hold-out machine, onboarding review, sensor_channels v6 (proposal, 2026-11-08)

For Natalie's go. Nothing here is run yet.

## 1. Hold-out: air_compressor

The rule: ingest it, run understand with the locked prompts, and only then build its golden labels.
Nothing is tuned on it, and its score is our estimate for a new machine.

On disk in `machines/air_compressor/`:

| Document | Pages | Pages with images | Proposed kind |
|---|---|---|---|
| `manual/atlas_copco_air_compressor.pdf` | 44 | 17 | manual |
| `manual/compressed_air_manual.pdf` ("Compressed air manual, 9th edition") | 148 | 112 | handbook: a general reference, not this machine's documentation |

- There is no `machine.toml`. I would write one with the two kinds above. Deciding kinds is
  onboarding work, so you confirm it.
- The directory has legacy outputs (`schema/`, `codegen/`, `twin/`). I have not opened them. I won't
  open them before the run is scored, so the golden labels aren't anchored on legacy output.

**Cost (Haiku, locked prompts):**

| Step | Basis | Expected | Cap |
|---|---|---|---|
| Ingest: page reading, LLM layer for table and figure pages, embeddings | eversys (161 pages) $0.21, CNC $0.24 | $0.20 to $0.30 | $0.35 |
| Understand: whole Atlas Copco manual (about 30k tokens) for components and channels; fault modes with the handbook by retrieval | eversys $0.071 at 91k tokens; CNC about $0.03 to $0.05 at 40k | $0.03 to $0.06 | $0.08 |
| **Total** | | **$0.25 to $0.35** | **$0.43** |

Building the golden labels afterwards is offline: I pre-fill and you review the unsure rows.

**Prompts it would run with:** components v5, sensor_channels v5 and fault_modes v3. v6 is not
locked (section 3). If v6 locks first, the hold-out uses v6.

## 2. Onboarding review

One page per understand run, for an engineer, about 15 minutes per machine.
- **What it shows:** the component tree, the channels and the fault modes. Each item links to its
  cited page (doc and page, with the printed label).
- **Order:** failed checks and dropped items first, then downgraded items, then items by rising
  confidence.
- **Actions per item:** confirm, correct (name, parent, role, unit) or add a missing item.
- **What it writes:** a review record (`outputs/understand_reviews/<run_id>.toml`: item, action,
  reviewer, date). Corrections and additions go to machine.toml `[[components]]` /
  `[[instrumentation]]` as source `deployment`.
- **Confidence:** a confirmation never sets one. Hard rule 15: confidence is computed by code. An
  "engineer confirmed" flag can become one of its measured signals later.

**Proposal: build the minimum as M7's exit item.** That is a generated, read-only HTML page plus the
review TOML, which a `kaful understand review apply` command turns into machine.toml declarations.
An editable page (confirm buttons, saved in place) waits until a real engineer uses it. It would be
a later milestone, and the review data it writes would not change.

Cost: none, no LLM calls. About one session to build, with tests.

## 3. sensor_channels v6 (not locked; parked on branch `m7-sensor-channels-v6`)

v6 added two general rules: report each quantity once even when several pages list it, and report
per-instance channels only for the instances machine.toml `[instances]` states (default 1). It also
added a `cnc_instances` eval case (p200 (label 176), p730 (label 706)).

It failed its Haiku evals on both runs ($0.0225 in all):
- **tp300 `variant_found`: 0 on both runs.** The model dropped the range given per model ("0 to 16
  for the TP-300-15 and 0 to 25 for the TP-300-22"), which v5 kept on both of its runs. Adding "This
  is about parts, not about models... values given per model are variants" did not fix it. My best
  guess is that the instance rule and its "one of each" input push the model to treat per-model
  values as instances.
- **cnc `quantity_once`: 0.5 on run 2.** The model kept both the diagnosis-screen motor temperature
  and the trace list's motor winding temperature. Calling those the same quantity was my judgement
  when I wrote the case.
- `pulse` failed unit parsing. It is now a counter unit (committed on m7-understand).

**Proposed next step:** keep the "report each quantity once" rule. Move instances out of the prompt:
code drops a channel whose printed name carries an instance ordinal ("second", "2nd", "axis 2")
beyond machine.toml's count, and logs it. That is general, deterministic and testable offline, and
leaves the variant rule alone. Then 2 eval runs (about $0.015) and the last CNC iteration (about
$0.045, with components v5 fresh). Cap $0.06.
