# M7 understand: next prompt versions and paid runs (proposal, 2026-11-08)

This is for Natalie's go. Nothing in it is applied yet: a version bump without its evals turns the
lock test red, so the prompt changes, their eval runs and their locks go in together.

## 1. Why the eversys whole-manual run missed three fault modes

The run is 20261107T211601Z-85fe23. All three were in the context, and all three samples missed
them, so the cause is in the prompts, not in sampling.

| Missed (golden) | Cause | Fix |
|---|---|---|
| water filter exhaustion (W-014, p114 (label 114)) | `water_filter` was never a component. components v3 says "consumables ... are not components", and fault_modes may use only the components it is given. | components v4: a consumable that wears out or is replaced on a schedule is a component. |
| milk residue in the milk system (p84; mechanism on p14 (label 14)) | p14 says "milk and coffee residue may accumulate in the machine, clog the outlets". All 3 samples gave this sentence only to `coffee_outlet`. p84 states a hygiene hazard, not a mechanism. | fault_modes v3: when one passage states a mechanism for several components, give one fault mode per component. |
| grinder burr wear (golden row cites p84) | The manual never states burr wear. p84 gives only the machine service interval (12 months or 50'000 products). The model was right not to report it. | Golden question for Natalie (section 4). |

The run also made three sudden-jam modes (grinder blockage), which Natalie answered no: sudden
jams can't be forecast. fault_modes v3 leaves them out.

## 2. Prompt drafts

### understand.fault_modes v3 (system.md, replacing the last paragraph)

```
Report gradual mechanisms: ones that progress with use or time, which maintenance resets or
replacement ends (wear, buildup of residue, scaling, exhaustion, fatigue, contamination). Leave
out sudden events (a foreign object, a jam, a broken cable, a unit not connected) unless the
documentation says they build up.

Read the whole documentation for mechanisms, including warning and error lists, troubleshooting
tables, maintenance schedules and service intervals. An entry such as "the filter has to be
changed after its life time cycle, otherwise the boilers will be damaged due to the limescale"
states two mechanisms: exhaustion of the filter, and scaling of the boilers. Report each on the
component it happens to; an alarm, a service interval or a troubleshooting entry is evidence for a
fault mode, not a fault mode of its own.

When one passage states a mechanism for several components ("milk and coffee residue may
accumulate"), give one fault mode for each component it names. Report only what the documentation
states.
```

New eval case `eversys_gradual`: the raw text of eversys p14, p54, p84, p114, p115, p118 and p127
(labels equal pages), with the golden component ids including `water_filter` and `milk_system`.
- `gradual_recall`: `fault_modes[].component` contains_all
  [water_filter, milk_system, coffee_boiler, steam_boiler, steam_wand]. Threshold 1.0.
- `no_sudden_faults`: `fault_modes[component=ceramic_grinders].mechanism` not_in
  [blockage, blocking, jam]. Threshold 1.0.

### understand.components v4 (system.md, one line changed)

```
- Software functions, operating modes, alarms, ingredients, cleaning agents and settings are not
  components. A consumable part that wears out or is replaced on a schedule (a water filter, a
  cartridge, a gasket) is a component.
```

New eval case `eversys_consumables`: p114 and p45 raw text. `components[].id` contains_all
[water_filter]. Threshold 1.0.

### understand.sensor_channels v4 (system.md, two lines added under `role`)

```
  - A counter of one kind of product or operation (coffee products, milk products) is an
    operating_input: the mix drives the wear of the components that make it. The one running
    total all usage is measured against is the usage_clock.
- A count with no printed unit has unit "count".
```

The second line stops the retry every eversys sample needed ("total_counter: numeric channels need
a unit"). All 3 first attempts failed, against the prompt's first_attempt_valid threshold of 0.8.

New eval case `eversys_counters`: the p70 and p71 text, with the screenshot layer.
- `counter_roles`: keyed on name, role coffee_counter / milk_counter = operating_input and
  total_counter = usage_clock. Threshold 1.0.
- `counter_units`: unit = count. Threshold 1.0.

## 3. Merge (done, offline)

Near-duplicates are joined on evidence the run produced, never on golden aliases: the golden file
is the answer key.

- Components: a trailing generic head word is ignored, so grinder joins grinder_module and
  milk_module joins milk_system. A caption naming one part inside a longer name is one part
  (bean_hopper_with_lid_position_2).
- Channels: an identical quote that doesn't name both printed names joins (the cleaning trio). So
  does the same reading on two pages, when it occurs at most once in any line of each page
  (flow / ticks_per_second).
- On the cached eversys samples this joins 6 of the 7 duplicates. error_history / error_message
  share only a page, which is too weak to join on.

## 4. Golden questions for Natalie

- **eversys `ceramic_grinders: grinder burrs` (yes, p84).** The manual never states burr wear, so
  understand cannot find it from the documentation. Proposal: keep the row but mark it
  `documented = false`. The scorer then leaves it out of understand recall, and M9 takes it as a
  handbook prior.
- **CNC `spindle_motor: motor_overheating` (yes, p188 (label 164)).** p188 is a parameter list
  ("Motor overheat level", "Short-time overload detection period"). The Fanuc manual names
  overheating only as an alarm (p209 (label 185)) and states no mechanism. Same proposal:
  `documented = false`, or a no.

## 5. CNC retrieval (measured offline, $0)

| Policy | Context | Fault-mode evidence pages | Channel evidence pages |
|---|---|---|---|
| current shared retrieval | 39,761 tokens | 3/4 (NSK p60 (label 59), p61 (label 60), Fanuc p708 (label 684)) | 0/1 (misses Fanuc p200 (label 176)) |
| per component: chunks naming the component, scored by the cached fault-mode queries, top 2 to 6 per document | about 39,400 tokens | 3/4 (same pages) | not measured |

- The old "1 of 6" was measured before the six bearing rows moved to p60-61.
- Per-component retrieval gains nothing here: every Fanuc chunk names "spindle motor", so the name
  filter does not narrow it.
- The only page missed is p188, the weak row in section 4.
- So per-component retrieval is not built into the pipeline. A variant whose query names the
  component ("spindle motor wear, overheating and failure") needs about 8 new query embeddings
  (under $0.001), which is a paid call to approve.

## 6. Cache bug to fix before paying again

The gateway stores responses by sample index, and `_store` silently skips a sample when earlier
samples are missing from that file.
- A retry request carries its sample's own reply, so it has its own hash, and retries of samples
  1 and 2 are never stored.
- When sample 0 is served by the reuse rule, samples 1 and 2 of the new version are never stored.

In run 85fe23 this lost the replies of sensor_channels samples 1 and 2. They were paid for, a
replay would pay again, and they cannot be inspected.

Proposed fix:
- (a) Store and read retry responses at index 0 of their own hash. Each retry request is unique to
  its sample. No format change.
- (b) When a sample is served by reuse, also record it under the new hash, marked
  `source = "reused"`. This adds a value to `StoredRecord.source`, which is a format change.

Failing tests first.

## 7. Paid runs (Haiku), for approval

| Step | What | Expected | Cap |
|---|---|---|---|
| 1 | Evals for fault_modes v3, components v4 and sensor_channels v4: tp200 plus the new eversys cases, 2 runs each on Haiku, then lock | about $0.04 | $0.08 |
| 2 | eversys rerun, whole manual (91k tokens), new prompts, 3 samples | about $0.08: last run $0.05 with components cached, plus fresh components | $0.12 |
| 3 | First CNC run, shared retrieval (about 40k tokens), new prompts | about $0.05 (eversys retrieval on Haiku was $0.041 at 35k) | $0.08 |

Total cap: $0.28. Steps 2 and 3 are each scored, with the disagreement list pre-filled.
