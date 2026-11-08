# M7 exit criteria (proposal, 2026-11-08)

M7 closes when each scored machine (eversys_coffee_machine, cnc_spindle) meets every floor below on
one understand run. The run uses the locked prompts on Haiku with 3 samples and the configured
context policy. It is scored against the reviewed golden file, and its pre-filled disagreement list
has been adjudicated. Gaps that remain are logged in PLAN.md under the milestone that will use them;
they don't hold M7 open.

## Floors (all on one run, per machine)

| Measure | Floor | Why this one |
|---|---|---|
| Channel recall | 0.80 | M8 and M13 need the reported signals; a missed channel is an observation lost |
| Channel precision | 0.70 | Extras are cheap to drop by hand, so this floor can be lower than recall |
| Channel role accuracy | 0.90 | Roles decide what M13 estimates (observation) and what it conditions on (operating_input, usage_clock) |
| One usage clock, the golden one | yes | It is the time axis of every rate |
| Fault-mode recall (documented `yes` rows) | 0.70 | These are the degradation mechanisms M9 parameterizes |
| Fault-mode precision | 0.70 | An effect or a sudden fault taken as a mechanism becomes a wrong model |
| Fault-mode carriers found | 1.00 | Every component a documented `yes` fault mode sits on is extracted or declared; without it the mode cannot exist |
| Component recall | 0.60 | Below the carriers, other components matter less for M9 |
| Component precision | 0.60 | Same |
| Golden examples stored as values | 0 | Hard rule: examples never set priors |
| Components or channels quoted only from reference documents | 0 | The document-roles rule (2026-11-08) |
| First attempts valid | at least 8 of 9 calls | Retries cost money and hide prompt problems |
| Run cost (Haiku) | at most $0.10 | Measured runs: eversys $0.071, CNC $0.030 |
| Phase status | passed | A failed phase is not a result |

## Where the machines stand

| Measure | eversys 20261108T192349Z-21c0ec | CNC 20261108T192837Z-35431a (before the document-roles fix) |
|---|---|---|
| Channel recall / precision / roles | 1.00 / 0.83 / 1.00: meets | 0.05 / 0.33 / 1.00: fails |
| Fault-mode recall / precision | 0.71 / 0.83: meets | 0 / 0: fails |
| Fault-mode carriers | fails: no milk_system | fails: no bearings, no position_coder |
| Component recall / precision | 0.62 / 0.65: meets | 0 / 0: fails |
| First attempts valid / cost | 9 of 9 / $0.071: meets | 8 of 9 / $0.030: meets |

- eversys misses one floor, the carriers: milk_system was never extracted, so the milk residue
  mode had no component.
- CNC was rerun as 20261108T201412Z-378c70 after the document-roles fix. It failed on
  sensor_channels max_tokens (see M7-understand-next notes) and has no score yet.

## Not M7 (log for later)

- Sonnet locks and routing (cost and quality comparison): M15.
- Golden rows marked `documented = false` (eversys grinder burrs, CNC motor overheating): M9
  handbook priors.
- error_history / error_message joined only by page, and any merge misses left after the
  canonical step: logged with counts.
- Per-component retrieval: skipped (2026-11-08).
