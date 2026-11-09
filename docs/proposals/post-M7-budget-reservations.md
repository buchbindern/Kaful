# After M7: budget reservations that don't crawl (proposal, 2026-11-09)

For Natalie's go after M7. Nothing here is built.

## The problem

Before a call is sent, the gateway reserves its worst case: the whole prompt written to cache, plus
the full `max_tokens` generated. A Message Batch is cut into chunks whose summed worst cases fit
the remaining budget. That guarantees the run never overspends. With a small budget, though, it
sends a few requests per batch, and every round waits on the Batch API.

On air_compressor (2026-11-08), 14 Sonnet re-reads went out one or two per batch, each round
taking minutes to hours. The first 80-page batch took about 1.5 hours.

Measured over every fresh call in all ledgers so far:

| Prompt, model | Calls | max_tokens | Output median | p99 | max | Reserved / actual (median) |
|---|---|---|---|---|---|---|
| ingest.extract_page, Haiku | 901 | 8000 | 512 | 6826 | 9518 | 8.4x |
| ingest.extract_page, Sonnet | 436 | 8000 | 711 | 4298 | 4903 | 7.1x |
| understand.components, Haiku | 19 | 16000 | 7358 | 10116 | 10116 | 3.4x |
| understand.fault_modes, Haiku | 15 | 16000 | 3960 | 5453 | 5453 | 5.3x |
| understand.sensor_channels, Haiku | 26 | 6000 to 32000 | 5964 | 17241 | 17241 | 3.7x |

(max_tokens is the first row's value; the prompts' versions changed it over time.)

The tail is real: Haiku extract_page reached 9518 tokens once, and CNC sensor_channels 17241. So
calibrated estimates alone cannot keep a never-overspend guarantee. Any call can run to its
`max_tokens`.

## Proposal: plan on calibrated estimates, guarantee on a hard cap

1. **Two limits per run.**
   - `--budget B` is the spend we expect. New work is scheduled only while spent plus the
     calibrated estimates of the calls in flight fit within B.
   - `--hard-cap H` (default 2 x B, recorded in the manifest) is the guarantee. Spent plus the worst
     cases in flight must fit within H.
   - Batches are cut on H, so a normal run goes in one or two rounds.
   - When either limit would be exceeded, the run stops cleanly, as today. **The run never spends
     more than H.**
2. **Calibrated estimates are data, not code.** `kaful llm calibrate` writes `llm/calibration.json`
   from the ledgers. Per prompt id, version and model, it records the input-cost ratio and the
   p99 output tokens, with at least 30 fresh calls; below that, `max_tokens` is used, as today.
   A run reads the file once (deterministic) and records its hash in the manifest. Reservations
   and actual costs stay in the ledger, so the calibration can be checked.
3. **Lower `max_tokens` where the tail allows**, through prompt version bumps with their evals.
   Example: understand.fault_modes, p99 5453 against 16000. A call that hits the lower limit
   already retries or escalates (`first_pass_failed`), and that retry is reserved at its own worst
   case. This shrinks the worst case H has to cover. extract_page keeps 8000: its tail reaches
   9518, so no.

Tests: offline, with a scripted transport.
- A batch is cut on H, not B.
- A run where every call reaches `max_tokens` stops at H.
- A run without calibration behaves as today.
- A calibration file below 30 calls falls back to `max_tokens`.

Cost to build: none (no API calls). About one session.
