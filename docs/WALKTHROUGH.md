# Kaful walkthrough: M0–M6

This is a guided tour of what has been built, written for someone who has not read the code.
- **Part 1** goes milestone by milestone.
- **Part 2** follows one real run, file by file.
- **Part 3** is a glossary.
- **Part 4** lists what is not built yet.

Paths use the layout from the rename (`kaful/phases/p1_ingest/…`).

---

## Part 1. Milestones

### M0. Freeze the baseline
- **Problem.** Before changing anything we needed a fixed point to compare against and to
  roll back to.
- **Built.**
  - The tag `pre-revamp`.
  - Pinned dependencies.
  - pytest set-up with markers (`slow`, `evals`, `live`).
  - A tarball of the legacy outputs, kept outside the repo.
- **Files.**
  - `requirements.txt`: exact versions of everything the code imports.
  - `pyproject.toml`: the package definition and pytest settings (plain `pytest` never makes
    live calls).
  - `.env.example`: which API keys `.env` must hold, without their values.
- **See it.** `git show pre-revamp --stat`, then `pytest -q`.
- **Your decisions.**
  - The uncommitted eversys changes were committed before tagging.
  - The repo moved out of iCloud to `~/code/Kaful`.
- **Open issues.** None.

### M1. Measure the legacy pipeline's cost
- **Problem.** Nobody knew what a run cost, or where the money went.
- **Built.**
  - A minimal ledger wrapped around every legacy LLM, PDF-extraction and embedding call.
  - A pricing table.
  - A cost table per phase and per step.
- **Files.**
  - `utils/ledger.py`: the legacy ledger (one JSON line per call).
  - `pricing.toml`: prices per model; costs are computed from it, never typed in.
  - `docs/m1_cost_baseline.md`: the baseline. eversys costs $5.96 from scratch; CNC $22.06.
    The three biggest costs are phase 1 re-extraction, phase 6b code generation and CNC
    step 2c.
- **See it.** Open `docs/m1_cost_baseline.md`.
- **Your decisions.** Estimate from existing data instead of paying for fresh legacy runs
  ($0 spent).
- **Open issues.** The CNC baseline is partly estimated ($14.46–17.61 for phase 1).

### M2. Offline harness and golden run
- **Problem.** Tests must never call an API. We also needed a way to notice when a change
  alters results.
- **Built.**
  - A network block for tests.
  - Fake LLM and embedding clients that replay recorded answers by prompt hash.
  - The tiny fixture machine `tiny_pump`.
  - A golden snapshot of eversys's late-phase outputs.
- **Files.**
  - `tests/harness/netblock.py`: makes any network access in a test fail.
  - `tests/harness/fake_anthropic.py` and `tests/harness/fake_embedder.py`: replay recorded
    responses.
  - `tests/fixtures/machines/tiny_pump/`: a 3-page synthetic manual with 7 sensor channels.
  - `tests/integration/test_golden_eversys.py`: the `slow` regression against the
    snapshot (about 9 minutes).
- **See it.** `pytest -q tests/integration/test_tiny_pump_replay.py` (about 1 s), or
  `pytest -m slow`.
- **Your decisions.** Close M2 early and freeze the legacy code rather than fix it.
- **Open issues.**
  - The full end-to-end tiny-machine run was not met.
  - Legacy phase 3 can't replay because of the hash-order bug (3a). It is an eval case in M7.

### M3. Remove dead code
- **Problem.** `data_pipeline/` belonged to another project.
- **Built.** It was deleted, after `reconcile.py` was saved outside the repo.
- **Files.** `kaful/reference/`: where M4 brought `reconcile.py` back as a reference.
- **See it.** `ls data_pipeline` (it is gone).
- **Your decisions.** The rest of the legacy code stays frozen until M15.
- **Open issues.** None.

### M4. Package skeleton and typed artifacts
- **Problem.** Legacy phases passed loose JSON to each other, and runs overwrote each other.
- **Built.**
  - The `kaful/` package and the `kaful run` command.
  - Typed models for every artifact.
  - A run manifest, with one folder per run.
  - A sandbox for generated code.
  - The interfaces for later milestones: `TwinBackend`, `DataSource`, templates.
- **Files.**
  - `kaful/artifacts/`: one Pydantic model per artifact (DocumentIndex, MachineModel,
    ParamSet, and so on), each with a schema version.
  - `kaful/pipeline.py`: starts a run, runs the selected phases in order, writes the
    manifest.
  - `kaful/runs.py`: run ids, run folders, saving and loading artifacts.
  - `kaful/sandbox/runner.py`: runs generated code in a separate process with a time limit.
  - `kaful/backends/base.py`: the `TwinBackend` interface. ProgPy will be its first adapter.
- **See it.** `pytest -q tests/unit/test_kaful_artifacts.py`, then open any
  `outputs/runs/<run_id>/manifest.json`.
- **Your decisions.** Health is tracked as physical damage D, with health = 1 − D/D_limit.
- **Open issues.** `hypothesis` (property-based tests) is not installed yet; templates need
  it in M10.

### M5. LLM gateway, prompt registry, evals
- **Problem.** LLM calls were scattered across the legacy code, with no cost control, no
  caching and no way to tell whether a prompt was good.
- **Built.**
  - **One gateway** that every model call goes through. It handles the ledger, the response
    cache, the budget stop, batches, and one retry when output fails validation.
  - **A prompt registry,** with each prompt versioned and locked.
  - **An eval framework,** with checks, golden cases and an optional LLM judge.
- **Files.**
  - `kaful/llm/gateway.py`: the only door to a model.
  - `kaful/llm/ledger.py` and `kaful/llm/cache.py`: the record of every call, and the stored
    answers.
  - `kaful/prompts/registry.py` and `kaful/prompts/versions.lock.json`: prompt files, and
    which versions are approved.
  - `kaful/evals/runner.py`: runs a prompt's eval set and scores it against thresholds.
- **See it.**
  - `kaful eval run sample.list_sensor_channels` replays offline at $0.
  - `kaful prompts check`.
- **Your decisions.**
  - Model choice is per prompt, in config.
  - The cheapest model that passes wins, so the sample prompt and the judge run on Haiku.
- **Open issues.** None (unit parsing came in M7).

### M6. Ingest
- **Problem.** Legacy sent every page to an LLM ($2.18 for eversys, about $15.85 for CNC).
  Its quotes came from that transcription, not from the manual.
- **Built.** PDFs become a `DocumentIndex`:
  - raw text and printed labels for every page;
  - a class for every page;
  - free tables where PyMuPDF can read them;
  - Haiku on hard pages, with Sonnet re-reading pages that have tables or failed;
  - deterministic chunks;
  - embeddings that never cut text.
- **Cost.** eversys **$0.21**, CNC **$3.71**.
- **Files.**
  - `kaful/phases/p1_ingest/phase.py`: runs the steps a→g, and opens with a map of them.
  - `c_classify_pages.py`: decides which pages need the LLM.
  - `e_extract_hard_pages.py`: Haiku first; escalation to Sonnet.
  - `f_chunk_pages.py`: chunks and the coverage check.
  - `kaful/llm/embeddings.py` and `kaful/llm/tokens.py`: exact token counts and windows.
- **See it.** `kaful run eversys_coffee_machine --only ingest --budget 0.65` costs $0 now,
  because every answer is cached.
- **Your decisions.** Recorded in `docs/milestones/M6.md`, among them:
  - the gate;
  - the routing;
  - chunker v2;
  - the matcher rules;
  - two-run locks;
  - reuse of v2 answers.
- **Open issues.**
  - A batch-poll network error crashes the run.
  - The cheaper escalation is parked.

---

## Part 2. One run, file by file: `kaful run eversys_coffee_machine --only ingest`

1. **The command.** `kaful/cli.py` parses the arguments and loads settings. Settings come
   from `kaful.toml`, `KAFUL_*` environment variables and `.env` for API keys. It creates
   the real API clients (`kaful/llm/anthropic_api.py` and `openai_api.py`) and calls
   `kaful/pipeline.py`.
2. **The run folder and manifest.** `pipeline.start_run` reads
   `machines/eversys_coffee_machine/machine.toml`. It picks a new run id (time plus random
   suffix), creates `machines/eversys_coffee_machine/outputs/runs/<run_id>/` and writes
   `manifest.json`. The manifest holds the inputs and their hashes, the code version, a
   snapshot of the config, the budget, and a pending status for each of the 9 phases. It
   also builds the run's **gateway** and **embedder**. Both write to the same `ledger.jsonl`
   and share one **budget**.
3. **Phase dispatch.** `pipeline.execute` marks the phases before ingest as skipped, then
   calls `IngestPhase.run` (`kaful/phases/p1_ingest/phase.py`, found through
   `kaful/phases/registry.py`). Logs go to `run.log.jsonl`.
4. **a_check_documents.** The document list in machine.toml is compared with the PDFs on
   disk. Each file gets a doc id and a sha256 (`DocumentRecord`).
5. **b_read_pages.** PyMuPDF reads every page's raw text, which is never changed, and the
   printed page label (e.g. physical p590 has label 566). It also measures *signals*:
   characters, image area, drawings, ruled tables, short and numeric lines.
   `helpers/page_signals.py` does the measuring.
6. **c_classify_pages.** Rules turn the signals into a class: blank, scanned, garbled,
   unruled table, figure, ruled table or text. Scanned, garbled, unruled-table and figure
   pages need the LLM; Fanuc and NSK ruled tables do too (`ruled_tables = "llm"`).
7. **d_add_ruled_tables.** Pages with a good ruled table get a free markdown copy of it.
8. **e_extract_hard_pages.** Every page needing the LLM is drawn as a PNG
   (`helpers/render_pages.py`) and sent with its raw text to `ingest.extract_page`. The
   prompt must be locked in `versions.lock.json`. For each page the **gateway**
   (`kaful/llm/gateway.py`):
   - builds the request and its **prompt hash**;
   - looks in the response cache, `outputs/llm_cache/`, where an answer costs $0;
   - otherwise reserves the worst-case cost against the budget and sends the pages as a
     **batch**. Batch state is saved in `outputs/llm_batches/`, so a crash can resume
     without paying twice;
   - waits, charges the real cost, stores the answers in the cache, and checks each against
     the output model, retrying once if it is invalid;
   - writes a ledger row: model, tokens, cost, route, and cache hit or miss.

   Haiku goes first. Any page where Haiku found a table, or failed, is sent again to Sonnet
   (the *escalation*), and Sonnet's answer becomes the page's `llm` layer.
9. **f_chunk_pages.** Pages are packed into chunks inside a section, never splitting a page.
   Each chunk id is a hash of the file, the page range and the chunker version. The
   coverage check confirms every page with text is in exactly one chunk; if not, the phase
   stops here, before paying for embeddings.
10. **g_embed_and_index.**
    - **Embedding:** each chunk is embedded by OpenAI. A chunk over 8,191 tokens is
      embedded as overlapping windows. Vectors are cached in `outputs/embedding_cache/`.
    - **Fingerprint:** the classifier and chunker versions, the routing rule and the
      embedding model.
    - **Index hash:** covers the fingerprint and every chunk's text.
    - **Saved:** `ingest/embeddings.npy` and `ingest/document_index.json` in the run folder.
11. **Closing.** The phase returns its record (status, artifacts, failures, notes). The
    pipeline adds its cost from the ledger and writes the final `manifest.json`. The CLI
    prints `passed: <run folder>`.

**Where each output lands.** Run folders are gitignored; the caches are per machine and
shared by every run.

| Output | Location |
|---|---|
| This run's record | `outputs/runs/<run_id>/manifest.json`, `ledger.jsonl`, `run.log.jsonl` |
| This run's index | `outputs/runs/<run_id>/ingest/document_index.json` and `embeddings.npy` |
| Shared caches | `outputs/llm_cache/`, `outputs/llm_batches/`, `outputs/embedding_cache/` |
| Eval results (committed) | `evals/results/`; eval ledgers in `outputs/evals/` |

---

## Part 3. Glossary

- **Artifact.** A typed file one phase hands to the next (e.g. `DocumentIndex`). It carries
  the run id that produced it and a schema version.
- **Batch.** Anthropic's Message Batches API: many requests at half price, answered within
  hours. Batches are sent in slices, each sized to fit the remaining budget.
- **Budget.** The cap for one run (`--budget`). Before each call the gateway reserves its
  worst-case cost and refuses it if that won't fit, so a run never overspends.
- **Cache (response cache).** Stored answers, keyed by prompt hash. A repeated request costs
  nothing.
- **Chunk.** A group of whole pages used for search; its id is deterministic.
- **Escalation / route.** Re-running a page on a stronger model when the first pass meets a
  rule. The *route* written on each ledger row says which pass it was.
- **Eval / eval set.** The checks and golden cases a prompt is scored on. Thresholds decide
  pass or fail.
- **Evidence class.** How a quote was found on its cited page:
  - *verbatim*: word for word in the raw text;
  - *near*: almost word for word;
  - *in_layer*: only in an LLM transcription;
  - *unsupported*: not found.
- **Fingerprint.** Everything besides chunk text that changes what an index returns.
- **Fixture.** A small, committed test input (e.g. `tiny_pump`) or a recorded response.
- **Gate.** A check that stops the run when it fails, unless `--allow-failures` is given.
- **Golden (case, snapshot).** Hand-checked expected output that a run is compared against.
- **Grounding.** How many legacy evidence quotes can be found in an index; reported for
  context only.
- **Index hash.** A hash of the fingerprint and every chunk's text. Later phases record it,
  so a changed index is noticed.
- **Ledger.** One JSON line per model or embedding call: cost, tokens, route, cache hit.
- **Lock.** Approval of one prompt version, in `versions.lock.json`. The pipeline refuses
  unlocked prompts. A lock needs two passing eval runs and records both runs' scores.
- **Manifest.** The record of one run: inputs, code version, config, overrides, cost, and
  pass/fail per phase.
- **Matcher.** The code that finds a quote in page text. It normalises case and dashes;
  `2,0` equals `2.0`; a hyphen at a line break joins letters only.
- **Page class / signals.** See steps b and c in Part 2.
- **Physical page / printed label.** p590 is the 590th page of the PDF; "566" is the number
  printed on it.
- **Prompt hash.** A hash of the full request: model, prompt text, inputs, images, output
  format. It is the cache key.
- **Provenance.** Where a number came from (manual, standard, literature, …) and how
  confident we are, computed by code.
- **PYTHONHASHSEED.** Python's per-run random seed for sets and dicts. Tests run under
  several seeds to prove no output depends on set or dict order.
- **Step / helper / tool.**
  - *Step:* a lettered file `phase.py` runs in order.
  - *Helper:* code steps use.
  - *Tool:* code for people, not runs.

---

## Part 4. Not built yet

| Milestone | What remains |
|---|---|
| **M7 Understand** | Built offline on branch `m7-understand`: tiny_pump passes on Haiku and Sonnet, nothing locked. Remaining: your golden labels for eversys and CNC, value roles (spec, variant, default, setting range, example), live runs, locks, report. |
| M8 Scope | Decide which components get full models, before any expensive per-component work. |
| M9 Parameterize | Every rate as a prior with provenance, calibrated from documented life; value roles decide what may set a prior. |
| M10 Build | Degradation templates (pure numpy, property-tested), the ProgPy adapter, `TwinDefinition`. |
| M11 Verify | Static and dynamic checks that stop the run. |
| M12 Synthesize | Optional synthetic data when no real data exists. |
| M13 Estimate | State estimation and remaining-life forecasts from a `DataSource`; what-if profiles. |
| M14 Evaluate | Grounding, twin quality, forecast calibration, sensitivity. |
| M15 Delivery | CI, a demo, deleting the legacy code. |
