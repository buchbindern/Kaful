# M6b. Rename phases (layout only, plus approved ingest cleanups)

## The problem it solved
The code didn't show the order things run in. Phase folders were plain names (`ingest/`),
and a phase's files mixed real pipeline steps with helper code and offline tooling. So
reading `phase.py` didn't show which step ran when, and nothing stopped the layout from
drifting.

## What was built
- **Folders:** phase folders carry their order (`kaful/phases/p1_ingest` … `p9_evaluate`).
- **Steps:** inside a phase, only true pipeline steps are lettered, a→g for ingest, and
  `phase.py` calls them directly, in order, after a map that names them.
- **Helpers and tools:** code the steps use lives in `helpers/`, each file naming the steps
  that use it; offline tooling lives in `tools/`.
- **Shared code:** index loading and vector search moved to `kaful/retrieval/`, so M7 uses
  them without importing ingest.
- **Unchanged:** prompt ids, artifact kinds, CLI phase names and run directories.
- **Approved cleanups (separate commits):**
  - failures stop ingest before any embedding is paid for;
  - a missing gateway or embedder is a clean failed status;
  - reused responses are recorded as a structured `reused_from` on the page layer;
  - review files show printed labels next to physical pages and list every page where a
    quote occurs.

## The most important files
- `kaful/phases/p1_ingest/phase.py`: the step map, then steps a→g called in order.
- `kaful/phases/p1_ingest/b_read_pages.py`, `c_classify_pages.py`, `d_add_ruled_tables.py`:
  reading, classifying and free tables, now separate steps.
- `kaful/phases/p1_ingest/g_embed_and_index.py`: embed, fingerprint, save.
- `kaful/retrieval/index.py` and `kaful/retrieval/search.py`: loading a saved index with
  its hash check, and vector search.
- `tests/unit/test_kaful_phase_layout.py`: enforces the naming rule in CLAUDE.md.

## How to see it working
- `pytest -q tests/unit/test_kaful_phase_layout.py`.
- Then read the docstring at the top of `kaful/phases/p1_ingest/phase.py`.

## Checks
- **The rename and restructure commits:**
  - `pytest -q` 408 passed and `pytest -m evals` 6 passed, as on `main`;
  - mypy clean;
  - the ingest output and every LLM request hash are byte-identical to `main`.
- **The cleanups change the saved index format by design:** layers gain `reused_from`.
  Request hashes are unchanged, and the saved eversys and CNC indexes still load.
- **After all commits:** 416 passed, evals 6 passed. $0 spent.

## Decisions you made
- Only true steps get letters; helpers and tools are separate, and every phase follows the
  same rule.
- Search moves to a shared `kaful/retrieval/`.
- M7 uses `a_select_context`, `b_merge_samples`, `c_verify_quotes` and `d_build_artifacts`.
- Printed labels are shown next to physical pages in every review file and report.

## Open issues
- None from the rename.
- `m7-understand` is rebased onto this branch and follows the same layout; its
  determinism output matches the pre-rebase M7.
