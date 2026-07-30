"""
state_data.py
-------------
Checkpoint state for the data-first pipeline.

Mirrors the doc pipeline's checkpoint pattern: every step saves its output
to disk. If the output file already exists, the step is skipped.

This module provides the helpers that phase __init__ files use to check
and record completion. Individual steps handle their own file I/O directly
(save to disk, skip if exists) — this module is for run-level bookkeeping.

Usage:
    from state_data import PipelineState

    state = PipelineState(cfg)
    if state.is_done("phase1"):
        print("phase1 already complete")
    state.mark_done("phase1")
"""

import json
from pathlib import Path


class PipelineState:
    """
    Tracks which phases have been completed for a given dataset + record run.

    Persisted as a JSON file at:
        datasets/<dataset_id>/outputs/<record_id>/pipeline_state.json

    Each phase is responsible for marking itself done at the end of its run()
    function. Individual steps within a phase use file-existence checks directly
    (same pattern as the doc pipeline).
    """

    def __init__(self, cfg: dict):
        self._path = cfg["record_dir"] / "pipeline_state.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_done(self, phase: str) -> bool:
        return self._state.get(phase, {}).get("done", False)

    def mark_done(self, phase: str) -> None:
        import datetime
        self._state.setdefault(phase, {})["done"] = True
        self._state[phase]["completed_at"] = datetime.datetime.now().isoformat()
        self._save()

    def mark_undone(self, phase: str) -> None:
        """Force a phase to re-run (called by --force logic in run_data.py)."""
        if phase in self._state:
            self._state[phase]["done"] = False
            self._save()

    def summary(self) -> str:
        lines = []
        for phase, info in sorted(self._state.items()):
            status = "✓" if info.get("done") else "·"
            ts     = info.get("completed_at", "")
            lines.append(f"  {status} {phase}  {ts}")
        return "\n".join(lines) if lines else "  (no phases run yet)"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._state, f, indent=2)