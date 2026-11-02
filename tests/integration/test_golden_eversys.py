"""
Golden run (M2, `pytest -m slow`): eversys phases 6–7 re-derived offline from
frozen LLM checkpoints must match the reviewed snapshot within tolerances.toml.

A failure is a behaviour change to report, not a reason to refresh the snapshot
(scripts/update_golden.py is a separately approved step).
"""

from pathlib import Path

import pytest

from tests.harness.golden import (
    EXPECTED_DIR, GOLDEN_SEED, build_scratch, compare, load_expected, load_tolerances, run_golden)

MAX_REPORTED = 50


@pytest.mark.slow
def test_eversys_phase6_7_matches_golden(tmp_path: Path) -> None:
    if not any(EXPECTED_DIR.rglob("*.json.gz")):
        pytest.fail(f"no golden snapshot in {EXPECTED_DIR}; run scripts/update_golden.py (approved step)")
    machines_dir = tmp_path / "machines"
    machines_dir.mkdir()
    build_scratch(machines_dir)

    diffs = compare(load_expected(), run_golden(machines_dir, GOLDEN_SEED), load_tolerances())

    assert not diffs, f"{len(diffs)} differences from golden:\n" + "\n".join(diffs[:MAX_REPORTED])
