"""`kaful runs prune` (M15): which runs can go and why. A dry run by default; --apply deletes exactly the plan."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.manifest import PhaseRecord, RunManifest
from kaful.cli import main
from kaful.config import KafulSettings, RunsSettings
from kaful.phases import PHASE_ORDER
from kaful import prune
from kaful.prune import apply_plan, prune_plan
from kaful.runs import MANIFEST_FILE, dump_json, runs_dir
from tests.harness import kaful_factories as f

NOW = datetime(2026, 11, 18, 12, tzinfo=UTC)
SHA = "a" * 64


def _run(machine_dir: Path, run_id: str, hours_ago: float, status: str = "passed",
         passed: tuple[str, ...] = (), reads: tuple[str, ...] = (), base: str | None = None) -> None:
    """A run directory with a manifest: `passed` phases passed, `reads` runs whose artifacts it uses."""
    refs = tuple(ArtifactRef(kind="document_index", run_id=r, path=f"outputs/runs/{r}/ingest/document_index.json",
                             sha256=SHA) for r in reads)
    phases = tuple(PhaseRecord(phase=p, status="passed" if p in passed else "reused" if refs and p == "ingest"
                               else "pending", artifacts=refs if p == "ingest" else ()) for p in PHASE_ORDER)
    manifest = f.run_manifest().model_copy(update={
        "run_id": run_id, "machine_id": machine_dir.name, "created_at": NOW - timedelta(hours=hours_ago),
        "status": status, "phases": phases,
        "base_run": None if base is None else {"run_id": base, "selected_by": "explicit", "inputs_match": True}})
    manifest = RunManifest.model_validate(manifest.model_dump())
    run_dir = runs_dir(machine_dir) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / MANIFEST_FILE).write_text(dump_json(manifest))
    (run_dir / "samples.csv").write_text("x\n" * 100)


@pytest.fixture
def machine(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "docs" / "milestones").mkdir(parents=True)
    (root / "docs" / "milestones" / "M9.md").write_text("The final run was `20261110T000000Z-f0f0f0`.\n")
    (root / "PLAN.md").write_text("Base run 0b0b0b (short id).\n")
    machine_dir = root / "machines" / "m"
    _run(machine_dir, "20261101T000000Z-aaaaaa", 400, passed=("ingest", "understand"))
    _run(machine_dir, "20261102T000000Z-bbbbbb", 380, status="failed")
    _run(machine_dir, "20261103T000000Z-0b0b0b", 370, passed=("ingest",))
    _run(machine_dir, "20261110T000000Z-f0f0f0", 200, passed=("ingest",))
    _run(machine_dir, "20261111T000000Z-dddddd", 48, status="running")
    _run(machine_dir, "20261112T000000Z-eeeeee", 30, passed=("ingest",))
    _run(machine_dir, "20261117T000000Z-cccccc", 12, passed=tuple(PHASE_ORDER), reads=("20261101T000000Z-aaaaaa",))
    _run(machine_dir, "20261118T110000Z-111111", 1, status="running")
    (runs_dir(machine_dir) / "20261105T000000Z-999999").mkdir()  # no manifest
    return machine_dir


def _plan(machine: Path, keep: tuple[str, ...] = ()) -> dict[str, tuple[bool, str]]:
    plan = prune_plan(machine, RunsSettings(), NOW, root=machine.parent.parent, keep=keep)
    return {v.run_id[-6:]: (v.keep, v.reason) for v in plan}


def test_each_run_is_kept_or_dropped_for_a_stated_reason(machine: Path) -> None:
    got = _plan(machine)
    assert got["cccccc"] == (True, "latest run that passed build, estimate, evaluate, ingest, parameterize, scope, "
                                   "synthesize, understand, verify")
    assert got["aaaaaa"] == (True, "read by 20261117T000000Z-cccccc")
    assert got["f0f0f0"] == (True, "cited in docs/milestones/M9.md")
    assert got["0b0b0b"] == (True, "cited in PLAN.md")
    assert got["111111"] == (True, "marked running, started 1.0 h ago: may be live")
    assert got["999999"] == (True, "no readable manifest")
    assert got["bbbbbb"] == (False, "failed")
    assert got["dddddd"] == (False, "stale: still marked running 48.0 h after it started")
    assert got["eeeeee"] == (False, "superseded")


def test_the_newest_passing_run_of_a_phase_is_kept_and_keep_protects_its_lineage(machine: Path) -> None:
    got = _plan(machine, keep=("20261112T000000Z-eeeeee",))
    assert got["eeeeee"] == (True, "named with --keep")
    assert _plan(machine)["cccccc"][0]  # ingest's newest pass is cccccc, so eeeeee is not "latest"


def test_a_dry_run_deletes_nothing_and_apply_deletes_exactly_the_plan(machine: Path, capsys: pytest.CaptureFixture[str],
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prune, "now_utc", lambda: NOW)
    monkeypatch.setattr(prune, "REPO_ROOT", machine.parent.parent)
    settings = KafulSettings(machines_dir=machine.parent)
    before = sorted(p.name for p in runs_dir(machine).iterdir())
    assert main(["runs", "prune", "m"], settings) == 0
    assert sorted(p.name for p in runs_dir(machine).iterdir()) == before
    out = capsys.readouterr().out
    assert "3 runs to delete" in out and "dry run: nothing deleted" in out
    plan = prune_plan(machine, RunsSettings(), NOW, root=machine.parent.parent)
    freed = apply_plan(machine, plan)
    left = sorted(p.name[-6:] for p in runs_dir(machine).iterdir())
    assert left == sorted(["aaaaaa", "0b0b0b", "f0f0f0", "cccccc", "111111", "999999"])
    assert freed > 0


def test_a_base_run_is_part_of_the_lineage(tmp_path: Path) -> None:
    machine_dir = tmp_path / "machines" / "m"
    _run(machine_dir, "20261101T000000Z-aaaaaa", 400, passed=("ingest",))
    _run(machine_dir, "20261117T000000Z-cccccc", 12, passed=tuple(PHASE_ORDER), base="20261101T000000Z-aaaaaa")
    plan = {v.run_id[-6:]: v for v in prune_plan(machine_dir, RunsSettings(), NOW, root=tmp_path)}
    assert plan["aaaaaa"].keep and plan["aaaaaa"].reason == "read by 20261117T000000Z-cccccc"
