"""Run ids, run directories, artifact save/load, phase selection and base-run choice."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.scope import Scope
from kaful.phases import PHASE_ORDER, PhaseSelectionError, select_phases, upstream_phases
from kaful.pipeline import RunRequest, start_run
from kaful.runs import (
    RUN_ID_PATTERN,
    BaseRunError,
    list_manifests,
    load_artifact,
    new_run_id,
    read_manifest,
    runs_dir,
)
from tests.harness import kaful_factories as f
from tests.harness.kaful_machine import MACHINE_ID, make_machine, settings_for


def test_run_ids_are_sortable_and_well_formed() -> None:
    early = new_run_id(datetime(2026, 11, 2, 9, 5, 7, tzinfo=UTC), "abc123")
    late = new_run_id(datetime(2026, 11, 2, 10, 0, 0, tzinfo=UTC))
    assert early == "20261102T090507Z-abc123"
    assert RUN_ID_PATTERN.match(late) and early < late
    with pytest.raises(ValueError):
        new_run_id(datetime(2026, 1, 1, tzinfo=UTC), "XYZ")


def test_select_phases() -> None:
    assert select_phases(None, ()) == PHASE_ORDER
    assert select_phases("verify", ()) == PHASE_ORDER[5:]
    assert select_phases(None, ("scope", "understand")) == ("understand", "scope")
    assert select_phases(None, ("verify", "estimate")) == ("verify", "estimate")  # synthesize optional
    with pytest.raises(PhaseSelectionError, match="contiguous"):
        select_phases(None, ("scope", "verify"))
    with pytest.raises(PhaseSelectionError, match="not both"):
        select_phases("scope", ("scope",))


def test_upstream_phases_skip_optional_ones() -> None:
    assert upstream_phases(("ingest",)) == ()
    assert upstream_phases(("estimate",)) == PHASE_ORDER[:6]


def test_start_run_creates_run_dir_and_manifest(tmp_path: Path) -> None:
    machine_dir = make_machine(tmp_path)
    ctx, base = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=("run",)))
    assert base is None
    assert ctx.run_dir.parent == runs_dir(machine_dir) == machine_dir / "outputs" / "runs"
    manifest = read_manifest(ctx.run_dir)
    assert manifest == ctx.manifest
    assert [i.path for i in manifest.inputs] == ["machine.toml", "manual/toy_manual.txt"]
    assert manifest.budget_usd == settings_for(tmp_path).default_budget_usd
    assert "machines_dir" not in manifest.config
    assert str(tmp_path) not in (ctx.run_dir / "manifest.json").read_text()


def test_missing_listed_document_is_an_error(tmp_path: Path) -> None:
    machine_dir = make_machine(tmp_path)
    (machine_dir / "manual" / "toy_manual.txt").unlink()
    with pytest.raises(FileNotFoundError, match="toy_manual"):
        start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))


def test_save_and_load_artifact(tmp_path: Path) -> None:
    make_machine(tmp_path)
    ctx, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    scope = f.scope().model_copy(update={"run_id": ctx.manifest.run_id})
    ref = ctx.save_artifact("scope", scope)
    assert ref.path == f"outputs/runs/{ctx.manifest.run_id}/scope/scope.json"
    assert load_artifact(ctx.machine_dir, ref, Scope) == scope
    path = ctx.machine_dir / ref.path
    path.write_text(path.read_text().replace("documented", "DOCUMENTED"))
    with pytest.raises(ValueError, match="sha256"):
        load_artifact(ctx.machine_dir, ref, Scope)


def test_save_artifact_rejects_foreign_run_id(tmp_path: Path) -> None:
    make_machine(tmp_path)
    ctx, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    with pytest.raises(ValueError, match="run_id"):
        ctx.save_artifact("scope", f.scope())


def _passed_run(tmp_path: Path) -> str:
    """A finished run whose phases all passed."""
    ctx, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    for phase in PHASE_ORDER:
        ctx.set_phase(PhaseRecord(phase=phase, status="passed", cost_usd=0.25))
    assert ctx.manifest.total_cost_usd == pytest.approx(0.25 * len(PHASE_ORDER))
    return ctx.manifest.run_id


def test_base_run_is_chosen_automatically_when_inputs_match(tmp_path: Path) -> None:
    make_machine(tmp_path)
    base_id = _passed_run(tmp_path)
    ctx, base = start_run(settings_for(tmp_path),
                          RunRequest(machine_id=MACHINE_ID, argv=(), from_phase="scope"))
    assert base is not None and base.run_id == base_id
    assert ctx.manifest.base_run is not None
    assert ctx.manifest.base_run.model_dump() == {"run_id": base_id, "selected_by": "auto", "inputs_match": True}
    assert [m.run_id for m in list_manifests(ctx.machine_dir)] == [base_id, ctx.manifest.run_id]


def test_changed_inputs_block_automatic_base_run(tmp_path: Path) -> None:
    machine_dir = make_machine(tmp_path)
    base_id = _passed_run(tmp_path)
    (machine_dir / "manual" / "toy_manual.txt").write_text("Replace the seal every 1500 h.\n")
    request = RunRequest(machine_id=MACHINE_ID, argv=(), from_phase="scope")
    with pytest.raises(BaseRunError, match="manual/toy_manual.txt: changed"):
        start_run(settings_for(tmp_path), request)
    explicit = RunRequest(machine_id=MACHINE_ID, argv=(), from_phase="scope", base_run=base_id)
    ctx, _ = start_run(settings_for(tmp_path), explicit)
    assert ctx.manifest.base_run is not None and not ctx.manifest.base_run.inputs_match
    (override,) = ctx.manifest.overrides
    assert (override.kind, override.target) == ("base_run", base_id)
    assert "manual/toy_manual.txt: changed" in override.reason


def test_base_run_needs_passed_upstream_phases(tmp_path: Path) -> None:
    make_machine(tmp_path)
    request = RunRequest(machine_id=MACHINE_ID, argv=(), from_phase="scope")
    with pytest.raises(BaseRunError, match="no earlier run passed"):
        start_run(settings_for(tmp_path), request)
    first, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    first.set_phase(PhaseRecord(phase="ingest", status="passed"))
    first.set_phase(PhaseRecord(phase="understand", status="failed"))
    with pytest.raises(BaseRunError, match="did not pass"):
        start_run(settings_for(tmp_path), RunRequest(
            machine_id=MACHINE_ID, argv=(), from_phase="scope", base_run=first.manifest.run_id))
    with pytest.raises(BaseRunError, match="not found"):
        start_run(settings_for(tmp_path), RunRequest(
            machine_id=MACHINE_ID, argv=(), from_phase="scope", base_run="20200101T000000Z-000000"))


def test_manifest_json_has_sorted_keys(tmp_path: Path) -> None:
    make_machine(tmp_path)
    ctx, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    data = json.loads((ctx.run_dir / "manifest.json").read_text())
    assert list(data) == sorted(data)


def test_runs_in_the_same_second_are_ordered_by_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: run ids resolve to the second, so a later run can have a smaller id."""
    make_machine(tmp_path)
    ids = iter(["20261102T120000Z-ffffff", "20261102T120000Z-000000"])
    monkeypatch.setattr("kaful.pipeline.new_run_id", lambda now: next(ids))
    first = _passed_run(tmp_path)
    second, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=()))
    for phase in PHASE_ORDER:
        second.set_phase(PhaseRecord(phase=phase, status="passed"))
    assert [m.run_id for m in list_manifests(second.machine_dir)] == [first, second.manifest.run_id]
    monkeypatch.setattr("kaful.pipeline.new_run_id", lambda now: "20261102T120001Z-aaaaaa")
    ctx, _ = start_run(settings_for(tmp_path), RunRequest(machine_id=MACHINE_ID, argv=(), from_phase="scope"))
    assert ctx.manifest.base_run is not None and ctx.manifest.base_run.run_id == "20261102T120000Z-000000"
