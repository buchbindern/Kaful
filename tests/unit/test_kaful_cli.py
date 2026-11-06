"""`kaful run`: argument handling, manifests and exit codes, with no phase implemented
(the phase registry is emptied; ingest has its own tests)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kaful import pipeline
from kaful.artifacts.manifest import PhaseRecord, RunManifest
from kaful.cli import EXIT_FAILED, EXIT_NOT_IMPLEMENTED, main
from kaful.config import REPO_ROOT, load_machine_config
from kaful.phases import PHASE_ORDER
from kaful.runs import RunContext, list_manifests
from tests.harness.kaful_machine import MACHINE_ID, make_machine, settings_for


@pytest.fixture(autouse=True)
def no_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "PHASE_IMPLEMENTATIONS", {})


def _latest_manifest(machine_dir: Path) -> RunManifest:
    return list_manifests(machine_dir)[-1]


def test_full_run_records_every_phase_as_not_implemented(tmp_path: Path) -> None:
    machine_dir = make_machine(tmp_path)
    assert main(["run", MACHINE_ID], settings_for(tmp_path)) == EXIT_NOT_IMPLEMENTED
    manifest = _latest_manifest(machine_dir)
    assert manifest.status == "not_implemented" and manifest.finished_at is not None
    assert {r.status for r in manifest.phases} == {"not_implemented"}
    assert manifest.argv == ("run", MACHINE_ID)
    log_lines = (machine_dir / "outputs" / "runs" / manifest.run_id / "run.log.jsonl").read_text()
    assert manifest.run_id in log_lines and '"phase": "ingest"' in log_lines


def test_from_reuses_upstream_and_skips_nothing_silently(tmp_path: Path) -> None:
    machine_dir = make_machine(tmp_path)
    main(["run", MACHINE_ID], settings_for(tmp_path))
    base = _latest_manifest(machine_dir)
    # Mark the base run's phases as passed, as a finished real run would be.
    ctx = RunContext(settings_for(tmp_path), load_machine_config(machine_dir), machine_dir,
                     machine_dir / "outputs" / "runs" / base.run_id, base)
    for phase in PHASE_ORDER:
        ctx.set_phase(PhaseRecord(phase=phase, status="passed"))
    code = main(["run", MACHINE_ID, "--from", "build", "--budget", "2.5", "--allow-failures"],
                settings_for(tmp_path))
    assert code == EXIT_NOT_IMPLEMENTED
    manifest = _latest_manifest(machine_dir)
    statuses = {r.phase: r.status for r in manifest.phases}
    assert [statuses[p] for p in PHASE_ORDER[:4]] == ["reused"] * 4
    assert [statuses[p] for p in PHASE_ORDER[4:]] == ["not_implemented"] * 5
    assert manifest.base_run is not None and manifest.base_run.run_id == base.run_id
    assert manifest.budget_usd == 2.5 and manifest.allow_failures
    assert [o.kind for o in manifest.overrides] == ["allow_failures"]


def test_from_without_a_base_run_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_machine(tmp_path)
    assert main(["run", MACHINE_ID, "--from", "scope"], settings_for(tmp_path)) == EXIT_FAILED
    assert "no earlier run passed" in capsys.readouterr().err


@pytest.mark.parametrize(("args", "message"), [
    (["run", "ghost_machine"], "no machine.toml"),
    (["run", MACHINE_ID, "--only", "scope", "verify"], "contiguous"),
    (["run", MACHINE_ID, "--budget", "0"], "--budget must be > 0"),
])
def test_bad_requests_exit_with_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                        args: list[str], message: str) -> None:
    make_machine(tmp_path)
    assert main(args, settings_for(tmp_path)) == EXIT_FAILED
    assert message in capsys.readouterr().err


def test_from_and_only_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", MACHINE_ID, "--from", "scope", "--only", "scope"], settings_for(tmp_path))
    assert exc.value.code == 2


def test_machine_id_must_match_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    machine_dir = make_machine(tmp_path)
    toml = machine_dir / "machine.toml"
    toml.write_text(toml.read_text().replace('machine_id = "toy_machine"', 'machine_id = "other"'))
    assert main(["run", MACHINE_ID], settings_for(tmp_path)) == EXIT_FAILED
    assert "!= directory" in capsys.readouterr().err


def test_python_dash_m_kaful_uses_env_settings(tmp_path: Path) -> None:
    make_machine(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    env["KAFUL_MACHINES_DIR"] = str(tmp_path)
    # --from scope needs a base run: failing with that message shows the env machines dir was used.
    done = subprocess.run([sys.executable, "-B", "-m", "kaful", "run", MACHINE_ID, "--from", "scope"],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == EXIT_FAILED, done.stderr
    assert "no earlier run passed" in done.stderr
