"""Pinned sandbox drivers (M11, Natalie 2026-11-13): the only code the sandbox runs from outside a machine
directory is a committed driver whose content still has its pinned sha256, checked by the parent and again by
the child; its target must be a file inside the machine directory.

The registry is empty since M15 deleted the legacy probe (its only driver); these tests pin a test-only driver."""

import json
from pathlib import Path

import pytest

from kaful.config import SandboxSettings
from kaful.runs import sha256_file
from kaful.sandbox import drivers, runner
from kaful.sandbox.drivers import DRIVERS, REPO_ROOT, Driver, DriverNotAllowed, driver_file
from kaful.sandbox.runner import run_driver

FAST = SandboxSettings(timeout_s=60.0, cpu_limit_s=60)
TEST_DRIVER = "tests/fixtures/sandbox/line_count_driver.py"


@pytest.fixture
def line_count(monkeypatch: pytest.MonkeyPatch) -> Driver:
    driver = Driver(path=TEST_DRIVER, sha256=sha256_file(REPO_ROOT / TEST_DRIVER))
    monkeypatch.setitem(drivers.DRIVERS, "line_count", driver)
    return driver


def _machine_file(machine_dir: Path, rel: str = "outputs/notes.txt", text: str = "a\nb\nc\n") -> str:
    (machine_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (machine_dir / rel).write_text(text)
    return rel


def test_every_pinned_driver_still_has_its_pinned_hash() -> None:
    """Editing a driver needs its new hash in kaful/sandbox/drivers.py, a reviewed change."""
    for name, driver in DRIVERS.items():
        assert sha256_file(REPO_ROOT / driver.path) == driver.sha256, name
        assert driver_file(name) == REPO_ROOT / driver.path


def test_a_name_that_is_not_pinned_is_refused() -> None:
    with pytest.raises(DriverNotAllowed, match="not a pinned sandbox driver"):
        driver_file("run_anything")


def test_a_driver_whose_content_changed_is_refused_before_the_child_starts(monkeypatch: pytest.MonkeyPatch,
                                                                         tmp_path: Path, line_count: Driver) -> None:
    monkeypatch.setitem(drivers.DRIVERS, "line_count", Driver(path=line_count.path, sha256="0" * 64))
    rel = _machine_file(tmp_path)
    with pytest.raises(DriverNotAllowed, match="pinned 0000"):
        run_driver(tmp_path, "line_count", rel, "count", {}, settings=FAST)


def test_the_child_refuses_a_file_that_changed_after_the_parent_checked_it(tmp_path: Path) -> None:
    code = tmp_path / "driver.py"
    code.write_text("def run(payload):\n    return 1\n")
    base = {"code_path": "driver.py", "code_sha256": "1" * 64, "entrypoint": "run"}
    result = runner._spawn(code, base, None, FAST)
    assert result.status == "error" and "refused" in (result.error_message or "")


@pytest.mark.parametrize("target", ["../outside.py", "/etc/hosts"])
def test_a_target_outside_the_machine_directory_is_refused(tmp_path: Path, target: str, line_count: Driver) -> None:
    with pytest.raises(ValueError):
        run_driver(tmp_path / "machine", "line_count", target, "count", {}, settings=FAST)


def test_a_pinned_driver_runs_on_its_target_and_records_what_ran(tmp_path: Path, line_count: Driver) -> None:
    rel = _machine_file(tmp_path)
    result = run_driver(tmp_path, "line_count", rel, "count", {}, settings=FAST)
    assert result.status == "ok", result.traceback
    assert (result.code_path, result.code_sha256) == (line_count.path, line_count.sha256)
    assert (result.target_path, result.target_sha256) == (rel, sha256_file(tmp_path / rel))
    assert result.value == {"lines": 3}
    json.dumps(result.value)
