"""Pinned sandbox drivers (M11, Natalie 2026-11-13): the only code the sandbox runs from outside a machine
directory is a committed driver whose content still has its pinned sha256, checked by the parent and again by
the child; its target must be a file inside the machine directory."""

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from kaful.config import SandboxSettings
from kaful.runs import sha256_file
from kaful.sandbox import drivers, runner
from kaful.sandbox.drivers import DRIVERS, REPO_ROOT, Driver, DriverNotAllowed, driver_file
from kaful.sandbox.runner import run_driver

FAST = SandboxSettings(timeout_s=60.0, cpu_limit_s=60)
COMPONENT = textwrap.dedent("""
    import numpy as np
    from progpy import PrognosticsModel

    class Seal(PrognosticsModel):
        inputs = ["load"]
        states = ["seal_health"]
        outputs = ["leak"]
        events = ["seal_failure"]
        default_parameters = {"x0": {"seal_health": 1.0}, "rate": 0.01}

        def initialize(self, u=None, z=None):
            return self.StateContainer({"seal_health": 1.0})

        def next_state(self, x, u, dt):
            x["seal_health"] = max(x["seal_health"] - self.parameters["rate"] * u["load"] * dt, 0.0)
            return x

        def output(self, x):
            return self.OutputContainer({"leak": 1.0 - x["seal_health"]})

        def event_state(self, x):
            return {"seal_failure": max(float(x["seal_health"]), 0.0)}

        def threshold_met(self, x):
            return {"seal_failure": bool(x["seal_health"] <= 0.0)}
""")


def _legacy_component(machine_dir: Path, name: str = "seal", body: str = COMPONENT) -> str:
    rel = f"outputs/codegen/components/{name}.py"
    (machine_dir / rel).parent.mkdir(parents=True, exist_ok=True)
    (machine_dir / rel).write_text(body)
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
                                                                         tmp_path: Path) -> None:
    pinned = DRIVERS["legacy_probe"]
    monkeypatch.setitem(drivers.DRIVERS, "legacy_probe", Driver(path=pinned.path, sha256="0" * 64))
    rel = _legacy_component(tmp_path)
    with pytest.raises(DriverNotAllowed, match="pinned 0000"):
        run_driver(tmp_path, "legacy_probe", rel, "probe", {}, settings=FAST)


def test_the_child_refuses_a_file_that_changed_after_the_parent_checked_it(tmp_path: Path) -> None:
    code = tmp_path / "driver.py"
    code.write_text("def run(payload):\n    return 1\n")
    base = {"code_path": "driver.py", "code_sha256": "1" * 64, "entrypoint": "run"}
    result = runner._spawn(code, base, None, FAST)
    assert result.status == "error" and "refused" in (result.error_message or "")


@pytest.mark.parametrize("target", ["../outside.py", "/etc/hosts"])
def test_a_target_outside_the_machine_directory_is_refused(tmp_path: Path, target: str) -> None:
    with pytest.raises(ValueError):
        run_driver(tmp_path / "machine", "legacy_probe", target, "probe", {}, settings=FAST)


def test_the_probe_steps_a_legacy_component_and_records_what_ran(tmp_path: Path) -> None:
    rel = _legacy_component(tmp_path)
    result = run_driver(tmp_path, "legacy_probe", rel, "probe", {
        "mode": "component", "name": "seal", "inputs": ["load"], "steps": 150, "dt": 1.0}, settings=FAST)
    assert result.status == "ok", result.traceback
    assert (result.code_path, result.code_sha256) == (DRIVERS["legacy_probe"].path, DRIVERS["legacy_probe"].sha256)
    assert (result.target_path, result.target_sha256) == (rel, sha256_file(tmp_path / rel))
    trace: Any = result.value
    assert isinstance(trace, dict) and trace["error"] is None
    health = trace["states"]["seal_health"]
    assert health[0] == 1.0 and len(health) == 151 and health[100] == pytest.approx(0.0, abs=1e-9)
    assert trace["threshold_met"]["seal_failure"][-1] is True
    json.dumps(trace)  # plain JSON, no NaN
