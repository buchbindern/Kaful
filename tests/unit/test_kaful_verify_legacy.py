"""The legacy baseline (M11): the verify checks on legacy twins through the sandbox, and the two pass rates
(Natalie 2026-11-13: dynamic, the comparable number for M15; full, 0 for legacy by construction)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaful.config import KafulSettings, SandboxSettings
from kaful.phases.p6_verify.tools.legacy_baseline import baseline, score_component
from tests.unit.test_kaful_sandbox_drivers import COMPONENT, _legacy_component

V = KafulSettings().verify
SETTINGS = KafulSettings(sandbox=SandboxSettings(timeout_s=60.0, cpu_limit_s=60))
SPEC = {"inputs": [{"name": "load"}], "parameters": [{"name": "rate", "default": 0.01, "unit": "1/step"}],
        "states": [{"name": "seal_health", "state_type": "degradation", "min_value": 0.0, "max_value": 1.0}]}
COMPOSITE = """
import os, sys
component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "components")  # as legacy composites do
if component_dir not in sys.path:
    sys.path.insert(0, component_dir)
from seal import Seal
composite_model = Seal()

def future_loading_eqn(t, x=None):
    return composite_model.InputContainer({"load": 1.0})
"""


def _machine(tmp_path: Path) -> Path:
    """Two legacy components: `seal` wears from 1.0; `pump` is the same model with its health starting at 100
    (the eversys pattern OVERVIEW §0 found)."""
    machine = tmp_path / "legacy_machine"
    _legacy_component(machine, "seal")
    _legacy_component(machine, "pump", COMPONENT.replace("class Seal", "class Pump").replace(
        '"seal_health": 1.0})', '"seal_health": 100.0})'))
    (machine / "outputs/codegen/composite_model.py").write_text(COMPOSITE)
    (machine / "outputs/codegen/specs").mkdir(parents=True)
    for name in ("pump", "seal"):
        (machine / f"outputs/codegen/specs/{name}.json").write_text(json.dumps(SPEC))
    (machine / "outputs/triage").mkdir(parents=True)
    (machine / "outputs/triage/step_c_full_components.json").write_text(json.dumps([{"name": "seal"},
                                                                                    {"name": "pump"}]))
    return machine


def test_a_legacy_machine_gets_both_pass_rates(tmp_path: Path) -> None:
    result = baseline(_machine(tmp_path), "legacy_machine", SETTINGS, datetime(2026, 11, 13, tzinfo=timezone.utc))
    by = {c.component: {r.check: r for r in c.results} for c in result.components}
    assert [c.component for c in result.components] == ["pump", "seal"]
    assert (result.dynamic_pass_rate, result.full_pass_rate) == (0.5, 0.0)
    assert by["pump"]["starts_healthy"].status == "fail" and "starts at 100.0" in by["pump"]["starts_healthy"].message
    assert {k: r.status for k, r in by["seal"].items()} == {
        "event_agreement": "pass", "finite_bounded": "pass", "life_consistency": "skipped", "loads": "pass",
        "monotone": "pass", "provenance": "fail", "starts_healthy": "pass", "tracking_converges": "skipped"}
    assert {r.check: r.status for r in result.composite} == {
        "event_agreement": "pass", "finite_bounded": "pass", "loads": "pass", "starts_healthy": "pass"}


def _trace(**states: Any) -> dict[str, Any]:
    return {"states": states, "event_state": {}, "threshold_met": {}, "outputs_t0": {}, "error": None}


def test_a_tracking_state_that_keeps_oscillating_does_not_converge() -> None:
    spec = {"states": [{"name": "temp", "state_type": "tracking"}]}
    settled = [20.0 + 10.0 * (1 - 0.9 ** i) for i in range(200)]
    oscillating = [20.0 + (5.0 if i % 2 else -5.0) for i in range(200)]
    ok = {r.check: r for r in score_component("c", spec, _trace(temp=settled), V)}["tracking_converges"]
    bad = {r.check: r for r in score_component("c", spec, _trace(temp=oscillating), V)}["tracking_converges"]
    assert ok.status == "pass" and bad.status == "fail" and "still moves 10" in bad.message


def test_a_component_that_does_not_load_fails_loads_and_runs_nothing_else() -> None:
    trace = _trace() | {"error": {"step": None, "where": "load", "message": "SyntaxError: bad"}}
    results = {r.check: r for r in score_component("c", SPEC, trace, V)}
    assert results["loads"].status == "fail" and results["monotone"].status == "skipped"
