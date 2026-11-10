"""The ProgPy adapter (M10, checkpoint 2): calibration, maintenance resets, channel inputs, determinism and
isolation, on the small machine of the scope, parameterize and build tests."""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kaful.artifacts.forecast import OperatingProfile, ProfileSegment
from kaful.backends.base import BuildError
from kaful.backends.progpy.backend import ProgPyBackend
from kaful.config import DeclaredMaintenance, DeclaredSpec
from tests.unit.test_kaful_build import _twin
from tests.unit.test_kaful_scope import _channel, _contract, _machine, _model

BACKEND = ProgPyBackend()
REPO = Path(__file__).resolve().parents[2]


def _run(machine: Any = None, profile: str = "nominal", dt: float = 10.0, horizon: float = 30000.0,
         values: dict[str, float] | None = None, model: Any = None, contract: Any = None,
         custom: OperatingProfile | None = None) -> tuple[Any, Any, Any]:
    twin, ps, failures = _twin(machine, model, contract)
    assert failures == ()
    built = BACKEND.build(twin, ps, contract or _contract())
    chosen = custom or {p.name: p for p in twin.profiles}[profile]
    return BACKEND.simulate(built, chosen, values or {}, dt=dt, horizon=horizon, seed=0), ps, twin


def test_a_nominal_run_reaches_d_limit_at_the_median_life_of_every_binding() -> None:
    result, ps, _ = _run()
    for event, when in result.event_times.items():
        component = event.removesuffix("_failure")
        median = ps.get(f"{component}.life_nominal").prior.quantile(0.5)
        assert when is not None and when == pytest.approx(median, rel=0.01, abs=10.0)  # within 1% (or one step)
    health = result.outputs["shaft_seal_health"]
    assert health[0] == 1.0 and np.all(np.diff(health) <= 0) and health[-1] == 0.0


def test_scheduled_replacement_renews_the_part_and_a_clean_removes_the_recovered_fraction() -> None:
    machine = _machine(maintenance=(DeclaredMaintenance(component="shaft_seal", action="clean", interval=1000,
                                                        unit="h"),))
    result, ps, _ = _run(machine, profile="documented_maintenance", values={"shaft_seal.recovery_clean": 0.5})
    t, filt, seal = result.times, result.states["filter_element_fe_damage"], result.states["shaft_seal_damage"]
    before, after = seal[np.searchsorted(t, 990.0)], seal[np.searchsorted(t, 1000.0)]
    assert after == pytest.approx(0.5 * before, rel=0.02)  # a clean with recovery 0.5 halves D
    assert filt[np.searchsorted(t, 4000.0)] < filt[np.searchsorted(t, 3990.0)] / 100  # the 4000 h replacement
    assert result.event_times["filter_element_fe_failure"] is None  # replaced before its median life (5,370 h)
    assert result.event_times["shaft_seal_failure"] is None  # halved every 1000 h (2000 h replacement too)


def test_a_threshold_replacement_renews_at_d_limit_and_counts_each_renewal() -> None:
    machine = _machine(maintenance=(DeclaredMaintenance(component="shaft_seal", action="replace", on_threshold=True),))
    result, ps, _ = _run(machine, profile="documented_maintenance", horizon=12000.0)
    median = ps.get("shaft_seal.life_nominal").prior.quantile(0.5)
    renewals = result.states["shaft_seal_renewals"]
    assert renewals[-1] == 2.0  # 12,000 h is two lives of about 5,370 h
    assert result.event_times["shaft_seal_failure"] == pytest.approx(median, rel=0.01, abs=10.0)
    assert result.states["shaft_seal_damage"].max() <= 1.0 + 10.0 / median


def test_a_driving_channel_with_a_declared_nominal_scales_the_rate() -> None:
    """[[spec]] declares the filter's nominal flow and load exponent; at twice the flow the life halves."""
    flow = _channel("flow", "Throughput").model_copy(update={"role": "operating_input", "unit": "m3/h"})
    model = _model()
    model = model.model_copy(update={"fault_modes": tuple(
        m.model_copy(update={"driven_by": ("flow",)}) if m.id == "filter_clogging" else m for m in model.fault_modes)})
    machine = _machine(spec=(DeclaredSpec(component="filter_element_fe", name="load_exponent", value=1.0, unit="1"),
                             DeclaredSpec(component="filter_element_fe", name="load_nominal", value=600.0,
                                          unit="L/min")))  # 36 m3/h, in another unit
    profile = OperatingProfile(name="double", segments=(ProfileSegment(start=0, end=30000, values={"flow": 72.0}),))
    result, ps, _ = _run(machine, model=model, contract=_contract(flow), custom=profile)
    median = ps.get("filter_element_fe.life_nominal").prior.quantile(0.5)
    assert result.event_times["filter_element_fe_failure"] == pytest.approx(median / 2, rel=0.02, abs=10.0)


def test_a_temperature_channel_is_converted_to_kelvin() -> None:
    from kaful.backends.progpy.model import _affine

    assert _affine("degC", "kelvin") == pytest.approx((1.0, 273.15))
    assert _affine("L/min", "m**3/h") == pytest.approx((0.06, 0.0))


def test_an_unknown_family_or_a_missing_parameter_is_a_build_error_naming_the_component() -> None:
    twin, ps, _ = _twin()
    odd = twin.model_copy(update={"bindings": tuple(b.model_copy(update={"template": "paris"}) if b.id == "bearings"
                                                    else b for b in twin.bindings)})
    with pytest.raises(BuildError) as exc:
        BACKEND.build(odd, ps, _contract())
    assert exc.value.problems == {"bearings": "no template family 'paris'"}


@pytest.mark.parametrize("seed", ["0", "1"])
def test_the_rendered_model_and_its_simulation_do_not_depend_on_hash_order(seed: str) -> None:
    import os
    import subprocess
    import sys

    code = ("from tests.unit.test_kaful_progpy import _run; r = _run(horizon=6000.0)[0]; "
            "print(sorted(r.states), [round(float(v), 9) for k in sorted(r.states) for v in r.states[k][::50]], "
            "sorted(r.event_times.items()))")
    run = [sys.executable, "-B", "-W", "ignore", "-c", code]
    out = subprocess.run(run, capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed})
    other = subprocess.run(run, capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": "7"})
    assert out.stdout == other.stdout


def test_progpy_is_imported_only_in_the_adapter() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+progpy\b", re.M)
    leaks = sorted(str(p.relative_to(REPO)) for p in (REPO / "kaful").rglob("*.py")
                   if pattern.search(p.read_text(encoding="utf-8")) and "backends/progpy" not in str(p))
    assert leaks == []
