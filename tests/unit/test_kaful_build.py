"""Build (M10), checkpoint 1: one binding per full-model component, wiring with provenance, maintenance rules
and profiles, and the gates, on the small two-document machine of the scope and parameterize tests."""

from typing import Any

import pytest

from kaful.artifacts.sensor_contract import SensorChannel
from kaful.config import DeclaredMaintenance, KafulSettings
from kaful.phases.p3_scope.phase import build_scope
from kaful.phases.p4_parameterize.phase import build_param_set
from kaful.phases.p5_build.b_wire import FLAG_UNDOCUMENTED
from kaful.phases.p5_build.c_maintenance import FLAG_NO_MAINTENANCE
from kaful.phases.p5_build.phase import build_twin
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_parameterize import _index, _judge_all
from tests.unit.test_kaful_scope import _channel, _contract, _machine, _model

SETTINGS = KafulSettings()


def _twin(machine: Any = None, model: Any = None, contract: Any = None) -> tuple[Any, Any, tuple[str, ...]]:
    machine, model, contract, index = machine or _machine(), model or _model(), contract or _contract(), _index(True)
    scope = build_scope({**BASE}, index, model, contract, machine, SETTINGS)[0]
    ps = build_param_set({**BASE}, index, model, scope, contract, machine, SETTINGS, _judge_all).param_set
    twin, failures = build_twin({**BASE}, ps, scope, contract, machine, SETTINGS)
    return twin, ps, failures


def test_one_binding_per_full_model_with_d_in_d_limit_units_and_one_event_each() -> None:
    twin, ps, failures = _twin()
    assert failures == ()
    assert [(b.id, b.template, b.damage_unit, b.fault_mode) for b in twin.bindings] == [
        ("bearings", "bearing_fatigue", "1", "bearings_flaking"),
        ("filter_element_fe", "accumulation", "1", "filter_clogging"),
        ("shaft_seal", "accumulation", "1", "seal_wear")]
    assert twin.bindings[0].params == {"k": "bearings.k", "life_exponent": "bearings.life_exponent"}
    assert [e.name for e in twin.events] == ["bearings_failure", "filter_element_fe_failure", "shaft_seal_failure"]
    assert {(i.binding, i.condition, i.provenance) for i in twin.external_inputs} == {
        ("bearings", "load", "undocumented"), ("bearings", "speed", "undocumented"),
        ("filter_element_fe", "load", "undocumented"), ("shaft_seal", "load", "undocumented")}
    assert FLAG_UNDOCUMENTED.format(condition="speed") in twin.flags["bearings"]


def test_a_driving_channel_of_the_right_dimension_feeds_the_condition() -> None:
    flow = _channel("flow", "Throughput").model_copy(update={"role": "operating_input", "unit": "m3/h"})
    model = _model()
    modes = tuple(m.model_copy(update={"driven_by": ("flow",)}) if m.id == "filter_clogging" else m
                  for m in model.fault_modes)
    twin, _, failures = _twin(model=model.model_copy(update={"fault_modes": modes}), contract=_contract(flow))
    [load] = [i for i in twin.external_inputs if i.binding == "filter_element_fe"]
    assert failures == () and (load.provenance, load.channel, load.unit) == ("channel", "flow", "m3/h")


def test_documented_and_declared_maintenance_become_rules_and_a_schedule() -> None:
    """The documents give the filter a 4000 h replacement and the seal a 2000 h one; a declared 1000 h clean
    of the filter adds a recovery parameter, and at 4000 h the replacement wins over the clean."""
    machine = _machine(maintenance=(DeclaredMaintenance(component="filter_element_fe", action="clean",
                                                        interval=1000, unit="h"),))
    twin, ps, failures = _twin(machine)
    assert failures == ()
    assert [(r.id, r.action, r.interval, r.source, r.recovery_param) for r in twin.maintenance] == [
        ("filter_element_fe_clean", "clean", 1000.0, "deployment", "filter_element_fe.recovery_clean"),
        ("filter_element_fe_replace", "replace", 4000.0, "manual", None),
        ("shaft_seal_replace", "replace", 2000.0, "manual", None)]
    assert ps.get("filter_element_fe.recovery_clean").prior.quantile(0.5) == pytest.approx(0.95)  # uniform 0.9-1
    scheduled = {p.name: p for p in twin.profiles}["documented_maintenance"].events
    at_4000 = [(e.channel, e.value) for e in scheduled if e.t == 4000.0]
    assert at_4000 == [("maintenance_filter_element_fe", "filter_element_fe_replace"),
                       ("maintenance_shaft_seal", "shaft_seal_replace")]
    assert {p.name: p for p in twin.profiles}["nominal"].events == ()
    assert FLAG_NO_MAINTENANCE in twin.flags["bearings"]


def test_a_wrong_k_unit_fails_the_gate() -> None:
    from kaful.phases.p5_build.d_check import check

    twin, ps, _ = _twin()
    bad = ps.model_copy(update={"parameters": tuple(x.model_copy(update={"unit": "mm / h"}) if x.param_id == "shaft_seal.k"
                                                    else x for x in ps.parameters)})
    assert check(twin, bad, ()) == ("binding shaft_seal: k unit 'mm / h' is not '1 / h'",)


@pytest.mark.parametrize("seed", ["0", "1"])
def test_the_twin_does_not_depend_on_hash_order(seed: str) -> None:
    import os
    import subprocess
    import sys

    from kaful.runs import dump_json

    code = "from tests.unit.test_kaful_build import _twin; from kaful.runs import dump_json; print(dump_json(_twin()[0]))"
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": seed})
    assert out.stdout.strip() == dump_json(_twin()[0]).strip()


def test_a_calendar_interval_goes_through_the_duty_cycle_and_a_threshold_replacement_schedules_nothing() -> None:
    """Natalie 2026-11-10: daily cleaning converts through the default duty cycle (flagged); the eversys water
    filter is replaced when its warning fires (on_threshold), never on a schedule."""
    from kaful.phases.p5_build.c_maintenance import FLAG_CALENDAR

    machine = _machine(maintenance=(
        DeclaredMaintenance(component="filter_element_fe", action="clean", interval=1, unit="day"),
        DeclaredMaintenance(component="shaft_seal", action="replace", on_threshold=True)))
    twin, _, failures = _twin(machine)
    rules = {r.id: r for r in twin.maintenance}
    duty = SETTINGS.parameterize.duty_defaults["hour"].per_year_median
    assert failures == () and rules["filter_element_fe_clean"].interval == pytest.approx(duty / 365.25)
    assert FLAG_CALENDAR.format(rule="filter_element_fe_clean") in twin.flags["filter_element_fe"]
    seal = rules["shaft_seal_replace"]
    assert (seal.trigger, seal.interval, seal.trigger_channel) == ("threshold", None, None)
    scheduled = {p.name: p for p in twin.profiles}["documented_maintenance"].events
    assert not [e for e in scheduled if e.channel == "maintenance_shaft_seal"]
