"""Verify (M11), checkpoint 1: the static re-check, the simulations and the dynamic checks.

Each dynamic check is shown failing on a trace broken in exactly one way, then the whole of steps a-c runs on the
small machine of the scope, parameterize and build tests (a twin that passes, one whose life prior no longer
matches its rate, and one with a threshold renewal)."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from kaful.artifacts.distributions import Lognormal
from kaful.artifacts.twin_definition import MaintenanceRule
from kaful.config import DeclaredMaintenance, KafulSettings
from kaful.phases.p6_verify.a_recheck_static import static_results
from kaful.phases.p6_verify.b_simulate import simulate
from kaful.phases.p6_verify.c_check_dynamics import dynamic_results
from kaful.phases.p6_verify.helpers import trace_checks as tc
from kaful.phases.p6_verify.helpers.traces import BindingTrace, Reset
from tests.unit.test_kaful_build import _twin
from tests.unit.test_kaful_scope import _contract, _machine

V = KafulSettings().verify
TOL = V.damage_tolerance
N, RATE = 101, 0.01  # D grows 0.01 per step of 1.0 and reaches D_limit = 1 at t = 100


def _trace(damage: Any = None, **changes: Any) -> BindingTrace:
    d = np.asarray(damage if damage is not None else RATE * np.arange(N), dtype=np.float64)
    health = np.clip(1.0 - d, 0.0, 1.0)
    base = BindingTrace(component="seal", times=np.arange(d.size, dtype=np.float64), damage=d, limit=1.0,
                        health=health, event_state=health.copy(), met=d >= 1.0, renewals=None, resets=(), recovery={})
    return replace(base, **changes)


def _rule(action: str = "replace") -> MaintenanceRule:
    return MaintenanceRule(id=f"seal_{action}", binding="seal", action=action,
                           recovery_param=None if action == "replace" else "seal.recovery_clean", trigger="schedule",
                           trigger_channel="maintenance_seal", trigger_value=f"seal_{action}", interval=50.0,
                           source="manual", ref="manual.pdf p.3")


def _with_reset(action: str, recovery: float, after: float) -> BindingTrace:
    """A reset at t = 50 that leaves D at `after`."""
    d = RATE * np.arange(N)
    d[50:] = after + RATE * np.arange(N - 50)
    return _trace(d, resets=(Reset(rule=_rule(action), t=50.0, window=(49, 50, 51)),),
                  recovery={f"seal_{action}": recovery})


def test_an_intact_trace_passes_every_check() -> None:
    tr = _trace()
    for check in (tc.starts_healthy, tc.finite_bounded, tc.monotone, tc.event_agreement):
        assert check(tr, TOL).ok, check.__name__
    assert tc.maintenance_resets(tr, TOL) is None  # nothing to check
    assert tc.life_consistency(tr, 100.0, V.life_band).ok


def test_an_event_at_t0_or_a_part_that_starts_worn_fails_starts_healthy() -> None:
    worn = tc.starts_healthy(_trace(0.2 + RATE * np.arange(N)), TOL)
    assert not worn.ok and "D starts at 0.2" in worn.message and "health starts at 0.8" in worn.message
    met = _trace().met.copy()
    met[0] = True
    assert "already triggered at t=0" in tc.starts_healthy(_trace(met=met), TOL).message


def test_a_negative_damage_hidden_by_the_clipped_health_fails_finite_bounded() -> None:
    """The backend clips health to [0, 1], so D < 0 shows only on the raw state (PLAN M11 b)."""
    d = -RATE * np.arange(N)
    found = tc.finite_bounded(_trace(d), TOL)
    assert not found.ok and "D falls to -1 < 0" in found.message


def test_nan_or_a_health_that_is_not_one_minus_d_over_the_limit_fails_finite_bounded() -> None:
    d = RATE * np.arange(N)
    d[30] = np.nan
    assert "D is not finite from t=30" in tc.finite_bounded(_trace(d), TOL).message
    off = _trace().health.copy()
    off[10] = 1.0
    assert "health is not 1 - D/D_limit" in tc.finite_bounded(_trace(health=off), TOL).message


def test_damage_that_falls_with_no_maintenance_fails_monotone_but_a_scheduled_reset_does_not() -> None:
    d = RATE * np.arange(N)
    d[50:] -= 0.3
    found = tc.monotone(_trace(d), TOL)
    assert not found.ok and "falls by 0.29 at t=50 with no maintenance" in found.message
    assert tc.monotone(_with_reset("replace", 0.0, 0.0), TOL).ok


def test_threshold_met_without_event_state_zero_fails_event_agreement() -> None:
    met = _trace().met.copy()
    met[90] = True
    found = tc.event_agreement(_trace(met=met), TOL)
    assert not found.ok and "threshold_met is True while event_state is 0.1 at t=90" in found.message


def _renewing(first: int = 100) -> BindingTrace:
    """Renewed at D_limit at t = 100: D returns to 0, the count goes to 1 and the event keeps holding."""
    d = RATE * np.arange(N + 50)
    d[100:] = RATE * np.arange(N + 50 - 100)
    renewals = (np.arange(N + 50) >= 100).astype(np.float64)
    health = np.clip(1.0 - d, 0.0, 1.0)
    return _trace(d, health=health, event_state=health.copy(), met=np.arange(N + 50) >= first, renewals=renewals)


def test_a_renewal_binding_agrees_when_the_event_first_holds_at_the_first_renewal() -> None:
    """PLAN M11 c: after a renewal event_state is back near 1 while threshold_met stays True, by design."""
    assert tc.event_agreement(_renewing(), TOL).ok
    assert tc.maintenance_resets(_renewing(), TOL).ok  # type: ignore[union-attr]
    early = tc.event_agreement(_renewing(first=90), TOL)
    assert not early.ok and "first holds at t=90 but the first renewal is at t=100" in early.message


def test_a_reset_that_leaves_the_wrong_damage_fails_maintenance_resets() -> None:
    assert tc.maintenance_resets(_with_reset("replace", 0.0, 0.0), TOL).ok  # type: ignore[union-attr]
    assert tc.maintenance_resets(_with_reset("clean", 0.5, 0.25), TOL).ok  # type: ignore[union-attr]
    found = tc.maintenance_resets(_with_reset("clean", 0.9, 0.25), TOL)
    assert found is not None and not found.ok and "seal_clean at t=50: D is 0.25" in found.message
    replaced = tc.maintenance_resets(_with_reset("replace", 0.0, 0.2), TOL)
    assert replaced is not None and "expected 0" in replaced.message


@pytest.mark.parametrize(("median", "verdict"), [(100.0, "pass"), (100.9, "pass"), (102.0, "too fast"),
                                                 (98.0, "too slow"), (500.0, "too fast")])
def test_life_consistency_fails_outside_the_band_in_both_directions(median: float, verdict: str) -> None:
    """Natalie 2026-11-13: ±1%, too fast and too slow both fail."""
    found = tc.life_consistency(_trace(), median, V.life_band)
    assert found.ok == (verdict == "pass") and (verdict == "pass" or verdict in found.message)


def test_a_part_that_never_reaches_its_limit_is_too_slow() -> None:
    found = tc.life_consistency(_trace(0.001 * np.arange(N)), 100.0, V.life_band)
    assert not found.ok and "no event before t=100" in found.message and "too slow" in found.message


def _verify(machine: Any = None, ps_change: Any = None) -> tuple[Any, Any, Any]:
    twin, ps, failures = _twin(machine)
    assert failures == ()
    ps = ps_change(ps) if ps_change else ps
    sims, composite = simulate(twin, ps, _contract(), "progpy", V)
    return static_results(twin, ps), composite, {(r.target, r.check): r for r in dynamic_results(twin, ps, sims, V)}


def test_the_small_machine_passes_every_check_on_every_run() -> None:
    static, composite, dynamic = _verify(_machine(maintenance=(
        DeclaredMaintenance(component="filter_element_fe", action="clean", interval=1000, unit="h"),)))
    assert {r.status for r in (*static, *composite)} == {"pass"}
    assert {r.status for r in dynamic.values()} == {"pass", "skipped"}
    assert dynamic[("bearings", "maintenance_resets")].status == "skipped"
    assert dynamic[("filter_element_fe", "maintenance_resets")].measured["resets"] > 0
    assert dynamic[("shaft_seal", "life_consistency")].measured["ratio"] == pytest.approx(1.0, abs=V.life_band)


def test_a_threshold_renewal_passes_event_agreement_and_counts_its_renewals() -> None:
    _, _, dynamic = _verify(_machine(maintenance=(
        DeclaredMaintenance(component="shaft_seal", action="replace", on_threshold=True),)))
    assert dynamic[("shaft_seal", "event_agreement")].status == "pass"
    assert dynamic[("shaft_seal", "maintenance_resets")].measured["renewals"] >= 1


def test_a_life_prior_that_no_longer_matches_the_rate_fails_only_that_component() -> None:
    def doubled(ps: Any) -> Any:
        life = ps.get("shaft_seal.life_nominal")
        assert isinstance(life.prior, Lognormal)
        prior = life.prior.model_copy(update={"median": 2 * life.prior.median})
        return ps.model_copy(update={"parameters": tuple(x.model_copy(update={"prior": prior}) if x is life else x
                                                         for x in ps.parameters)})

    _, _, dynamic = _verify(ps_change=doubled)
    failing = {k for k, r in dynamic.items() if r.status == "fail"}
    assert failing == {("shaft_seal", "life_consistency")}
    assert "too fast" in dynamic[("shaft_seal", "life_consistency")].message


def test_a_static_failure_is_recorded_against_its_target() -> None:
    twin, ps, _ = _twin()
    bad = ps.model_copy(update={"parameters": tuple(x.model_copy(update={"unit": "mm / h"}) if x.param_id == "shaft_seal.k"
                                                    else x for x in ps.parameters)})
    results = {(r.target, r.check): r for r in static_results(twin, bad)}
    assert results[("shaft_seal", "units")].status == "fail"
    assert results[("composite", "provenance")].status == "pass" and ("composite", "units") not in results
