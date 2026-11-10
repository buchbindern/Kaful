"""Property tests for the first template families (M9, hypothesis): the protocol, rate >= 0, f(nominal) = 1 (so
calibration round-trips), monotone in each condition, zero load -> zero damage where that applies, health starts
at 1.0, stays in [0, 1] and never rises, and simulating at nominal reaches D_limit at the calibrated life.
Derandomized, so a run is the same everywhere."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kaful.templates.families import FAMILIES, Accumulation, Arrhenius, BearingFatigue
from kaful.templates.protocol import DegradationTemplate, health_from_damage

FAMILY_NAMES = sorted(FAMILIES)
PROPERTY = settings(derandomize=True, max_examples=200, deadline=None)
SIMULATION = settings(derandomize=True, max_examples=25, deadline=None)
Case = tuple[str, dict[str, float], dict[str, float], float, float]


def _f(low: float, high: float) -> st.SearchStrategy[float]:
    return st.floats(low, high, allow_nan=False, allow_infinity=False)


@st.composite
def _params(draw: st.DrawFn, name: str) -> tuple[dict[str, float], dict[str, float]]:
    if name == "accumulation":
        p = {"load_exponent": draw(_f(0.1, 3.0)), "load_nominal": draw(_f(0.1, 1e4))}
        return p, {"load": p["load_nominal"]}
    if name == "arrhenius":
        p = {"activation_energy": draw(_f(2e4, 2e5)), "temperature_nominal": draw(_f(280.0, 420.0))}
        return p, {"temperature": p["temperature_nominal"]}
    p = {"life_exponent": draw(st.sampled_from([3.0, 10 / 3])), "load_nominal": draw(_f(100.0, 5e4)),
         "speed_nominal": draw(_f(10.0, 3e4))}
    return p, {"load": p["load_nominal"], "speed": p["speed_nominal"]}


@st.composite
def cases(draw: st.DrawFn, names: tuple[str, ...] = tuple(FAMILY_NAMES)) -> Case:
    name = draw(st.sampled_from(names))
    params, nominal = draw(_params(name))
    return name, params, nominal, draw(_f(1.0, 1e6)), draw(_f(1e-3, 1e3))


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_each_family_satisfies_the_protocol(name: str) -> None:
    template = FAMILIES[name]()
    assert isinstance(template, DegradationTemplate)
    assert {"k"} <= {s.name for s in template.params}


@PROPERTY
@given(cases())
def test_f_is_one_at_nominal_so_calibration_round_trips(case: Case) -> None:
    name, params, nominal, life, limit = case
    template = FAMILIES[name]()
    assert float(template.multiplier(nominal, params)) == pytest.approx(1.0)
    k = template.calibrate_k(life, nominal, params, limit)
    assert k == pytest.approx(limit / life)
    assert template.life_at(k, nominal, params, limit) == pytest.approx(life)
    assert float(template.multiplier({}, params)) == pytest.approx(1.0)  # a missing condition is nominal


@PROPERTY
@given(cases())
def test_rate_is_never_negative_and_rises_with_each_condition(case: Case) -> None:
    name, params, nominal, life, limit = case
    template = FAMILIES[name]()
    p = {**params, "k": template.calibrate_k(life, nominal, params, limit)}
    for cond, value in nominal.items():
        grid = np.linspace(0.5, 2.0, 7) * value if cond == "temperature" else np.linspace(0.0, 3.0, 7) * value
        rates = template.rate(np.zeros(7), {**nominal, cond: grid}, p)
        assert np.all(rates >= 0) and np.all(np.diff(rates) >= -1e-12 * np.max(rates))


@PROPERTY
@given(cases(("accumulation", "bearing_fatigue")))
def test_zero_load_means_zero_damage(case: Case) -> None:
    name, params, nominal, life, limit = case
    template = FAMILIES[name]()
    k = template.calibrate_k(life, nominal, params, limit)
    for cond in nominal:
        rate = template.rate(np.zeros(1), {**nominal, cond: np.zeros(1)}, {**params, "k": k})
        assert float(rate[0]) == 0.0


@SIMULATION
@given(cases())
def test_simulated_health_starts_at_one_stays_bounded_falls_and_reaches_zero_at_the_life(case: Case) -> None:
    name, params, nominal, life, limit = case
    template = FAMILIES[name]()
    steps = 400
    k = template.calibrate_k(life, nominal, params, limit)
    dt = life / steps
    damage = np.zeros(1)
    health = [float(health_from_damage(damage, limit)[0])]
    for _ in range(steps + 10):
        damage = damage + dt * template.rate(damage, nominal, {**params, "k": k})
        health.append(float(health_from_damage(damage, limit)[0]))
    h = np.array(health)
    assert h[0] == 1.0 and np.all((h >= 0) & (h <= 1)) and np.all(np.diff(h) <= 1e-12)
    assert int(np.argmax(h <= 1e-9)) * dt == pytest.approx(life, rel=0.01)  # calibration reproduces T within 1%


def test_replacement_renews_the_part_and_other_actions_do_not() -> None:
    damage = np.array([0.2, 0.9])
    for template in (Accumulation(), Arrhenius(), BearingFatigue()):
        np.testing.assert_array_equal(template.apply_maintenance(damage, "replace", {}), [0.0, 0.0])
        np.testing.assert_array_equal(template.apply_maintenance(damage, "inspect", {}), damage)


def test_hotter_ages_faster_by_the_arrhenius_factor() -> None:
    params = {"activation_energy": 80e3, "temperature_nominal": 343.15, "k": 1.0}
    hot = float(Arrhenius().rate(np.zeros(1), {"temperature": np.array([353.15])}, params)[0])
    assert hot == pytest.approx(np.exp(80e3 / 8.314462618 * (1 / 343.15 - 1 / 353.15)))  # ~2.2x per +10 K
