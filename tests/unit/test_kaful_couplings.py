"""Property tests for the coupling forms (M13b, hypothesis): the protocol, m >= 0, m monotone in the source's damage
fraction, x clipped to [0, 1], m averaging 1 over a source cycle for every r (so the target's k keeps its meaning),
the neutral value giving m = 1, and the closed-form crossing time agreeing with a fine numerical integration.
Derandomized, so a run is the same everywhere."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kaful.templates.couplings import COUPLINGS, CouplingForm, HardnessPassthrough

PROPERTY = settings(derandomize=True, max_examples=200, deadline=None)
FORM = HardnessPassthrough()
R = st.floats(0.0, 1.0, allow_nan=False)
X = st.floats(-0.5, 2.0, allow_nan=False)


def test_each_form_satisfies_the_protocol() -> None:
    for form in COUPLINGS.values():
        assert isinstance(form(), CouplingForm)
        assert set(form.neutral) == {s.name for s in form.params} == set(form.bounds)


@PROPERTY
@given(r=R, x=X)
def test_multiplier_is_non_negative_and_bounded(r: float, x: float) -> None:
    m = float(FORM.multiplier(x, {"residual_fraction": r}))
    assert 0.0 <= m <= 2.0 / (1.0 + r) + 1e-12


@PROPERTY
@given(r=R, a=X, b=X)
def test_multiplier_is_monotone_in_the_fraction(r: float, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    p = {"residual_fraction": r}
    assert float(FORM.multiplier(lo, p)) <= float(FORM.multiplier(hi, p)) + 1e-12


@PROPERTY
@given(r=R, x=X)
def test_fraction_is_clipped(r: float, x: float) -> None:
    p = {"residual_fraction": r}
    assert float(FORM.multiplier(x, p)) == float(FORM.multiplier(min(max(x, 0.0), 1.0), p))


@PROPERTY
@given(r=R)
def test_multiplier_averages_one_over_a_cycle(r: float) -> None:
    p = {"residual_fraction": r}
    assert FORM.cycle_integral(1.0, p) == pytest.approx(1.0, abs=1e-12)
    x = (np.arange(200_000) + 0.5) / 200_000
    assert float(FORM.multiplier(x, p).mean()) == pytest.approx(1.0, abs=1e-9)


@PROPERTY
@given(r=R, phi=st.floats(0.0, 1.0))
def test_cycle_integral_matches_the_multiplier(r: float, phi: float) -> None:
    p = {"residual_fraction": r}
    x = (np.arange(20_000) + 0.5) / 20_000 * phi
    assert FORM.cycle_integral(phi, p) == pytest.approx(float(FORM.multiplier(x, p).mean()) * phi, abs=1e-8)


@PROPERTY
@given(x=X)
def test_neutral_value_is_no_coupling(x: float) -> None:
    assert float(FORM.multiplier(x, FORM.neutral)) == 1.0


def test_multiplier_vectorizes_over_particles() -> None:
    r = np.array([0.0, 0.5, 1.0])
    m = FORM.multiplier(np.array([0.0, 0.5, 1.0]), {"residual_fraction": r})
    assert m == pytest.approx([0.0, 1.0, 1.0])
    assert FORM.multiplier(1.0, {"residual_fraction": r}) == pytest.approx([2.0, 4 / 3, 1.0])


def _integrated(target: float, source: float, r: float, renews: bool, n: int = 400_000) -> float:
    """The first time the integral of m(x(t)) dt reaches `target`, on a fine grid."""
    horizon = 3.0 * max(target, source) + target * 3.0
    dt = horizon / n
    t = (np.arange(n) + 0.5) * dt
    x = np.mod(t / source, 1.0) if renews else np.minimum(t / source, 1.0)
    done = np.cumsum(FORM.multiplier(x, {"residual_fraction": r}) * dt)
    return float((np.argmax(done >= target) + 1) * dt)


@PROPERTY
@given(r=R, target=st.floats(10.0, 5e4), source=st.floats(10.0, 5e4), renews=st.booleans())
def test_crossing_time_matches_numerical_integration(r: float, target: float, source: float, renews: bool) -> None:
    expected = FORM.crossing_time(target, source, {"residual_fraction": r}, renews)
    horizon = 3.0 * max(target, source) + target * 3.0
    jumps = target / source + 2.0  # the grid misplaces each sawtooth jump by up to one cell
    assert expected == pytest.approx(_integrated(target, source, r, renews), abs=2.0 * jumps * horizon / 400_000)


@PROPERTY
@given(target=st.floats(1.0, 1e5), source=st.floats(1.0, 1e5), renews=st.booleans())
def test_uncoupled_crossing_time_is_the_target_life(target: float, source: float, renews: bool) -> None:
    assert FORM.crossing_time(target, source, FORM.neutral, renews) == pytest.approx(target, rel=1e-12)


def test_a_short_target_life_early_in_a_fresh_cycle_is_slower() -> None:
    # eversys at prior medians: steam boiler 1,158 products, filter 13,907, r = 0.5 -> m starts at 2/3
    t = FORM.crossing_time(1158.0, 13907.0, {"residual_fraction": 0.5}, renews=True)
    assert 1158.0 * 1.3 < t < 1158.0 * 1.5


@PROPERTY
@given(r=R, factor=st.floats(1.0, 5.0))
def test_a_delayed_renewal_raises_the_long_run_mean_multiplier_unless_uncoupled(r: float, factor: float) -> None:
    p = {"residual_fraction": r}
    mean = FORM.delayed_mean(p, factor)
    assert 1.0 - 1e-12 <= mean <= float(FORM.multiplier(1.0, p)) + 1e-12
    assert FORM.delayed_mean(p, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert FORM.delayed_mean(FORM.neutral, factor) == pytest.approx(1.0, abs=1e-12)
    n = 100_000  # numerically: x = t / T for t in [0, factor T), clipped at 1
    x = np.minimum((np.arange(n) + 0.5) / n * factor, 1.0)
    assert mean == pytest.approx(float(FORM.multiplier(x, p).mean()), abs=1e-6)
