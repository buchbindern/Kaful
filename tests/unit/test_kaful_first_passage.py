"""First passage of D to its limit (M13): the inverse Gaussian survival and density against simulated paths."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats  # type: ignore[import-untyped]

from kaful.observation.first_passage import log_density, log_survival


def _paths(a: float, r: float, s2: float, horizon: float, n: int = 40_000, steps: int = 4000) -> np.ndarray:
    """First passage times of simulated Brownian paths with drift r from a distance a (NaN: not before horizon)."""
    rng = np.random.default_rng(11)
    dt = horizon / steps
    d, first = np.zeros(n), np.full(n, np.nan)
    for i in range(steps):
        d += r * dt + np.sqrt(s2 * dt) * rng.standard_normal(n)
        hit = (d >= a) & np.isnan(first)
        first[hit] = (i + 1) * dt
    return first


@pytest.mark.parametrize("rho", [0.05, 0.1, 0.3])
def test_survival_matches_simulated_paths(rho: float) -> None:
    a, r = 1.0, 0.01
    s2 = rho * rho * a * r
    first = _paths(a, r, s2, horizon=200.0)
    for t in (70.0, 90.0, 100.0, 110.0, 130.0):
        empirical = float(np.mean(np.isnan(first) | (first > t)))
        assert np.exp(log_survival(t, np.array([a]), np.array([r]), np.array([s2])))[0] == pytest.approx(
            empirical, abs=0.01 + 0.02 * rho)  # the discrete-time simulation crosses slightly late


@settings(max_examples=40, deadline=None)
@given(st.floats(min_value=0.05, max_value=5.0), st.floats(min_value=1e-3, max_value=1.0),
       st.floats(min_value=0.02, max_value=0.5), st.floats(min_value=0.05, max_value=3.0))
def test_survival_and_density_are_the_inverse_gaussian(a: float, r: float, rho: float, frac: float) -> None:
    s2 = rho * rho * a * r
    mu, lam = a / r, a * a / s2
    t = frac * mu
    ig = stats.invgauss(mu / lam, scale=lam)
    assert np.exp(log_survival(t, np.array([a]), np.array([r]), np.array([s2])))[0] == pytest.approx(
        ig.sf(t), abs=1e-9, rel=1e-6)
    assert log_density(t, np.array([a]), np.array([r]), np.array([s2]))[0] == pytest.approx(ig.logpdf(t), abs=1e-6)


def test_a_new_part_has_coefficient_of_variation_rho() -> None:
    a, r, rho = 2.0, 0.05, 0.1
    s2 = rho * rho * a * r
    grid = np.linspace(1.0, 120.0, 2001)
    pdf = np.exp([log_density(float(x), np.array([a]), np.array([r]), np.array([s2]))[0] for x in grid])
    mean = np.trapezoid(grid * pdf, grid)
    sd = np.sqrt(np.trapezoid((grid - mean) ** 2 * pdf, grid))
    assert mean == pytest.approx(a / r, rel=1e-3) and sd / mean == pytest.approx(rho, rel=1e-2)


def test_a_failed_or_idle_part_cannot_reach_its_limit_for_the_first_time() -> None:
    distance, rate, s2 = np.array([0.0, -0.2, 0.5]), np.array([0.1, 0.1, 0.0]), np.array([1e-3, 1e-3, 1e-3])
    assert log_survival(10.0, distance, rate, s2).tolist() == [0.0, 0.0, 0.0]
    assert np.all(log_density(10.0, distance, rate, s2) == -np.inf)


def test_far_tails_stay_finite_or_minus_infinity_without_warnings() -> None:
    with np.errstate(all="raise"):
        s = log_survival(1e6, np.array([1.0]), np.array([0.01]), np.array([1e-6]))
        f = log_density(1e-3, np.array([1.0]), np.array([0.01]), np.array([1e-6]))
    assert s[0] < -1e3 and (f[0] == -np.inf or f[0] < -1e3)
