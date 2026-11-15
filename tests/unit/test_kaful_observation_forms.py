"""Observation forms (M12): linear and threshold_event, with property tests."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kaful.observation.forms import FORMS, Linear, ThresholdEvent
from kaful.observation.protocol import DamageView, ObservationForm

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)
positive = st.floats(min_value=1e-3, max_value=1e2, allow_nan=False)
fractions = st.lists(st.floats(min_value=0.0, max_value=3.0, allow_nan=False), min_size=1, max_size=20)


def _view(fraction: list[float], crossed: list[bool] | None = None) -> DamageView:
    f = np.asarray(fraction, dtype=np.float64)
    c = np.zeros(f.shape, dtype=bool) if crossed is None else np.asarray(crossed, dtype=bool)
    return DamageView(fraction=f, crossed=c)


def test_forms_implement_the_protocol() -> None:
    assert sorted(FORMS) == ["linear", "threshold_event"]
    for cls in FORMS.values():
        assert isinstance(cls(), ObservationForm)


@settings(max_examples=60, deadline=None)
@given(y0=finite, delta=finite, sd=positive, fraction=fractions)
def test_linear_runs_from_healthy_to_limit_and_saturates(y0: float, delta: float, sd: float,
                                                         fraction: list[float]) -> None:
    p = {"y_healthy": y0, "delta": delta, "noise_sd": sd}
    y = Linear().expected(_view(fraction), p)
    f = np.clip(np.asarray(fraction), 0.0, 1.0)
    np.testing.assert_allclose(y, y0 + delta * f, rtol=1e-12, atol=1e-9)
    assert Linear().expected(_view([0.0]), p)[0] == pytest.approx(y0)
    np.testing.assert_allclose(Linear().expected(_view([1.0, 2.5]), p), [y0 + delta] * 2, atol=1e-9)


@settings(max_examples=40, deadline=None)
@given(y0=finite, delta=st.floats(min_value=1e-3, max_value=1e3), sd=positive,
       a=st.floats(min_value=0.0, max_value=1.0), b=st.floats(min_value=0.0, max_value=1.0))
def test_linear_is_monotone_in_damage_with_the_sign_of_delta(y0: float, delta: float, sd: float, a: float,
                                                             b: float) -> None:
    lo, hi = sorted((a, b))
    up = {"y_healthy": y0, "delta": delta, "noise_sd": sd}
    down = {**up, "delta": -delta}
    y_up = Linear().expected(_view([lo, hi]), up)
    y_down = Linear().expected(_view([lo, hi]), down)
    assert y_up[0] <= y_up[1] + 1e-9 and y_down[0] >= y_down[1] - 1e-9


@settings(max_examples=40, deadline=None)
@given(y0=finite, delta=finite, sd=positive, fraction=fractions, y=finite)
def test_linear_log_likelihood_is_the_gaussian_density_and_peaks_at_the_mean(
        y0: float, delta: float, sd: float, fraction: list[float], y: float) -> None:
    p = {"y_healthy": y0, "delta": delta, "noise_sd": sd}
    view = _view(fraction)
    mean = Linear().expected(view, p)
    ll = Linear().log_likelihood(np.full(mean.shape, y), view, p)
    want = [-0.5 * ((y - m) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi) for m in mean]
    np.testing.assert_allclose(ll, want, rtol=1e-9, atol=1e-9)
    assert np.all(Linear().log_likelihood(mean, view, p) >= ll - 1e-12)


def test_linear_sample_has_the_declared_noise() -> None:
    p = {"y_healthy": 2.0, "delta": 3.0, "noise_sd": 0.5}
    view = _view([0.5] * 20_000)
    y = Linear().sample(view, p, np.random.default_rng(1))
    assert abs(float(y.mean()) - 3.5) < 0.02 and abs(float(y.std()) - 0.5) < 0.02


def test_linear_sample_is_seeded() -> None:
    p = {"y_healthy": 0.0, "delta": 1.0, "noise_sd": 0.1}
    view = _view([0.1, 0.2, 0.3])
    a = Linear().sample(view, p, np.random.default_rng(5))
    b = Linear().sample(view, p, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_linear_refuses_a_non_positive_noise() -> None:
    with pytest.raises(ValueError, match="noise_sd"):
        Linear().sample(_view([0.0]), {"y_healthy": 0.0, "delta": 1.0, "noise_sd": 0.0}, np.random.default_rng(0))


def test_threshold_event_fires_exactly_where_damage_crossed() -> None:
    view = _view([0.2, 0.99, 0.01, 0.5], [False, False, True, False])
    out = ThresholdEvent().sample(view, {}, np.random.default_rng(0))
    assert out.tolist() == [0.0, 0.0, 1.0, 0.0]
    ll = ThresholdEvent().log_likelihood(np.array([0.0, 0.0, 1.0, 0.0]), view, {})
    assert ll.tolist() == [0.0, 0.0, 0.0, 0.0]
    miss = ThresholdEvent().log_likelihood(np.array([0.0, 1.0, 0.0, 0.0]), view, {})
    assert miss[1] == -math.inf and miss[2] == -math.inf and miss[0] == 0.0


def test_damage_view_shapes_must_match() -> None:
    with pytest.raises(ValueError, match="same shape"):
        DamageView(fraction=np.zeros(3), crossed=np.zeros(2, dtype=bool))
