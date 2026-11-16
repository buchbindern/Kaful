"""Observation forms (M12): linear and threshold_event, with property tests."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kaful.observation.forms import FORMS, Linear, ThresholdEvent, linear_channel
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
@given(y0=finite, y1=finite, sd=positive, fraction=fractions)
def test_linear_runs_from_healthy_to_limit_and_saturates(y0: float, y1: float, sd: float,
                                                         fraction: list[float]) -> None:
    p = {"y_healthy": y0, "y_limit": y1, "noise_sd": sd}
    y = Linear().expected(_view(fraction), p)
    f = np.clip(np.asarray(fraction), 0.0, 1.0)
    np.testing.assert_allclose(y, y0 + (y1 - y0) * f, rtol=1e-12, atol=1e-9)
    assert Linear().expected(_view([0.0]), p)[0] == pytest.approx(y0)
    np.testing.assert_allclose(Linear().expected(_view([1.0, 2.5]), p), [y1] * 2, rtol=1e-12, atol=1e-9)


@settings(max_examples=40, deadline=None)
@given(y0=finite, rise=st.floats(min_value=1e-3, max_value=1e3), sd=positive,
       a=st.floats(min_value=0.0, max_value=1.0), b=st.floats(min_value=0.0, max_value=1.0))
def test_linear_moves_monotonically_toward_y_limit(y0: float, rise: float, sd: float, a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    up = Linear().expected(_view([lo, hi]), {"y_healthy": y0, "y_limit": y0 + rise, "noise_sd": sd})
    down = Linear().expected(_view([lo, hi]), {"y_healthy": y0, "y_limit": y0 - rise, "noise_sd": sd})
    assert up[0] <= up[1] + 1e-9 and down[0] >= down[1] - 1e-9


@settings(max_examples=40, deadline=None)
@given(y0=finite, y1=finite, sd=positive, fraction=fractions, y=finite)
def test_linear_log_likelihood_is_the_gaussian_density_and_peaks_at_the_mean(
        y0: float, y1: float, sd: float, fraction: list[float], y: float) -> None:
    p = {"y_healthy": y0, "y_limit": y1, "noise_sd": sd}
    view = _view(fraction)
    mean = Linear().expected(view, p)
    ll = Linear().log_likelihood(np.full(mean.shape, y), view, p)
    want = [-0.5 * ((y - m) / sd) ** 2 - math.log(sd) - 0.5 * math.log(2 * math.pi) for m in mean]
    np.testing.assert_allclose(ll, want, rtol=1e-9, atol=1e-9)
    assert np.all(Linear().log_likelihood(mean, view, p) >= ll - 1e-12)


@settings(max_examples=40, deadline=None)
@given(y0=st.floats(min_value=0.0, max_value=10.0), l1=st.floats(min_value=0.0, max_value=10.0),
       l2=st.floats(min_value=0.0, max_value=10.0), f1=fractions, f2=fractions)
def test_a_shared_channel_adds_each_bindings_change(y0: float, l1: float, l2: float, f1: list[float],
                                                    f2: list[float]) -> None:
    n = min(len(f1), len(f2))
    v1, v2 = _view(f1[:n]), _view(f2[:n])
    both = linear_channel(y0, [(v1, l1), (v2, l2)])
    want = y0 + (l1 - y0) * np.clip(v1.fraction, 0, 1) + (l2 - y0) * np.clip(v2.fraction, 0, 1)
    np.testing.assert_allclose(both, want, rtol=1e-12, atol=1e-9)
    healthy = linear_channel(y0, [(_view([0.0] * n), l1), (_view([0.0] * n), l2)])
    np.testing.assert_allclose(healthy, [y0] * n)


def test_a_linear_channel_needs_a_binding() -> None:
    with pytest.raises(ValueError, match="at least one"):
        linear_channel(1.0, [])


def test_linear_sample_has_the_declared_noise() -> None:
    p = {"y_healthy": 2.0, "y_limit": 5.0, "noise_sd": 0.5}
    view = _view([0.5] * 20_000)
    y = Linear().sample(view, p, np.random.default_rng(1))
    assert abs(float(y.mean()) - 3.5) < 0.02 and abs(float(y.std()) - 0.5) < 0.02


def test_linear_sample_is_seeded() -> None:
    p = {"y_healthy": 0.0, "y_limit": 1.0, "noise_sd": 0.1}
    view = _view([0.1, 0.2, 0.3])
    a = Linear().sample(view, p, np.random.default_rng(5))
    b = Linear().sample(view, p, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_linear_refuses_a_non_positive_noise() -> None:
    with pytest.raises(ValueError, match="noise_sd"):
        Linear().sample(_view([0.0]), {"y_healthy": 0.0, "y_limit": 1.0, "noise_sd": 0.0}, np.random.default_rng(0))


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
