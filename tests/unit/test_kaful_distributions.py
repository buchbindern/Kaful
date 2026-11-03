"""Distributions: validation, seeded sampling and quantiles."""

import math
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from kaful.artifacts.distributions import (
    Delta,
    Distribution,
    Empirical,
    Lognormal,
    Normal,
    TruncatedNormal,
    Uniform,
)

ADAPTER: TypeAdapter[Any] = TypeAdapter(Distribution)
EXAMPLES = [
    Delta(value=2.0),
    Normal(mean=1.0, sd=0.5),
    TruncatedNormal(mean=0.0, sd=1.0, low=-0.5, high=2.0),
    Lognormal(median=3.0, sigma_log=0.4),
    Uniform(low=1.0, high=4.0),
    Empirical(samples=(1.0, 2.0, 3.0, 10.0)),
]
N = 2000


@pytest.mark.parametrize("dist", EXAMPLES, ids=lambda d: d.family)
def test_round_trip_through_discriminated_union(dist: Any) -> None:
    assert ADAPTER.validate_json(ADAPTER.dump_json(dist)) == dist


@pytest.mark.parametrize("dist", EXAMPLES, ids=lambda d: d.family)
def test_sampling_is_seeded_and_shaped(dist: Any) -> None:
    a = dist.sample(np.random.default_rng(1), N)
    b = dist.sample(np.random.default_rng(1), N)
    assert a.shape == (N,) and a.dtype == np.float64
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dist", EXAMPLES, ids=lambda d: d.family)
def test_quantiles_are_monotone_and_match_samples(dist: Any) -> None:
    qs = [dist.quantile(p) for p in (0.05, 0.5, 0.95)]
    assert qs == sorted(qs)
    if not isinstance(dist, (Delta, Empirical)):
        samples = dist.sample(np.random.default_rng(0), 20000)
        assert abs(float(np.mean(samples <= qs[1])) - 0.5) < 0.02


def test_lognormal_median_and_spread() -> None:
    dist = Lognormal(median=3.0, sigma_log=0.4)
    assert dist.quantile(0.5) == pytest.approx(3.0)
    assert dist.quantile(0.8413447) == pytest.approx(3.0 * math.exp(0.4), rel=1e-6)


def test_truncated_normal_stays_in_bounds() -> None:
    dist = TruncatedNormal(mean=0.0, sd=1.0, low=-0.5, high=2.0)
    samples = dist.sample(np.random.default_rng(3), N)
    assert samples.min() >= -0.5 and samples.max() <= 2.0


@pytest.mark.parametrize("data", [
    {"family": "normal", "mean": 0.0, "sd": 0.0},
    {"family": "lognormal", "median": -1.0, "sigma_log": 0.3},
    {"family": "uniform", "low": 2.0, "high": 1.0},
    {"family": "truncated_normal", "mean": 0.0, "sd": 1.0, "low": 50.0, "high": 60.0},
    {"family": "empirical", "samples": []},
    {"family": "point", "value": 1.0},
])
def test_invalid_distributions_are_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ADAPTER.validate_python(data)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1])
def test_quantile_rejects_bad_probabilities(p: float) -> None:
    with pytest.raises(ValueError):
        Normal(mean=0.0, sd=1.0).quantile(p)
