"""Probability distributions for parameter priors and posteriors.

Every value in a ParamSet is a distribution, not a bare number. `Delta` (a point
mass) exists only for definitional values; ParamSet restricts where it may be used.
Sampling uses numpy; quantiles use the stdlib `statistics.NormalDist`. The median of
any distribution is `quantile(0.5)`.
"""

import math
from statistics import NormalDist
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from kaful.arrays import FloatArray
from kaful.artifacts.base import StrictModel

_STANDARD_NORMAL = NormalDist()


def _check_probability(p: float) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")


class Delta(StrictModel):
    """Point mass at `value`. Only for definitional values (see ParamSet rules)."""

    family: Literal["delta"] = "delta"
    value: float

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return np.full(n, self.value, dtype=np.float64)

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return self.value


class Normal(StrictModel):
    """Normal distribution with mean and standard deviation."""

    family: Literal["normal"] = "normal"
    mean: float
    sd: float = Field(gt=0.0)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.normal(self.mean, self.sd, size=n).astype(np.float64)

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return NormalDist(self.mean, self.sd).inv_cdf(p)


class TruncatedNormal(StrictModel):
    """Normal(mean, sd) restricted to [low, high]; sampled by inverse CDF."""

    family: Literal["truncated_normal"] = "truncated_normal"
    mean: float
    sd: float = Field(gt=0.0)
    low: float
    high: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "TruncatedNormal":
        if not self.low < self.high:
            raise ValueError("low must be < high")
        if self._mass() <= 0.0:
            raise ValueError("[low, high] holds no probability mass for this mean and sd")
        return self

    def _mass(self) -> float:
        base = NormalDist(self.mean, self.sd)
        return base.cdf(self.high) - base.cdf(self.low)

    def _inv(self, u: float) -> float:
        base = NormalDist(self.mean, self.sd)
        x = base.inv_cdf(base.cdf(self.low) + u * self._mass())
        return min(max(x, self.low), self.high)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return np.array([self._inv(float(u)) for u in rng.uniform(size=n)], dtype=np.float64)

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return self._inv(p)


class Lognormal(StrictModel):
    """Lognormal given by its median and the standard deviation of log(x)."""

    family: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0.0)
    sigma_log: float = Field(gt=0.0)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return (self.median * np.exp(self.sigma_log * rng.standard_normal(n))).astype(
            np.float64
        )

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return self.median * math.exp(self.sigma_log * _STANDARD_NORMAL.inv_cdf(p))


class Uniform(StrictModel):
    """Uniform on [low, high]."""

    family: Literal["uniform"] = "uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _check_bounds(self) -> "Uniform":
        if not self.low < self.high:
            raise ValueError("low must be < high")
        return self

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.uniform(self.low, self.high, size=n).astype(np.float64)

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return self.low + p * (self.high - self.low)


class Empirical(StrictModel):
    """Equally weighted samples, e.g. a particle-filter posterior. Sampling resamples."""

    family: Literal["empirical"] = "empirical"
    samples: tuple[float, ...] = Field(min_length=1)

    def sample(self, rng: np.random.Generator, n: int) -> FloatArray:
        return rng.choice(np.asarray(self.samples, dtype=np.float64), size=n, replace=True)

    def quantile(self, p: float) -> float:
        _check_probability(p)
        return float(np.quantile(np.asarray(self.samples, dtype=np.float64), p))



Distribution = Annotated[
    Delta | Normal | TruncatedNormal | Lognormal | Uniform | Empirical,
    Field(discriminator="family"),
]
