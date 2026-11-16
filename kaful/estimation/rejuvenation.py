"""Liu-West rejuvenation of the carried parameters, so parameter particles do not collapse onto a few values.

Static parameters get no dynamics, so resampling alone would leave ever fewer distinct values. At each resample,
the parameters move to an unconstrained scale (log for a positive prior, logit within a bounded prior's support)
and each resampled particle i becomes

    theta_i' = a * theta_i + (1 - a) * mean + h * L * eps,   a = (3 delta - 1) / (2 delta),   h^2 = 1 - a^2,

where mean and L L^T = covariance are the weighted moments before resampling (Liu and West 2001). The shrinkage
toward the mean cancels the jitter's added variance, so the mean and covariance are kept, unlike plain roughening,
which adds variance at every resample. The jitter is a fraction h of the current posterior spread, so it shrinks as
the posterior contracts; it runs only at a resample, so a group that learns nothing gets none.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Delta, Empirical, Lognormal, Normal, TruncatedNormal, Uniform

EDGE = 1e-12  # keeps a value at a bound inside the logit's domain


@dataclass(frozen=True)
class Transform:
    """To and from an unconstrained scale for one parameter's support."""

    kind: str  # "log", "logit" or "identity"
    low: float = 0.0
    high: float = 1.0

    def forward(self, x: FloatArray) -> FloatArray:
        if self.kind == "log":
            return np.asarray(np.log(x), dtype=np.float64)
        if self.kind == "logit":
            p = np.clip((x - self.low) / (self.high - self.low), EDGE, 1.0 - EDGE)
            return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    def backward(self, z: FloatArray) -> FloatArray:
        if self.kind == "log":
            return np.asarray(np.exp(z), dtype=np.float64)
        if self.kind == "logit":
            return np.asarray(self.low + (self.high - self.low) / (1.0 + np.exp(-z)), dtype=np.float64)
        return np.asarray(z, dtype=np.float64)


def transform_for(prior: Delta | Normal | TruncatedNormal | Lognormal | Uniform | Empirical) -> Transform:
    if isinstance(prior, Lognormal):
        return Transform("log")
    if isinstance(prior, (Uniform, TruncatedNormal)):
        return Transform("logit", prior.low, prior.high)
    return Transform("identity")


def shrinkage(discount: float) -> tuple[float, float]:
    """(a, h) for a Liu-West discount delta."""
    a = (3.0 * discount - 1.0) / (2.0 * discount)
    return a, float(np.sqrt(1.0 - a * a))


def _sqrt_cov(cov: FloatArray) -> FloatArray:
    """A square root of a covariance that may be singular (e.g. one parameter collapsed)."""
    values, vectors = np.linalg.eigh(cov)
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)), dtype=np.float64)


def liu_west(params: Mapping[str, FloatArray], w: FloatArray, idx: np.ndarray, transforms: Mapping[str, Transform],
             discount: float, rng: np.random.Generator) -> dict[str, FloatArray]:
    """Resample the parameters at `idx` (drawn from weights `w`) and rejuvenate them; names in sorted order."""
    names = sorted(params)
    if not names:
        return {}
    z = np.stack([transforms[k].forward(params[k]) for k in names])  # (d, n)
    mean = z @ w
    centred = z - mean[:, None]
    cov = (centred * w) @ centred.T
    a, h = shrinkage(discount)
    eps = rng.standard_normal((len(names), idx.size))
    moved = a * z[:, idx] + (1.0 - a) * mean[:, None] + h * (_sqrt_cov(cov) @ eps)
    return {k: transforms[k].backward(moved[i]) for i, k in enumerate(names)}
