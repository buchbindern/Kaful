"""First passage of damage to its limit under the estimators' process noise (M13): what a threshold-event channel
tells a particle filter.

Between rows, D follows a Wiener process with drift r = dD/dt and variance rate s2 (cloud.py: s2 = rho^2 * D_limit *
r). Starting a distance a = D_limit - D below the limit, the time T to reach it is inverse Gaussian with mean
mu = a / r and shape lambda = a^2 / s2. With a new part (a = D_limit), T's coefficient of variation is rho.

- A silent row on a monitored event channel says the part has not reached its limit since the previous row:
  log S(dt) = log P(T > dt).
- A report says it reached it now (the alarm fires at the crossing, so the report time is the crossing time):
  log f(dt), the density. The report's clock resolution is a constant factor that cancels across particles.

A part already at or past its limit (a <= 0, a failed part with no renewal) or not degrading (r <= 0) cannot
reach it for the first time: S = 1, f = 0. Everything is in log space and broadcasts over particles.
"""

import numpy as np
from scipy.special import log_ndtr  # type: ignore[import-untyped]

from kaful.arrays import FloatArray

LOG_2PI = float(np.log(2.0 * np.pi))


def _shape(distance: FloatArray, rate: FloatArray, s2: FloatArray) -> tuple[FloatArray, FloatArray, np.ndarray]:
    a, r, v = (np.asarray(x, dtype=np.float64) for x in (distance, rate, s2))
    a, r, v = np.broadcast_arrays(a, r, v)
    live = (a > 0.0) & (r > 0.0) & (v > 0.0)
    safe_a, safe_r, safe_v = np.where(live, a, 1.0), np.where(live, r, 1.0), np.where(live, v, 1.0)
    return safe_a / safe_r, safe_a * safe_a / safe_v, live


def log_survival(dt: float, distance: FloatArray, rate: FloatArray, s2: FloatArray) -> FloatArray:
    """log P(T > dt): the part has not reached its limit within dt."""
    mu, lam, live = _shape(distance, rate, s2)
    if dt <= 0.0:
        return np.zeros(live.shape)
    root = np.sqrt(lam / dt)
    first = log_ndtr(-root * (dt / mu - 1.0))  # log Phi(-A)
    second = 2.0 * lam / mu + log_ndtr(-root * (dt / mu + 1.0))  # log(exp(2 lam / mu) Phi(-B))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = first + np.log1p(-np.exp(np.minimum(second - first, 0.0)))
    out = np.where(np.isnan(out), -np.inf, out)
    return np.asarray(np.where(live, out, 0.0), dtype=np.float64)


def log_density(dt: float, distance: FloatArray, rate: FloatArray, s2: FloatArray) -> FloatArray:
    """log f(dt): the part reaches its limit for the first time exactly dt after the previous row."""
    mu, lam, live = _shape(distance, rate, s2)
    if dt <= 0.0:
        return np.full(live.shape, -np.inf)
    out = 0.5 * (np.log(lam) - LOG_2PI - 3.0 * np.log(dt)) - lam * (dt - mu) ** 2 / (2.0 * mu * mu * dt)
    return np.asarray(np.where(live, out, -np.inf), dtype=np.float64)
