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


def _shape(dt: FloatArray | float, distance: FloatArray, rate: FloatArray,
           s2: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray, np.ndarray]:
    t, a, r, v = (np.asarray(x, dtype=np.float64) for x in (dt, distance, rate, s2))
    t, a, r, v = np.broadcast_arrays(t, a, r, v)
    live = (a > 0.0) & (r > 0.0) & (v > 0.0)
    started = live & (t > 0.0)
    safe_a, safe_r, safe_v = np.where(live, a, 1.0), np.where(live, r, 1.0), np.where(live, v, 1.0)
    return np.where(started, t, 1.0), safe_a / safe_r, safe_a * safe_a / safe_v, live, started


def log_survival(dt: FloatArray | float, distance: FloatArray, rate: FloatArray, s2: FloatArray) -> FloatArray:
    """log P(T > dt): the part has not reached its limit within dt (dt per particle, or one for all)."""
    t, mu, lam, live, started = _shape(dt, distance, rate, s2)
    root = np.sqrt(lam / t)
    first = log_ndtr(-root * (t / mu - 1.0))  # log Phi(-A)
    second = 2.0 * lam / mu + log_ndtr(-root * (t / mu + 1.0))  # log(exp(2 lam / mu) Phi(-B))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = first + np.log1p(-np.exp(np.minimum(second - first, 0.0)))
    out = np.where(np.isnan(out), -np.inf, out)
    return np.asarray(np.where(started, out, 0.0), dtype=np.float64)


def log_density(dt: FloatArray | float, distance: FloatArray, rate: FloatArray, s2: FloatArray) -> FloatArray:
    """log f(dt): the part reaches its limit for the first time exactly dt after the previous row."""
    t, mu, lam, live, started = _shape(dt, distance, rate, s2)
    out = 0.5 * (np.log(lam) - LOG_2PI - 3.0 * np.log(t)) - lam * (t - mu) ** 2 / (2.0 * mu * mu * t)
    return np.asarray(np.where(started, out, -np.inf), dtype=np.float64)
