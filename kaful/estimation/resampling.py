"""Weights, effective sample size and systematic resampling."""

import numpy as np

from kaful.arrays import FloatArray


class DegenerateWeights(RuntimeError):
    """Every particle has zero likelihood: the data cannot come from this model and prior."""


def normalized(log_w: FloatArray) -> FloatArray:
    """Weights summing to 1 from log weights. Raises DegenerateWeights when all are -inf or any is NaN."""
    if np.any(np.isnan(log_w)):
        raise DegenerateWeights("a log weight is NaN")
    top = float(np.max(log_w))
    if not np.isfinite(top):
        raise DegenerateWeights("every particle has zero likelihood")
    w = np.exp(log_w - top)
    return np.asarray(w / w.sum(), dtype=np.float64)


def ess(w: FloatArray) -> float:
    """Effective sample size 1 / sum(w^2) of normalized weights."""
    return float(1.0 / np.sum(w * w))


def systematic(w: FloatArray, u: float) -> np.ndarray:
    """Systematic resampling: n indices from one offset u in [0, 1); low variance, in index order."""
    n = w.size
    if not 0.0 <= u < 1.0:
        raise ValueError(f"offset u must be in [0, 1), got {u}")
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0  # guard against rounding
    return np.asarray(np.searchsorted(cumulative, (np.arange(n) + u) / n, side="right"), dtype=np.int64)
