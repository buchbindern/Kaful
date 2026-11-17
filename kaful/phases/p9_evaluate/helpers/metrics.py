"""Calibration metrics on sampled times of event (used by f_calibration). Pure functions, numpy only.

Samples are absolute times of event, +inf for a censored sample (no crossing before the horizon); the truth is the
true next crossing after the issue time, None when there is none before the horizon. Interval ends are sample
quantiles (inverted CDF), as the forecast's own percentiles.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kaful.artifacts.evaluation import Bounds, PairClass
from kaful.scoring.samples import interval


def wilson(k: float, n: float, z: float) -> Bounds | None:
    """Wilson score interval for k successes in n trials (k may be fractional when n is an effective size)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return Bounds(low=max(0.0, center - half), high=min(1.0, center + half))


def classify(samples: np.ndarray, truth: float | None, t_issue: float, qs: tuple[float, float],
             decidable: bool = True) -> PairClass:
    """Which coverage class one (unit, issue) falls in."""
    if not decidable:
        return "undecidable"
    if truth is not None and truth <= t_issue:
        return "already_triggered"
    lo, hi = interval(samples, qs)
    if truth is None:
        return "censored_covered" if not np.isfinite(hi) else "censored_missed"
    return "covered" if lo <= truth <= hi else "missed"


def alpha_mass(samples: np.ndarray, truth: float, t_issue: float, alpha: float) -> float:
    """The RUL mass within (1 +- alpha) x the true RUL (censored samples count outside)."""
    true_rul = truth - t_issue
    rul = samples - t_issue
    return float(np.mean(np.isfinite(rul) & (np.abs(rul - true_rul) <= alpha * true_rul)))


def alpha_lambda(samples: np.ndarray, truth: float, t_issue: float, alpha: float, beta: float) -> bool:
    """alpha-lambda accuracy: at least `beta` of the RUL mass within (1 +- alpha) x the true RUL."""
    return alpha_mass(samples, truth, t_issue, alpha) >= beta


def triggered(samples: np.ndarray, t_issue: float) -> bool:
    """The forecast says the part is already at its limit: at least half its samples at or before the issue time."""
    return bool(np.mean(samples <= t_issue) >= 0.5)


@dataclass(frozen=True)
class Horizon:
    value: float
    share: float


def prognostic_horizon(issues: Sequence[tuple[float, float | None, bool]]) -> Horizon | None:
    """(t_issue, truth, alpha-lambda passed) in time order -> PH for the first true crossing ahead.

    Only issues aiming at that same crossing count. PH = crossing - the earliest issue time from which every later
    such issue passes; 0 if the last one fails. share = PH / (crossing - first such issue time). None: no crossing
    ahead at any issue."""
    ahead = [(t, truth, ok) for t, truth, ok in issues if truth is not None and truth > t]
    if not ahead:
        return None
    eol = ahead[0][1]
    assert eol is not None
    same = [(t, ok) for t, truth, ok in ahead if truth == eol]
    start = None
    for t, ok in reversed(same):
        if not ok:
            break
        start = t
    span = eol - same[0][0]
    value = 0.0 if start is None else eol - start
    return Horizon(value=value, share=value / span if span > 0 else 0.0)


def median_error(samples: np.ndarray, truth: float, t_issue: float) -> tuple[float, float] | None:
    """(median RUL - true RUL, that / true RUL); None when the median is censored."""
    med = float(np.quantile(samples, 0.5, method="inverted_cdf"))
    if not np.isfinite(med):
        return None
    err = (med - t_issue) - (truth - t_issue)
    return err, err / (truth - t_issue)
