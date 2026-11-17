"""A forecast issue's sampled times of event (no truth here): censored samples are +inf, and the 5-95% interval is
taken as sample quantiles (inverted CDF, the forecast's own percentiles), never interpolated through infinity."""

from collections.abc import Mapping, Sequence

import numpy as np

INTERVAL = (0.05, 0.95)


def toe_samples(rows: Sequence[Mapping[str, str]], event: str) -> np.ndarray:
    """Each sample's time of event (+inf: censored, no crossing before the horizon)."""
    return np.array([float(r[f"{event}.toe"]) if r[f"{event}.toe"] else np.inf for r in rows])


def interval(samples: np.ndarray, qs: tuple[float, float] = INTERVAL) -> tuple[float, float]:
    lo, hi = (float(np.quantile(samples, q, method="inverted_cdf")) for q in qs)
    return lo, hi


def covers(samples: np.ndarray, truth: float) -> bool:
    """Whether the 5-95% interval holds the truth; samples may be +inf (censored)."""
    lo, hi = interval(samples)
    return bool(lo <= truth <= hi)
