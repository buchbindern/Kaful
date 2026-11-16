"""Time-of-event forecasts: every sample moved forward at once (vectorized Monte Carlo), every sample kept.

A forecast event is the next time the binding reaches D_limit after t_issue (approved 2026-11-16): a renewed
binding's next renewal, any other binding's next crossing (again after a scheduled reset). Per sample:

- already triggered: at t_issue the binding is at or past D_limit and has no renewal rule (it has failed and
  waits for maintenance); its time of event is t_issue (RUL 0);
- censored: no crossing before the horizon (NaN);
- otherwise the crossing time.

Percentiles are over every sample, a censored one counting as later than the horizon, so a percentile past the
horizon is None. The run stops early once every sample of every event is decided.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kaful.arrays import BoolArray, FloatArray
from kaful.estimation.cloud import Cloud, advance, value
from kaful.estimation.dynamics import Dynamics

PERCENTILES = {"p05": 0.05, "p50": 0.50, "p95": 0.95}


@dataclass(frozen=True)
class EventSamples:
    event: str
    binding: str
    toe: FloatArray  # per sample; NaN = censored; t_issue = already triggered
    triggered: BoolArray  # per sample: already triggered at t_issue


@dataclass(frozen=True)
class Prediction:
    t_issue: float
    horizon: float
    events: tuple[EventSamples, ...]  # sorted by event
    params: dict[str, FloatArray]  # carried param_id -> value per sample


def forecast(dyn: Dynamics, cloud: Cloud, t_issue: float, horizon: float, dt: float, rho: float,
             rng: np.random.Generator, chunks: int = 50) -> Prediction:
    """Move a copy of `cloud` (equal weights) from t_issue to the horizon under `dyn`'s profile."""
    if not horizon > t_issue:
        raise ValueError(f"horizon {horizon} must be after t_issue {t_issue}")
    c = cloud.copy()
    triggered = {b.id: (~np.asarray(b.renews_at_limit, dtype=bool))
                 & (c.damage[b.id] >= value(c, dyn.fixed, b.limit_param)) for b in dyn.bindings}
    toe = {b.id: np.where(triggered[b.id], t_issue, np.nan) for b in dyn.bindings}
    edges = np.linspace(t_issue, horizon, chunks + 1)
    for t0, t1 in zip(edges[:-1], edges[1:]):
        if all(not np.isnan(v).any() for v in toe.values()):
            break
        first = advance(c, dyn, float(t0), float(t1), dt, rho, rng)
        for bid, when in first.items():
            open_ = np.isnan(toe[bid]) & ~np.isnan(when)
            toe[bid][open_] = when[open_]
    events = tuple(sorted((EventSamples(event=b.event, binding=b.id, toe=toe[b.id], triggered=triggered[b.id])
                           for b in dyn.bindings), key=lambda e: e.event))
    return Prediction(t_issue=t_issue, horizon=horizon, events=events,
                      params={k: v.copy() for k, v in sorted(cloud.params.items())})


@dataclass(frozen=True)
class Summary:
    n_samples: int
    n_hits: int  # crossing before the horizon, already triggered included
    n_censored: int
    n_triggered: int
    p_event_before_horizon: float
    percentiles: dict[str, float | None]


def summarize(e: EventSamples) -> Summary:
    n = e.toe.size
    hits = ~np.isnan(e.toe)
    ordered = np.where(hits, e.toe, np.inf)
    pct: dict[str, float | None] = {}
    for name, q in PERCENTILES.items():
        v = float(np.quantile(ordered, q, method="inverted_cdf"))
        pct[name] = v if np.isfinite(v) else None
    return Summary(n_samples=n, n_hits=int(hits.sum()), n_censored=int(n - hits.sum()),
                   n_triggered=int(e.triggered.sum()), p_event_before_horizon=float(hits.mean()), percentiles=pct)


def cloud_from(particles: Mapping[str, FloatArray], damage_states: Mapping[str, str],
               renewal_states: Mapping[str, str], carried: tuple[str, ...]) -> Cloud:
    """A Cloud from a StateBelief's particles (binding -> damage state name, binding -> renewal state name)."""
    n = next(iter(particles.values())).size
    return Cloud(damage={b: np.asarray(particles[s], dtype=np.float64).copy() for b, s in damage_states.items()},
                 renewals={b: np.asarray(particles[s], dtype=np.float64).copy() for b, s in renewal_states.items()},
                 params={p: np.asarray(particles[p], dtype=np.float64).copy() for p in carried},
                 log_w=np.zeros(n))
