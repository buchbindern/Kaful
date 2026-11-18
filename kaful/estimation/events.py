"""Threshold-event channels in the particle filter: exact crossing observations (M13, checkpoint 2).

An alarm-observed binding (a threshold_event observation on a monitored channel) is filtered through its crossing
times, which carry its process noise exactly. Each particle keeps an anchor per such binding: the time t_a and
damage x_a of the binding's last reset (a scheduled maintenance action, a renewal at a report, or t = 0). Between
anchors its damage follows the mean path x_a + r (t - t_a) (cloud.advance with `free`), and the time T from the
anchor to D_limit is inverse Gaussian (kaful.observation.first_passage) with distance D_limit - x_a, drift r and
variance rate s2 = rho^2 * D_limit * r. Then, for a row at t after a row at t0:

- silence on the channel: log P(T > t - t_a | T > t0 - t_a) = log S(t - t_a) - log S(t0 - t_a);
- a report: the alarm fires at the crossing, so one of the channel's bindings crossed exactly at t and the others
  have not: sum over b of f_b(t - t_a) / S_b(t0 - t_a) x prod over the others of S_o(t - t_a) / S_o(t0 - t_a).

Each particle then draws which binding crossed, in proportion to its term (the eversys W-014 and W-087/88 share
one bool channel). That binding renews (re-anchored at 0) if a threshold rule replaces it, and otherwise fails
(re-anchored at D_limit: it cannot cross again until a scheduled reset).

The damage of an alarm-observed binding that has not reached its limit is, given the silence since its anchor, a
Brownian endpoint conditioned not to have crossed: `conditioned_damage` draws it (exported beliefs, forecasts).
A linear channel on the same binding reads its mean path, clipped at D_limit (an approximation).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kaful.arrays import FloatArray
from kaful.observation.first_passage import log_density, log_survival

MAX_ROUNDS = 200  # rejection rounds for a conditioned endpoint; leftovers take the clipped mean path


@dataclass
class Anchor:
    t: FloatArray  # per particle: time of the binding's last reset
    d: FloatArray  # per particle: damage just after it

    def take(self, idx: np.ndarray) -> "Anchor":
        return Anchor(t=self.t[idx].copy(), d=self.d[idx].copy())


@dataclass(frozen=True)
class Passage:
    """One binding's first-passage terms over a span (t0, t1], per particle."""

    log_s0: FloatArray  # log S(t0 - t_a)
    log_s1: FloatArray  # log S(t1 - t_a)
    log_f1: FloatArray  # log f(t1 - t_a)


def passage(anchor: Anchor, t0: float, t1: float, limit: FloatArray | float, rate: FloatArray,
            rho: float) -> Passage:
    n = anchor.t.size
    lim = np.broadcast_to(np.asarray(limit, dtype=np.float64), (n,))
    r = np.broadcast_to(np.asarray(rate, dtype=np.float64), (n,))
    distance, s2 = lim - anchor.d, rho * rho * lim * np.maximum(r, 0.0)
    e0, e1 = np.maximum(t0 - anchor.t, 0.0), np.maximum(t1 - anchor.t, 0.0)
    return Passage(log_s0=log_survival(e0, distance, r, s2), log_s1=log_survival(e1, distance, r, s2),
                   log_f1=log_density(e1, distance, r, s2))


def silence(p: Passage) -> FloatArray:
    with np.errstate(invalid="ignore"):
        out = p.log_s1 - p.log_s0
    return np.asarray(np.where(np.isnan(out), -np.inf, out), dtype=np.float64)


def report(passages: Sequence[Passage]) -> tuple[FloatArray, FloatArray]:
    """log-likelihood of one report per particle, and each binding's share of it (rows: bindings in order)."""
    terms = []
    for i, p in enumerate(passages):
        term = p.log_f1 - p.log_s0
        for j, o in enumerate(passages):
            if j != i:
                term = term + silence(o)
        terms.append(term)
    stacked = np.nan_to_num(np.stack(terms), nan=-np.inf, posinf=np.inf)
    top = np.max(stacked, axis=0)
    finite = np.isfinite(top)
    safe = np.where(finite, top, 0.0)
    total = np.where(finite, safe + np.log(np.sum(np.exp(stacked - safe), axis=0)), -np.inf)
    shares = np.where(finite, np.exp(stacked - np.where(finite, total, 0.0)), 1.0 / len(passages))
    return np.asarray(total, dtype=np.float64), np.asarray(shares, dtype=np.float64)


def attribute(shares: FloatArray, rng: np.random.Generator) -> np.ndarray:
    """Per particle, the index of the binding that crossed, drawn from its share."""
    cumulative = np.cumsum(shares, axis=0)
    u = rng.random(shares.shape[1]) * cumulative[-1]
    return np.asarray(np.argmax(cumulative > u[None, :], axis=0), dtype=np.int64)


def conditioned_damage(anchor: Anchor, t: float, limit: FloatArray | float, rate: FloatArray, rho: float,
                       rng: np.random.Generator) -> FloatArray:
    """Damage at t given no crossing since the anchor (a failed binding stays at its anchor damage).

    Brownian endpoint y from x_a with drift r over e = t - t_a, killed at the limit L: draw y ~ N(x_a + r e, s2 e),
    keep it when y < L and the path did not cross, P = 1 - exp(-2 (L - x_a)(L - y) / (s2 e)) (the bridge).
    """
    n = anchor.t.size
    lim = np.broadcast_to(np.asarray(limit, dtype=np.float64), (n,))
    r = np.maximum(np.broadcast_to(np.asarray(rate, dtype=np.float64), (n,)), 0.0)
    e = np.maximum(t - anchor.t, 0.0)
    mean = anchor.d + r * e
    out = np.minimum(mean, lim)
    s2e = rho * rho * lim * r * e
    todo = (anchor.d < lim) & (s2e > 0.0)
    out = np.where(anchor.d >= lim, anchor.d, out)
    for _ in range(MAX_ROUNDS):
        if not todo.any():
            break
        idx = np.flatnonzero(todo)
        y = mean[idx] + np.sqrt(s2e[idx]) * rng.standard_normal(idx.size)
        gap0, gap1 = lim[idx] - anchor.d[idx], lim[idx] - y
        keep = (gap1 > 0.0) & (rng.random(idx.size) > np.exp(-2.0 * gap0 * np.maximum(gap1, 0.0) / s2e[idx]))
        out[idx[keep]] = y[keep]
        todo[idx[keep]] = False
    return np.asarray(np.maximum(out, 0.0), dtype=np.float64)


def channel_bindings(observations: Sequence[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    """(channel, binding) pairs -> channel -> its bindings, sorted."""
    out: dict[str, set[str]] = {}
    for ch, b in observations:
        out.setdefault(ch, set()).add(b)
    return {ch: tuple(sorted(bs)) for ch, bs in sorted(out.items())}


def reported(value: object, event_value: str | None) -> bool:
    """Whether a row's value on an event channel reports the event (a bool channel: True; a category: its value)."""
    return value is True if event_value is None else value == event_value
