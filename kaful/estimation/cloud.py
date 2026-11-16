"""A cloud of particles over one group of bindings, and how it moves forward in time.

Each particle carries every binding's damage D, a renewal count for a binding replaced at its limit, and every
uncertain parameter the group reads (drawn from its prior; a delta prior stays a fixed value in the dynamics).

Process noise (an estimator assumption, settings.estimate.process_noise_cv = rho): D follows a Wiener process with
variance rate rho^2 * D_limit * dD/dt, so from a new part the time to D_limit is inverse Gaussian with mean
D_limit / rate and coefficient of variation exactly rho. D is clipped at 0. No noise is added in a step where a
maintenance action reset the binding.

A crossing is D reaching D_limit within a step (from below); its time is the step's end, as synthesize records
truth. A binding with a threshold rule is then renewed (D = 0, count + 1); any other stays failed until a
scheduled reset.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import ParamSet
from kaful.estimation.dynamics import BindingSpec, Dynamics


@dataclass
class Cloud:
    damage: dict[str, FloatArray]  # binding id -> D per particle
    renewals: dict[str, FloatArray]  # binding id -> renewal count per particle (renewing bindings only)
    params: dict[str, FloatArray]  # carried param_id -> value per particle
    log_w: FloatArray = field(default_factory=lambda: np.zeros(0))

    @property
    def n(self) -> int:
        return int(self.log_w.size)

    def take(self, idx: np.ndarray) -> "Cloud":
        """The particles at `idx`, with equal weights."""
        return Cloud(damage={k: v[idx].copy() for k, v in self.damage.items()},
                     renewals={k: v[idx].copy() for k, v in self.renewals.items()},
                     params={k: v[idx].copy() for k, v in self.params.items()}, log_w=np.zeros(idx.size))

    def copy(self) -> "Cloud":
        return self.take(np.arange(self.n)).with_log_w(self.log_w.copy())

    def with_log_w(self, log_w: FloatArray) -> "Cloud":
        self.log_w = log_w
        return self


def value(cloud: Cloud, fixed: Mapping[str, float], param_id: str) -> FloatArray | float:
    """A parameter per particle when carried, else its fixed value."""
    return cloud.params[param_id] if param_id in cloud.params else fixed[param_id]


def draw(ps: ParamSet, carried: Sequence[str], bindings: Sequence[BindingSpec], n: int,
         rng: np.random.Generator) -> Cloud:
    """n particles from the priors: new parts (D = 0), each carried parameter drawn in sorted param_id order."""
    params = {}
    for pid in sorted(carried):
        prior = ps.get(pid).prior
        if isinstance(prior, Delta):
            raise ValueError(f"{pid} has a delta prior: it is fixed, not carried")
        params[pid] = np.asarray(prior.sample(rng, n), dtype=np.float64)
    return Cloud(damage={b.id: np.zeros(n) for b in bindings},
                 renewals={b.id: np.zeros(n) for b in bindings if b.renews_at_limit},
                 params=params, log_w=np.zeros(n))


def advance(cloud: Cloud, dyn: Dynamics, t0: float, t1: float, dt: float, rho: float,
            rng: np.random.Generator) -> dict[str, FloatArray]:
    """Move the cloud from t0 to t1 in steps of at most dt; per binding, the first crossing time (NaN: none).

    Every step draws one standard normal per binding per particle, in binding order, whether or not it is used,
    so the random stream's use never depends on the data.
    """
    n = cloud.n
    first = {b.id: np.full(n, np.nan) for b in dyn.bindings}
    t = t0
    while t < t1:
        h = min(dt, t1 - t)
        before = cloud.damage
        drift, reset = dyn.step(before, cloud.params, t, h)
        rates = dyn.rates(cloud.params, t)
        t_next = t1 if h >= t1 - t else t + h
        after = {}
        for b in dyn.bindings:
            limit = value(cloud, dyn.fixed, b.limit_param)
            eps = rng.standard_normal(n)
            d = drift[b.id]
            if b.id not in reset:
                d = d + np.sqrt(rho * rho * limit * np.maximum(rates[b.id], 0.0) * h) * eps
            d = np.maximum(d, 0.0)
            if b.renews_at_limit:
                hit = d >= limit
                d = np.where(hit, 0.0, d)
                cloud.renewals[b.id] = cloud.renewals[b.id] + hit
            else:
                hit = (before[b.id] < limit) & (d >= limit)
            newly = hit & np.isnan(first[b.id])
            first[b.id][newly] = t_next
            after[b.id] = d
        cloud.damage = after
        t = t_next
    return first
