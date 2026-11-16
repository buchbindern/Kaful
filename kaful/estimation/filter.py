"""The particle filter and the open-loop baseline, fed one observation at a time (batch and streaming alike).

Per group (kaful.estimation.groups): the cloud moves to the row's time (cloud.advance: drift, known maintenance,
process noise, crossings), then an observed group running the particle filter adds each particle's
log-likelihood of the row. When the effective sample size falls below settings.ess_fraction x n, the particles are
resampled systematically and the carried parameters rejuvenated (Liu-West). Open-loop never weighs: the same
draws, the same process noise, no data. The unobserved group is open-loop under every estimator.

Every random draw comes from the group's own streams (SeedSequence(entropy, spawn_key=(..., group, stream))), never
from the global numpy state, so a run is reproducible and one group's draws never depend on another's.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.twin_definition import TemplateBinding, renewal_state
from kaful.backends.base import StateBelief
from kaful.config import EstimateSettings
from kaful.data.source import Observation
from kaful.estimation import likelihood
from kaful.estimation.cloud import Cloud, advance
from kaful.estimation.dynamics import Dynamics
from kaful.estimation.groups import Group
from kaful.estimation.rejuvenation import Transform, liu_west
from kaful.estimation.resampling import ess, normalized, systematic

INIT, NOISE, RESAMPLE, JITTER = range(4)
EXPORT_OFFSET = 0.5  # deterministic systematic offset when a belief is exported (no random draw)


@dataclass
class Diagnostics:
    """What a group's filter did: saved with the forecast."""

    updates: int = 0  # rows that carried data for the group
    resamples: int = 0
    degenerate: int = 0  # updates whose ESS fell below settings.degenerate_ess_fraction x n
    min_ess: float = float("inf")
    channels_used: dict[str, int] = field(default_factory=dict)


def streams(seed: np.random.SeedSequence, group: int) -> list[np.random.Generator]:
    """The group's generators: init, process noise, resampling, jitter."""
    return [np.random.default_rng(np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, group, s)))
            for s in (INIT, NOISE, RESAMPLE, JITTER)]


class GroupFilter:
    """One group's particles. `weigh` False: open-loop."""

    def __init__(self, group: Group, dyn: Dynamics, cloud: Cloud, bindings: Mapping[str, TemplateBinding],
                 transforms: Mapping[str, Transform], settings: EstimateSettings, dt: float,
                 rngs: Sequence[np.random.Generator], weigh: bool) -> None:
        self.group, self.dyn, self.cloud, self.bindings = group, dyn, cloud, bindings
        self.transforms, self.settings, self.dt = transforms, settings, dt
        self.noise, self.resample_rng, self.jitter_rng = rngs[NOISE], rngs[RESAMPLE], rngs[JITTER]
        self.weigh = weigh and group.observed
        self.t = 0.0
        self.diagnostics = Diagnostics()

    def update(self, obs: Observation) -> None:
        if obs.t < self.t:
            raise ValueError(f"observation at t={obs.t} is before the filter's time {self.t}")
        advance(self.cloud, self.dyn, self.t, obs.t, self.dt, self.settings.process_noise_cv, self.noise)
        self.t = obs.t
        if not self.weigh:
            return
        ll, used = likelihood.linear(obs, self.group, self.bindings, self.cloud, self.dyn.fixed)
        if not used:
            return
        for ch in used:
            self.diagnostics.channels_used[ch] = self.diagnostics.channels_used.get(ch, 0) + 1
        self._reweigh(self.cloud.log_w + ll)

    def _reweigh(self, log_w: FloatArray) -> None:
        w = normalized(log_w)
        n, e = w.size, ess(w)
        d = self.diagnostics
        d.updates += 1
        d.min_ess = min(d.min_ess, e)
        if e < self.settings.degenerate_ess_fraction * n:
            d.degenerate += 1
        if e >= self.settings.ess_fraction * n:
            with np.errstate(divide="ignore"):
                self.cloud.log_w = np.log(w)
            return
        idx = systematic(w, float(self.resample_rng.random()))
        params = liu_west(self.cloud.params, w, idx, self.transforms, self.settings.liu_west_discount,
                          self.jitter_rng)
        self.cloud = self.cloud.take(idx)
        self.cloud.params = params
        d.resamples += 1

    def weights(self) -> FloatArray:
        return normalized(self.cloud.log_w)

    def export(self) -> Cloud:
        """Equally weighted particles (deterministic systematic resampling of the current weights)."""
        return self.cloud.take(systematic(self.weights(), EXPORT_OFFSET))


class TwinEstimator:
    """Every group's filter behind the Estimator protocol. Groups are independent, so their exported particles
    pair up index by index into joint particles."""

    def __init__(self, filters: Sequence[GroupFilter], bindings: Mapping[str, TemplateBinding]) -> None:
        self.filters = tuple(filters)
        self._bindings = bindings
        self._t = 0.0

    def update(self, obs: Observation) -> StateBelief:
        for f in self.filters:
            f.update(obs)
        self._t = obs.t
        return self.belief

    @property
    def belief(self) -> StateBelief:
        particles: dict[str, FloatArray] = {}
        n = 0
        for f in self.filters:
            cloud = f.export()
            n = cloud.n
            for bid, d in cloud.damage.items():
                particles[self._bindings[bid].damage_state] = d
            for bid, r in cloud.renewals.items():
                particles[renewal_state(bid)] = r
            particles.update(cloud.params)
        return StateBelief(t=self._t, particles=dict(sorted(particles.items())), weights=np.full(n, 1.0 / n))
