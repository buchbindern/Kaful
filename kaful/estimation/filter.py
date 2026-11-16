"""The particle filter and the open-loop baseline, fed one observation at a time (batch and streaming alike).

Per group (kaful.estimation.groups): the cloud moves to the row's time (cloud.advance: drift, known maintenance,
process noise, crossings), then an observed group running the particle filter adds each particle's
log-likelihood of the row. When the effective sample size falls below settings.ess_fraction x n, the particles are
resampled systematically and the carried parameters rejuvenated (Liu-West). Open-loop never weighs: the same
draws, the same process noise, no data. The unobserved group is open-loop under every estimator.

Alarm-observed bindings (a threshold_event observation on a monitored channel) are weighed through their crossing
times: silence and reports on the channel, each particle anchored at its binding's last reset
(kaful.estimation.events). Event channels the source does not monitor are not used, and are listed in the
diagnostics.

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
from kaful.estimation import events, likelihood
from kaful.estimation.cloud import Cloud, advance, value
from kaful.estimation.dynamics import Dynamics
from kaful.estimation.groups import Group
from kaful.estimation.rejuvenation import Transform, liu_west
from kaful.estimation.resampling import DegenerateWeights, ess, normalized, systematic

INIT, NOISE, RESAMPLE, JITTER, ATTRIBUTE, EXPORT = range(6)
EXPORT_OFFSET = 0.5  # deterministic systematic offset when a belief is exported


@dataclass
class Diagnostics:
    """What a group's filter did: saved with the forecast."""

    updates: int = 0  # rows that carried data for the group
    resamples: int = 0
    degenerate: int = 0  # updates whose ESS fell below settings.degenerate_ess_fraction x n
    min_ess: float = float("inf")
    channels_used: dict[str, int] = field(default_factory=dict)  # channel -> rows it carried data (or silence)
    reports: dict[str, int] = field(default_factory=dict)  # event channel -> reports
    unmonitored_events: tuple[str, ...] = ()  # event channels with a model that the source does not monitor


def _rng(seed: np.random.SeedSequence, group: int, stream: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed.entropy,
                                                        spawn_key=(*seed.spawn_key, group, stream, *extra)))


def init_rng(seed: np.random.SeedSequence, group: int) -> np.random.Generator:
    """The stream a group's prior particles are drawn from."""
    return _rng(seed, group, INIT)


class GroupFilter:
    """One group's particles. `weigh` False: open-loop."""

    def __init__(self, group: Group, dyn: Dynamics, cloud: Cloud, bindings: Mapping[str, TemplateBinding],
                 transforms: Mapping[str, Transform], settings: EstimateSettings, dt: float,
                 seed: np.random.SeedSequence, index: int, weigh: bool,
                 monitored: frozenset[str] = frozenset()) -> None:
        self.group, self.dyn, self.cloud, self.bindings = group, dyn, cloud, bindings
        self.transforms, self.settings, self.dt = transforms, settings, dt
        self.seed, self.index = seed, index
        self.noise, self.resample_rng = _rng(seed, index, NOISE), _rng(seed, index, RESAMPLE)
        self.jitter_rng, self.attribute_rng = _rng(seed, index, JITTER), _rng(seed, index, ATTRIBUTE)
        self.weigh = weigh and group.observed
        self.t = 0.0
        self.rows = 0
        alarms = [o for o in group.observations if o.form == "threshold_event"]
        self.alarm = {} if not self.weigh else {
            ch: tuple(sorted((o.binding, o.event_value) for o in alarms if o.channel == ch))
            for ch in sorted({o.channel for o in alarms if o.channel in monitored})}
        self.free = frozenset(b for pairs in self.alarm.values() for b, _ in pairs)
        self.anchors = {b: events.Anchor(t=np.zeros(cloud.n), d=np.zeros(cloud.n)) for b in sorted(self.free)}
        self.specs = {b.id: b for b in dyn.bindings}
        self.diagnostics = Diagnostics(unmonitored_events=tuple(sorted(
            {o.channel for o in alarms} - set(self.alarm))) if self.weigh else ())

    def update(self, obs: Observation) -> None:
        if obs.t < self.t:
            raise ValueError(f"observation at t={obs.t} is before the filter's time {self.t}")
        t0, rho = self.t, self.settings.process_noise_cv
        self.rows += 1
        if not self.weigh:
            advance(self.cloud, self.dyn, t0, obs.t, self.dt, rho, self.noise)
            self.t = obs.t
            return
        log_event, used = self._advance_with_alarms(t0, obs)
        self.t = obs.t
        ll, linear_used = likelihood.linear(obs, self.group, self.bindings, self.cloud, self.dyn.fixed)
        used += list(linear_used)
        if not used:
            return
        for ch in used:
            self.diagnostics.channels_used[ch] = self.diagnostics.channels_used.get(ch, 0) + 1
        try:
            self._reweigh(self.cloud.log_w + log_event + ll)
        except DegenerateWeights as exc:
            raise DegenerateWeights(f"t={obs.t}, channels {sorted(set(used))}: {exc}") from exc

    def _limit(self, b: str) -> FloatArray | float:
        return value(self.cloud, self.dyn.fixed, self.specs[b].limit_param)

    def _advance_with_alarms(self, t0: float, obs: Observation) -> tuple[FloatArray, list[str]]:
        """Move to obs.t piece by piece (split at scheduled resets of alarm-observed bindings), adding each piece's
        first-passage terms; then apply a report. Returns the log-likelihood and the event channels used."""
        rho, n = self.settings.process_noise_cv, self.cloud.n
        cuts = {te for b in self.free for te in self.dyn.reset_times(b) if t0 <= te < obs.t}
        points = sorted({t0, obs.t, *cuts})
        log_event = np.zeros(n)
        if not self.free:
            advance(self.cloud, self.dyn, t0, obs.t, self.dt, rho, self.noise)
            return log_event, []
        last: dict[str, events.Passage] = {}
        for p0, p1 in zip(points[:-1], points[1:]):
            rates = self.dyn.rates(self.cloud.params, p0)
            advance(self.cloud, self.dyn, p0, p1, self.dt, rho, self.noise, self.free)
            for b in sorted(self.free):
                if any(p0 <= te < p1 for te in self.dyn.reset_times(b)):
                    self.anchors[b] = events.Anchor(t=np.full(n, p0),
                                                    d=self.cloud.damage[b] - rates[b] * (p1 - p0))
                last[b] = events.passage(self.anchors[b], p0, p1, self._limit(b), rates[b], rho)
                if p1 < obs.t:
                    log_event = log_event + events.silence(last[b])
        if len(points) == 1:  # a row at the filter's own time: nothing elapsed
            return log_event, []
        for ch, pairs in self.alarm.items():
            fired = [b for b, ev in pairs if events.reported(obs.values.get(ch), ev)]
            quiet = [b for b, _ in pairs if b not in fired]
            for b in quiet:
                log_event = log_event + events.silence(last[b])
            if fired:
                total, shares = events.report([last[b] for b in fired])
                log_event = log_event + total
                self._apply_report(fired, events.attribute(shares, self.attribute_rng), obs.t)
                self.diagnostics.reports[ch] = self.diagnostics.reports.get(ch, 0) + 1
        return log_event, list(self.alarm)

    def _apply_report(self, fired: Sequence[str], who: np.ndarray, t: float) -> None:
        for i, b in enumerate(fired):
            chosen = who == i
            limit = np.broadcast_to(np.asarray(self._limit(b), dtype=np.float64), chosen.shape)
            after = 0.0 if self.specs[b].renews_at_limit else limit
            d = np.where(chosen, after, self.cloud.damage[b])
            self.cloud.damage[b] = np.asarray(d, dtype=np.float64)
            if self.specs[b].renews_at_limit:
                self.cloud.renewals[b] = self.cloud.renewals[b] + chosen
            a = self.anchors[b]
            self.anchors[b] = events.Anchor(t=np.where(chosen, t, a.t), d=np.where(chosen, after, a.d))

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
        self.anchors = {b: a.take(idx) for b, a in self.anchors.items()}
        d.resamples += 1

    def weights(self) -> FloatArray:
        return normalized(self.cloud.log_w)

    def export(self, m: int) -> Cloud:
        """m equally weighted particles: systematic resampling of the current weights (fixed offset), with each
        alarm-observed binding's damage drawn given no crossing since its anchor."""
        idx = systematic(self.weights(), EXPORT_OFFSET, m)
        out = self.cloud.take(idx)
        if self.free:
            rng = _rng(self.seed, self.index, EXPORT, self.rows)
            rates = self.dyn.rates(out.params, self.t)
            for b in sorted(self.free):
                out.damage[b] = events.conditioned_damage(self.anchors[b].take(idx), self.t,
                                                          value(out, self.dyn.fixed, self.specs[b].limit_param),
                                                          rates[b], self.settings.process_noise_cv, rng)
        return out


class TwinEstimator:
    """Every group's filter behind the Estimator protocol. Groups are independent, so their exported particles
    pair up index by index into joint particles (settings.forecast_samples of them)."""

    def __init__(self, filters: Sequence[GroupFilter], bindings: Mapping[str, TemplateBinding],
                 samples: int) -> None:
        self.filters = tuple(filters)
        self._bindings = bindings
        self._samples = samples
        self._t = 0.0

    def update(self, obs: Observation) -> StateBelief:
        for f in self.filters:
            f.update(obs)
        self._t = obs.t
        return self.belief

    @property
    def belief(self) -> StateBelief:
        particles: dict[str, FloatArray] = {}
        for f in self.filters:
            cloud = f.export(self._samples)
            for bid, d in cloud.damage.items():
                particles[self._bindings[bid].damage_state] = d
            for bid, r in cloud.renewals.items():
                particles[renewal_state(bid)] = r
            particles.update(cloud.params)
        n = self._samples
        return StateBelief(t=self._t, particles=dict(sorted(particles.items())), weights=np.full(n, 1.0 / n))
