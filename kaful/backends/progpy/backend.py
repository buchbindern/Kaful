"""The ProgPy adapter: TwinBackend over rendered ProgPy PrognosticsModels (M10: build and simulate; M13: estimate
and predict).

simulate() runs ProgPy's simulate_to with fixed parameter values (default: prior medians) and no noise. The
operating profile feeds the model: a segment's values set the inputs of the channels they name (an active
condition's input carries its channel's value), and a scheduled event sets its rule's reset flag for the one
step it falls in. Event times are the first time each binding fails (a threshold-renewed binding: its first
renewal).

estimate() and predict() (M13) step the rendered model vectorized over particles (`ProgPyDynamics`); the filters
and the forecast are library-neutral numpy (kaful.estimation). ProgPy's own ParticleFilter and MonteCarlo are not
used: the first weighs every output with one Gaussian and draws from the global numpy state, the second simulates
one sample at a time (PLAN M13, decided 2026-11-16).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Delta
from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition, renewal_state
from kaful.backends.base import BuildError, Estimator, SimulationResult, StateBelief
from kaful.backends.progpy.model import Rendered, render
from kaful.config import EstimateSettings, EstimatorName
from kaful.estimation import cloud as clouds
from kaful.estimation.dynamics import BindingSpec
from kaful.estimation.filter import GroupFilter, TwinEstimator, init_rng
from kaful.estimation.groups import Group, carried, groups, read_by
from kaful.estimation.predict import EventSamples, Prediction, cloud_from, forecast
from kaful.estimation.rejuvenation import transform_for
from kaful.estimation.ukf import UKFGroup, ukf_ready
from kaful.templates.families import FAMILIES


@dataclass(frozen=True)
class ProgPyTwin:
    definition: TwinDefinition
    rendered: Rendered
    params: ParamSet

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.rendered.model_class.states)

    @property
    def event_names(self) -> tuple[str, ...]:
        return tuple(self.rendered.model_class.events)

    def units(self) -> dict[str, str]:
        return {x.param_id: x.unit for x in self.params.parameters}


class _Inputs:
    """The model inputs at each step from an operating profile: channel values and one-step reset flags."""

    def __init__(self, twin: TwinDefinition, rendered: Rendered, model: Any, profile: OperatingProfile) -> None:
        self.model, self.profile, self.rendered = model, profile, rendered
        self.channel_of = {i.name: i.channel for i in twin.external_inputs if i.channel}
        rule_of = {(m.trigger_channel, m.trigger_value): m.id for m in twin.maintenance if m.trigger != "threshold"}
        binding_of = {m.id: m.binding for m in twin.maintenance}
        self.events = sorted((e.t, rendered.reset_inputs[rule_of[(e.channel, e.value)]],
                              binding_of[rule_of[(e.channel, e.value)]])
                             for e in profile.events
                             if (e.channel, e.value) in rule_of and rule_of[(e.channel, e.value)] in rendered.reset_inputs)

    def at(self, t: float, dt: float) -> tuple[Any, frozenset[str]]:
        """The InputContainer for the step [t, t + dt), and the bindings a reset fires on in it."""
        seg = next((s for s in self.profile.segments if s.start <= t < s.end), self.profile.segments[-1])
        u = {name: float(seg.values.get(self.channel_of.get(name, ""), np.nan)) for name in self.model.inputs}
        for name in self.rendered.reset_inputs.values():
            u[name] = 0.0
        fired = set()
        for te, name, binding in self.events:
            if t <= te < t + dt:
                u[name] = 1.0
                fired.add(binding)
        return self.model.InputContainer(u), frozenset(fired)


class ProgPyDynamics:
    """kaful.estimation.dynamics.Dynamics over one rendered model (a group, or the whole twin)."""

    def __init__(self, twin: TwinDefinition, rendered: Rendered, profile: OperatingProfile,
                 fixed: Mapping[str, float]) -> None:
        self._model = rendered.model_class()
        self._rendered = rendered
        self._inputs = _Inputs(twin, rendered, self._model, profile)
        names = set(rendered.feeds)
        renewing = set(rendered.renewal_states)
        self._specs = tuple(BindingSpec(id=b.id, event=next(e.name for e in twin.events if e.binding == b.id),
                                        limit_param=b.threshold_param, renews_at_limit=b.id in renewing)
                            for b in sorted(twin.bindings, key=lambda b: b.id) if b.id in names)
        self._state_of = {b.id: b.damage_state for b in twin.bindings if b.id in names}
        self._fixed = dict(sorted(fixed.items()))

    @property
    def bindings(self) -> tuple[BindingSpec, ...]:
        return self._specs

    @property
    def fixed(self) -> Mapping[str, float]:
        return self._fixed

    def _container(self, damage: Mapping[str, FloatArray], params: Mapping[str, FloatArray], n: int) -> Any:
        values: dict[str, FloatArray] = {}
        for bid, state in self._state_of.items():
            values[state] = np.asarray(damage[bid], dtype=np.float64) if bid in damage else np.zeros(n)
        for state in self._rendered.renewal_states.values():
            values[state] = np.zeros(n)
        for pid, state in self._rendered.param_states.items():
            values[state] = np.asarray(params[pid], dtype=np.float64)
        return self._model.StateContainer({s: values[s] for s in self._model.states})

    def step(self, damage: Mapping[str, FloatArray], params: Mapping[str, FloatArray], t: float,
             dt: float) -> tuple[dict[str, FloatArray], frozenset[str]]:
        n = next(iter(damage.values())).size
        u, fired = self._inputs.at(t, dt)
        x = self._model.next_state(self._container(damage, params, n), u, dt)
        return {bid: np.array(x[s], dtype=np.float64) for bid, s in self._state_of.items()}, fired

    def reset_times(self, binding: str) -> tuple[float, ...]:
        return tuple(sorted(te for te, _, b in self._inputs.events if b == binding))

    def rates(self, params: Mapping[str, FloatArray], t: float) -> dict[str, FloatArray]:
        n = next(iter(params.values())).size if params else 1
        u, _ = self._inputs.at(t, 0.0)
        out = self._model.rates(self._container({}, params, n), u)
        return {bid: np.broadcast_to(np.asarray(r, dtype=np.float64), (n,)) for bid, r in out.items()}


def _medians(params: ParamSet) -> dict[str, float]:
    return {x.param_id: (x.prior.value if isinstance(x.prior, Delta) else x.prior.quantile(0.5))
            for x in sorted(params.parameters, key=lambda x: x.param_id)}


class ProgPyBackend:
    name: ClassVar[str] = "progpy"

    def build(self, twin: TwinDefinition, params: ParamSet, contract: SensorContract) -> ProgPyTwin:
        by_id = {x.param_id: x for x in params.parameters}
        problems = {b.id: f"no template family {b.template!r}" for b in twin.bindings if b.template not in FAMILIES}
        for b in twin.bindings:
            needed = [b.threshold_param, *b.params.values()]
            needed += [r.recovery_param for r in twin.maintenance if r.binding == b.id and r.recovery_param]
            lacking = sorted(p for p in needed if p not in by_id)
            if lacking:
                problems.setdefault(b.id, f"no parameters {lacking}")
        if problems:
            raise BuildError(problems)
        try:
            rendered = render(twin, _medians(params), {pid: x.unit for pid, x in by_id.items()})
        except ValueError as exc:
            raise BuildError({"inputs": str(exc)}) from exc
        return ProgPyTwin(twin, rendered, params)

    def simulate(self, twin: Any, profile: OperatingProfile, param_values: Mapping[str, float], *, dt: float,
                 horizon: float, seed: int) -> SimulationResult:
        assert isinstance(twin, ProgPyTwin)
        r = twin.rendered
        model = r.model_class(**{**r.defaults, **dict(param_values)})
        inputs = _Inputs(twin.definition, r, model, profile)

        def load(t: float, x: Any = None) -> Any:
            return inputs.at(t, dt)[0]

        result = model.simulate_to(horizon, load, dt=dt, save_freq=dt)
        times = np.asarray(result.times, dtype=np.float64)
        states = {s: np.asarray([x[s] for x in result.states], dtype=np.float64) for s in model.states}
        outputs = {o: np.asarray([z[o] for z in result.outputs], dtype=np.float64) for o in model.outputs}
        met = {e: np.asarray([model.threshold_met(x)[e] for x in result.states], dtype=bool) for e in model.events}
        event_states = {e: np.asarray([model.event_state(x)[e] for x in result.states], dtype=np.float64)
                        for e in model.events}
        event_times = {e: float(times[np.argmax(m)]) if m.any() else None for e, m in met.items()}
        return SimulationResult(times=times, states=states, outputs=outputs, event_times=event_times,
                                event_states=event_states, threshold_met=met)

    def dynamics(self, twin: ProgPyTwin, profile: OperatingProfile, bindings: Sequence[str],
                 per_particle: Sequence[str]) -> ProgPyDynamics:
        """A vectorized step over `bindings`, carrying `per_particle`; renewal at D_limit left to the caller."""
        medians = _medians(twin.params)
        rendered = render(twin.definition, medians, twin.units(), bindings=bindings, per_particle=per_particle,
                          threshold_renewal=False)
        chosen = set(bindings)
        group = Group(bindings=tuple(sorted(chosen)),
                      observations=tuple(o for o in twin.definition.observations if o.binding in chosen))
        fixed = {pid: medians[pid] for pid in read_by(twin.definition, group) if pid not in set(per_particle)}
        return ProgPyDynamics(twin.definition, rendered, profile, fixed)

    def estimate(self, twin: Any, params: ParamSet, profile: OperatingProfile, *, method: EstimatorName,
                 dt: float, settings: EstimateSettings, seed: np.random.SeedSequence,
                 monitored_events: Sequence[str] = (), bindings: Sequence[str] | None = None) -> Estimator:
        assert isinstance(twin, ProgPyTwin)
        definition = twin.definition
        only = None if bindings is None else set(bindings)
        by_id = {b.id: b for b in definition.bindings}
        filters: list[GroupFilter | UKFGroup] = []
        for gi, group in enumerate(groups(definition)):
            if only is not None and not set(group.bindings) & only:
                continue
            per_particle = carried(definition, params, group)
            dyn = self.dynamics(twin, profile, group.bindings, per_particle)
            cloud = clouds.draw(params, per_particle, dyn.bindings, settings.n_particles, init_rng(seed, gi))
            transforms = {pid: transform_for(params.get(pid).prior) for pid in per_particle}
            why_not = ukf_ready(group, dyn) if method == "ukf" and group.observed else None
            if method == "ukf" and group.observed and why_not is None:
                filters.append(UKFGroup(group, dyn, cloud, by_id, transforms, settings, dt, seed, gi))
                continue
            f = GroupFilter(group, dyn, cloud, by_id, transforms, settings, dt, seed, gi,
                            weigh=method == "particle_filter", monitored=frozenset(monitored_events))
            if why_not:
                f.diagnostics.note = f"UKF ran this group open-loop: {why_not}"
            filters.append(f)
        return TwinEstimator(filters, by_id, settings.forecast_samples)

    def predict(self, twin: Any, belief: StateBelief, profile: OperatingProfile, *, dt: float, horizon: float,
                settings: EstimateSettings, seed: np.random.SeedSequence,
                bindings: Sequence[str] | None = None) -> Prediction:
        """Forecast group by group (groups are independent): group g draws from spawn_key (..., g), so a group's
        forecast does not depend on which other groups are forecast. `bindings`: only the groups holding these."""
        assert isinstance(twin, ProgPyTwin)
        definition = twin.definition
        damage_state = {b.id: b.damage_state for b in definition.bindings}
        found: list[EventSamples] = []
        params: dict[str, FloatArray] = {}
        for gi, group in enumerate(groups(definition)):
            if bindings is not None and not set(group.bindings) & set(bindings):
                continue
            per_particle = carried(definition, twin.params, group)
            dyn = self.dynamics(twin, profile, group.bindings, per_particle)
            renewing = {b.id: renewal_state(b.id) for b in dyn.bindings if b.renews_at_limit}
            cloud = cloud_from(belief.particles, {b: damage_state[b] for b in group.bindings}, renewing,
                               per_particle)
            rng = np.random.default_rng(np.random.SeedSequence(seed.entropy, spawn_key=(*seed.spawn_key, gi)))
            p = forecast(dyn, cloud, belief.t, horizon, dt, settings.process_noise_cv, rng)
            found += p.events
            params.update(p.params)
        return Prediction(t_issue=belief.t, horizon=horizon, events=tuple(sorted(found, key=lambda e: e.event)),
                          params=dict(sorted(params.items())))
