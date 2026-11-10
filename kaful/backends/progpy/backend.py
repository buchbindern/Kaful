"""The ProgPy adapter: TwinBackend over one rendered PrognosticsModel per machine (M10: build and simulate).

simulate() runs ProgPy's simulate_to with fixed parameter values (default: prior medians) and no noise. The
operating profile feeds the model: a segment's values set the inputs of the channels they name (an active
condition's input carries its channel's value), and a scheduled event sets its rule's reset flag for the one
step it falls in. Event times are the first time each binding fails (a threshold-renewed binding: its first
renewal). estimate() and predict() belong to M13 (ProgPy's particle filter, UKF and Monte Carlo predictor).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from kaful.artifacts.distributions import Delta
from kaful.artifacts.forecast import EventForecast, OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import BuildError, Estimator, SimulationResult, StateBelief
from kaful.backends.progpy.model import Rendered, render
from kaful.templates.families import FAMILIES

M13 = "estimation and prediction arrive in M13 (ProgPy particle filter, UKF, Monte Carlo predictor)"


@dataclass(frozen=True)
class ProgPyTwin:
    definition: TwinDefinition
    rendered: Rendered

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(self.rendered.model_class.states)

    @property
    def event_names(self) -> tuple[str, ...]:
        return tuple(self.rendered.model_class.events)


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
        defaults = {pid: (x.prior.value if isinstance(x.prior, Delta) else x.prior.quantile(0.5))
                    for pid, x in sorted(by_id.items())}
        try:
            rendered = render(twin, defaults, {pid: x.unit for pid, x in by_id.items()})
        except ValueError as exc:
            raise BuildError({"inputs": str(exc)}) from exc
        return ProgPyTwin(twin, rendered)

    def simulate(self, twin: Any, profile: OperatingProfile, param_values: Mapping[str, float], *, dt: float,
                 horizon: float, seed: int) -> SimulationResult:
        assert isinstance(twin, ProgPyTwin)
        r = twin.rendered
        model = r.model_class(**{**r.defaults, **dict(param_values)})
        channel_of = {i.name: i.channel for i in twin.definition.external_inputs if i.channel}
        rule_of = {(m.trigger_channel, m.trigger_value): m.id for m in twin.definition.maintenance
                   if m.trigger != "threshold"}
        events = sorted((e.t, r.reset_inputs[rule_of[(e.channel, e.value)]]) for e in profile.events
                        if (e.channel, e.value) in rule_of)

        def load(t: float, x: Any = None) -> Any:
            seg = next((s for s in profile.segments if s.start <= t < s.end), profile.segments[-1])
            u = {name: float(seg.values.get(channel_of.get(name, ""), np.nan)) for name in model.inputs}
            for name in r.reset_inputs.values():
                u[name] = 0.0
            for te, name in events:
                if t <= te < t + dt:
                    u[name] = 1.0
            return model.InputContainer(u)

        result = model.simulate_to(horizon, load, dt=dt, save_freq=dt)
        times = np.asarray(result.times, dtype=np.float64)
        states = {s: np.asarray([x[s] for x in result.states], dtype=np.float64) for s in model.states}
        outputs = {o: np.asarray([z[o] for z in result.outputs], dtype=np.float64) for o in model.outputs}
        met = [model.threshold_met(x) for x in result.states]
        event_times = {e: next((float(t) for t, m in zip(times, met, strict=True) if m[e]), None)
                       for e in model.events}
        return SimulationResult(times=times, states=states, outputs=outputs, event_times=event_times)

    def estimate(self, twin: Any, params: ParamSet, *, method: Literal["particle_filter", "ukf"],
                 augment: Sequence[str], n_particles: int, seed: int) -> Estimator:
        raise NotImplementedError(M13)

    def predict(self, twin: Any, belief: StateBelief, profile: OperatingProfile, *, horizon: float, n_samples: int,
                seed: int) -> tuple[EventForecast, ...]:
        raise NotImplementedError(M13)
