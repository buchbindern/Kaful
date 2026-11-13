"""One binding's series from one simulation, and where its maintenance acted. Used by c_check_dynamics.

The checks read what the backend reports (states, event_state, threshold_met), never its internals. Where a
scheduled reset lands relative to its event time (the step before or after `t`) is the backend's convention, so
a reset is looked for in a window of one step on each side of the event.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from kaful.arrays import BoolArray, FloatArray
from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.twin_definition import MaintenanceRule, TemplateBinding, TwinDefinition, renewal_state
from kaful.backends.base import SimulationResult


@dataclass(frozen=True)
class Reset:
    """A scheduled maintenance event of one rule, and the saved steps where its effect may show."""

    rule: MaintenanceRule
    t: float
    window: tuple[int, ...]


@dataclass(frozen=True)
class BindingTrace:
    component: str
    times: FloatArray
    damage: FloatArray
    limit: float
    health: FloatArray
    event_state: FloatArray
    met: BoolArray
    renewals: FloatArray | None  # the renewal count of a threshold rule, else None
    resets: tuple[Reset, ...]
    recovery: Mapping[str, float]  # rule id -> recovery value in this run

    @property
    def reset_steps(self) -> frozenset[int]:
        """Steps where D may fall: every scheduled reset's window and every renewal."""
        steps = {i for r in self.resets for i in r.window}
        if self.renewals is not None:
            steps |= {int(i) + 1 for i in np.flatnonzero(np.diff(self.renewals))}
        return frozenset(steps)

    def increment(self) -> float:
        """The largest one-step growth of D outside resets (0 when D never grows)."""
        steps = np.diff(self.damage)
        keep = np.ones(steps.size, dtype=bool)
        keep[[i - 1 for i in self.reset_steps if 1 <= i <= steps.size]] = False
        grown = steps[keep]
        return float(max(grown.max(), 0.0)) if grown.size else 0.0


def _resets(twin: TwinDefinition, b: TemplateBinding, profile: OperatingProfile, times: FloatArray) -> tuple[Reset, ...]:
    rules = {(r.trigger_channel, r.trigger_value): r for r in twin.maintenance
             if r.binding == b.id and r.trigger != "threshold"}
    out = []
    for e in profile.events:
        rule = rules.get((e.channel, e.value))
        if rule is None:
            continue
        at = int(np.searchsorted(times, e.t, side="right"))
        window = tuple(i for i in (at - 1, at, at + 1) if 1 <= i < times.size)
        if window:
            out.append(Reset(rule=rule, t=e.t, window=window))
    return tuple(out)


def binding_trace(twin: TwinDefinition, b: TemplateBinding, result: SimulationResult, profile: OperatingProfile,
                  values: Mapping[str, float]) -> BindingTrace:
    event = next(e.name for e in twin.events if e.binding == b.id)
    renewal = renewal_state(b.id)
    recovery = {r.id: values[r.recovery_param] for r in twin.maintenance
                if r.binding == b.id and r.recovery_param is not None}
    return BindingTrace(
        component=b.component, times=result.times, damage=result.states[b.damage_state],
        limit=values[b.threshold_param], health=result.outputs[f"{b.component}_health"],
        event_state=result.event_states[event], met=result.threshold_met[event],
        renewals=result.states.get(renewal), resets=_resets(twin, b, profile, result.times), recovery=recovery)
