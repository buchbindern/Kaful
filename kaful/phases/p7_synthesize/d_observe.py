"""Step d: what one unit reports in its observation window, through the observation models, and its true events.

- Sampling grid: every `sample_interval` time units from 0 to the window end t_end. Each linear channel reads the
  forms at the grid (kaful.observation.forms.linear_channel: y_healthy plus each binding's change, its true values),
  plus Gaussian noise from the unit's noise stream (channels in name order, one draw per grid point); an int channel
  is rounded. A row carries the usage clock (rounded for an int clock) or, with no clock, a timestamp.
- Events: a threshold_event observation reports at each simulation step where its binding's D reached D_limit (a
  first failure, or every renewal of a threshold rule), up to t_end, as its own row; two reports at one time merge.
- Every other contract channel stays empty: nothing is invented for a channel no reviewed observation model covers.
- Truth: per event, its first time, every time D reached D_limit (renewals, and crossings again after a scheduled
  reset), and whether the first is inside the window; and the true
  states on the grid. Truth never enters the rows.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from kaful.arrays import BoolArray, FloatArray
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import EventTruth
from kaful.artifacts.twin_definition import TemplateBinding, TwinDefinition, renewal_state
from kaful.backends.base import SimulationResult
from kaful.data.source import ChannelValue, Observation
from kaful.observation.forms import ThresholdEvent, linear_channel
from kaful.observation.protocol import DamageView
from kaful.units import to_seconds


@dataclass(frozen=True)
class Observed:
    rows: tuple[Observation, ...]
    grid: FloatArray
    states: dict[str, FloatArray]  # true state -> value at each grid time
    events: tuple[EventTruth, ...]


def crossings(twin: TwinDefinition, b: TemplateBinding, result: SimulationResult) -> BoolArray:
    """Per simulation step: D reached D_limit in the step ending there: each renewal of a threshold rule, or each
    time threshold_met turns on (again after a scheduled reset)."""
    renewals = result.states.get(renewal_state(b.id))
    if renewals is not None:
        return np.concatenate(([False], np.diff(renewals) > 0.5))
    met = result.threshold_met[next(e.name for e in twin.events if e.binding == b.id)]
    return np.concatenate(([bool(met[0])], met[1:] & ~met[:-1]))


def _at(times: FloatArray, grid: FloatArray) -> np.ndarray:
    return np.clip(np.searchsorted(times, grid, side="right") - 1, 0, times.size - 1)


def _linear_rows(twin: TwinDefinition, result: SimulationResult, values: Mapping[str, float], grid: FloatArray,
                 contract: SensorContract, rng: np.random.Generator) -> dict[str, list[ChannelValue]]:
    by_binding, idx = {b.id: b for b in twin.bindings}, _at(result.times, grid)
    out: dict[str, list[ChannelValue]] = {}
    channels = sorted({o.channel for o in twin.observations if o.form == "linear"})
    for ch in channels:
        obs = sorted((o for o in twin.observations if o.form == "linear" and o.channel == ch), key=lambda o: o.id)
        terms = []
        for o in obs:
            b = by_binding[o.binding]
            fraction = result.states[b.damage_state][idx] / values[b.threshold_param]
            terms.append((DamageView(fraction, np.zeros(grid.size, dtype=bool)), values[o.params["y_limit"]]))
        y = linear_channel(values[obs[0].params["y_healthy"]], terms)
        y = y + values[obs[0].params["noise_sd"]] * rng.standard_normal(grid.size)
        integer = contract.channel(ch).dtype == "int"
        out[ch] = [int(round(v)) if integer else float(v) for v in y]
    return out


def _event_rows(twin: TwinDefinition, result: SimulationResult, contract: SensorContract,
                t_end: float) -> dict[float, dict[str, ChannelValue]]:
    by_binding = {b.id: b for b in twin.bindings}
    out: dict[float, dict[str, ChannelValue]] = {}
    for o in sorted((o for o in twin.observations if o.form == "threshold_event"), key=lambda o: o.id):
        crossed = crossings(twin, by_binding[o.binding], result)
        fired = ThresholdEvent().sample(DamageView(np.zeros(crossed.size), crossed), {}, np.random.default_rng(0))
        value: ChannelValue = True if contract.channel(o.channel).dtype == "bool" else o.event_value
        for t in result.times[fired > 0.5]:
            if t <= t_end:
                out.setdefault(float(t), {})[o.channel] = value
    return out


def _clock(contract: SensorContract, t: float, origin: datetime, unit_seconds: float | None
           ) -> tuple[dict[str, ChannelValue], datetime | None]:
    if contract.usage_clock is None:
        assert unit_seconds is not None
        return {}, origin + timedelta(seconds=t * unit_seconds)
    integer = contract.channel(contract.usage_clock).dtype == "int"
    return {contract.usage_clock: int(round(t)) if integer else float(t)}, None


def rows(grid: FloatArray, linear: dict[str, list[ChannelValue]], events: dict[float, dict[str, ChannelValue]],
         contract: SensorContract, time_unit: str, origin: datetime) -> tuple[Observation, ...]:
    """Grid rows and event rows, merged by time and sorted."""
    merged: dict[float, dict[str, ChannelValue]] = {}
    for k, t in enumerate(grid):
        merged[float(t)] = {ch: v[k] for ch, v in sorted(linear.items())}
    for t, vals in events.items():
        merged.setdefault(t, {}).update(vals)
    seconds = to_seconds(1.0, time_unit) if contract.usage_clock is None else None
    out = []
    for t in sorted(merged):
        clock, stamp = _clock(contract, t, origin, seconds)
        values = {**clock, **merged[t]}
        reported_t = float(clock[contract.usage_clock]) if contract.usage_clock else t  # type: ignore[arg-type]
        out.append(Observation(t=reported_t, timestamp=stamp, values=dict(sorted(values.items()))))
    return tuple(out)


def truth_events(twin: TwinDefinition, result: SimulationResult, t_end: float) -> tuple[EventTruth, ...]:
    out = []
    for e in sorted(twin.events, key=lambda e: e.name):
        b = next(b for b in twin.bindings if b.id == e.binding)
        first = result.event_times.get(e.name)
        reached = tuple(float(t) for t in result.times[crossings(twin, b, result)])
        out.append(EventTruth(event=e.name, first_time=first, limit_times=reached,
                              in_window=first is not None and first <= t_end))
    return tuple(out)


def observe(twin: TwinDefinition, result: SimulationResult, values: Mapping[str, float], contract: SensorContract,
            time_unit: str, sample_interval: float, t_end: float, origin: datetime,
            noise: np.random.SeedSequence) -> Observed:
    grid = np.arange(0.0, t_end + sample_interval * 1e-9, sample_interval)
    linear = _linear_rows(twin, result, values, grid, contract, np.random.default_rng(noise))
    found = rows(grid, linear, _event_rows(twin, result, contract, t_end), contract, time_unit, origin)
    idx = _at(result.times, grid)
    states = {s: result.states[s][idx] for s in sorted(result.states)}
    return Observed(rows=found, grid=grid, states=states, events=truth_events(twin, result, t_end))
