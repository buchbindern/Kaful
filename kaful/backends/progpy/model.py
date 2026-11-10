"""Render a TwinDefinition into one flat ProgPy PrognosticsModel (M10). Deterministic and LLM-free.

The model class is built from data (type(), no code text): states are each binding's damage D, plus a renewal
count for a binding replaced on threshold; inputs are the active template conditions and one reset flag per
scheduled or channel-triggered maintenance rule; outputs are each binding's health; events are each binding's
failure. All lists are sorted, so the model never depends on hash order (ProgPy's CompositeModel orders from
sets: M2 found its noise assignment changed with PYTHONHASHSEED, so it is not used).

next_state adds dt * rate from the binding's pure-numpy template (kaful.templates.families); then a reset flag
applies its rule (replace: D = 0; clean, descale: D *= 1 - recovery); a threshold rule renews D when it reaches
D_limit and counts the renewal. A condition is active only when a channel feeds it and the template has what
it needs (its nominal; accumulation's load exponent; arrhenius' nominal temperature); otherwise it is held at
nominal (f = 1), as the build flagged. A missing f parameter is then filled with 1, which leaves f at 1.
Channel values are converted to the template's units (temperature to kelvin; load and speed to the nominal's
unit) with pint once, at render time.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from progpy import PrognosticsModel  # type: ignore[import-untyped]

from kaful.artifacts.twin_definition import TemplateBinding, TwinDefinition
from kaful.templates.families import FAMILIES
from kaful.units import UnitError, parse_unit, registry

NEEDS: dict[tuple[str, str], tuple[str, ...]] = {  # (family, condition) -> parameters the condition needs
    ("accumulation", "load"): ("load_nominal", "load_exponent"),
    ("arrhenius", "temperature"): ("temperature_nominal",),
    ("bearing_fatigue", "load"): ("load_nominal",),
    ("bearing_fatigue", "speed"): ("speed_nominal",),
}
TEMPLATE_UNITS = {"temperature": "kelvin"}  # other conditions are ratios to their nominal, in its unit


@dataclass(frozen=True)
class Feed:
    """An active condition: the model input, the condition it feeds, and the affine map into template units."""

    input: str
    condition: str
    scale: float
    offset: float


@dataclass(frozen=True)
class Rendered:
    model_class: type[PrognosticsModel]
    defaults: dict[str, float]  # param_id -> prior median
    feeds: dict[str, tuple[Feed, ...]]  # binding id -> active conditions
    reset_inputs: dict[str, str]  # rule id -> input name
    renewal_states: dict[str, str]  # binding id -> renewal count state (threshold rules)


def _affine(unit: str, target: str) -> tuple[float, float]:
    q = registry().Quantity
    zero = float(q(0.0, parse_unit(unit)).to(parse_unit(target)).magnitude)
    return float(q(1.0, parse_unit(unit)).to(parse_unit(target)).magnitude) - zero, zero


def _feeds(twin: TwinDefinition, b: TemplateBinding, units: Mapping[str, str]) -> tuple[Feed, ...]:
    out = []
    for i in (i for i in twin.external_inputs if i.binding == b.id and i.provenance == "channel"):
        assert i.condition is not None
        needed = NEEDS.get((b.template, i.condition), ())
        if not all(n in b.params for n in needed):
            continue
        target = TEMPLATE_UNITS.get(i.condition) or units[b.params[f"{i.condition}_nominal"]]
        try:
            scale, offset = _affine(i.unit, target)
        except UnitError as exc:
            raise ValueError(f"input {i.name}: {exc}") from exc
        out.append(Feed(input=i.name, condition=i.condition, scale=scale, offset=offset))
    return tuple(sorted(out, key=lambda f: f.input))


def render(twin: TwinDefinition, defaults: Mapping[str, float], units: Mapping[str, str]) -> Rendered:
    """`defaults`: param_id -> value (prior medians); `units`: param_id -> unit."""
    bindings = sorted(twin.bindings, key=lambda b: b.id)
    feeds = {b.id: _feeds(twin, b, units) for b in bindings}
    resets = {r.id: f"{r.id}_reset" for r in twin.maintenance if r.trigger != "threshold"}
    renewals = {r.binding: f"{r.binding}_renewals" for r in twin.maintenance if r.trigger == "threshold"}
    rules = sorted(twin.maintenance, key=lambda r: r.id)
    states = sorted([b.damage_state for b in bindings] + list(renewals.values()))
    inputs = sorted([f.input for fs in feeds.values() for f in fs] + list(resets.values()))

    def params_of(self: Any, b: TemplateBinding) -> dict[str, Any]:
        values = {slot: self.parameters[pid] for slot, pid in b.params.items()}
        for slot in FAMILIES[b.template].params:
            values.setdefault(slot.name, 1.0)  # an f input no active condition uses: f stays 1
        return values

    def next_state(self: Any, x: Any, u: Any, dt: float) -> Any:
        for b in bindings:
            conditions = {f.condition: np.asarray(u[f.input] * f.scale + f.offset) for f in feeds[b.id]}
            d = np.asarray(x[b.damage_state], dtype=np.float64)
            x[b.damage_state] = d + dt * FAMILIES[b.template]().rate(d, conditions, params_of(self, b))
        for r in rules:
            limit = self.parameters[next(b.threshold_param for b in bindings if b.id == r.binding)]
            state = next(b.damage_state for b in bindings if b.id == r.binding)
            if r.trigger == "threshold":
                if x[state] >= limit:
                    x[state], x[renewals[r.binding]] = 0.0, x[renewals[r.binding]] + 1.0
            elif u[resets[r.id]] > 0.5:
                x[state] = 0.0 if r.action == "replace" else x[state] * (1.0 - self.parameters[r.recovery_param])
        return x

    def health(self: Any, x: Any, b: TemplateBinding) -> float:
        return float(np.clip(1.0 - x[b.damage_state] / self.parameters[b.threshold_param], 0.0, 1.0))

    def failed(self: Any, x: Any, b: TemplateBinding) -> bool:
        if b.id in renewals:
            return bool(x[renewals[b.id]] >= 1.0)
        return bool(x[b.damage_state] >= self.parameters[b.threshold_param])

    attrs = {
        "inputs": inputs, "states": states, "outputs": sorted(f"{b.component}_health" for b in bindings),
        "events": sorted(f"{b.component}_failure" for b in bindings),
        "default_parameters": {**dict(sorted(defaults.items())), "x0": {s: 0.0 for s in states},
                               "process_noise": 0.0, "measurement_noise": 0.0},
        "next_state": next_state,
        "output": lambda self, x: self.OutputContainer({f"{b.component}_health": health(self, x, b) for b in bindings}),
        "event_state": lambda self, x: {f"{b.component}_failure": health(self, x, b) for b in bindings},
        "threshold_met": lambda self, x: {f"{b.component}_failure": failed(self, x, b) for b in bindings},
    }
    model_class = type(f"Twin_{twin.machine_id}", (PrognosticsModel,), attrs)
    return Rendered(model_class, dict(defaults), feeds, resets, renewals)
