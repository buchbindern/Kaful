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

For estimation (M13), `render` can also take a subset of bindings (one observed group), parameters carried per
particle (each becomes a constant state named by `param_state`, read instead of the model parameter), and
`threshold_renewal=False`, which leaves renewing at D_limit to the estimator (it handles crossings itself). next_state
is vectorized: every state may hold one value per particle. `rates(x, u)` gives each binding's dD/dt, which scales
the estimator's process noise.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from progpy import PrognosticsModel  # type: ignore[import-untyped]

from kaful.artifacts.twin_definition import TemplateBinding, TwinDefinition, renewal_state
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
    param_states: dict[str, str]  # param_id -> state carrying it per particle (estimation only)


def param_state(param_id: str) -> str:
    """The state that carries a parameter per particle."""
    return f"param:{param_id}"


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


def render(twin: TwinDefinition, defaults: Mapping[str, float], units: Mapping[str, str], *,
           bindings: Sequence[str] | None = None, per_particle: Sequence[str] = (),
           threshold_renewal: bool = True) -> Rendered:
    """`defaults`: param_id -> value (prior medians); `units`: param_id -> unit.

    `bindings`: render only these binding ids (default: all). `per_particle`: param_ids carried as states.
    `threshold_renewal`: False leaves renewing at D_limit to the caller.
    """
    chosen = set(bindings) if bindings is not None else {b.id for b in twin.bindings}
    unknown = sorted(chosen - {b.id for b in twin.bindings})
    if unknown:
        raise ValueError(f"no bindings {unknown}")
    kept = sorted((b for b in twin.bindings if b.id in chosen), key=lambda b: b.id)
    feeds = {b.id: _feeds(twin, b, units) for b in kept}
    rules = sorted((r for r in twin.maintenance if r.binding in chosen), key=lambda r: r.id)
    resets = {r.id: f"{r.id}_reset" for r in rules if r.trigger != "threshold"}
    renewals = {r.binding: renewal_state(r.binding) for r in rules if r.trigger == "threshold"}
    carried = {pid: param_state(pid) for pid in sorted(set(per_particle))}
    states = sorted([b.damage_state for b in kept] + list(renewals.values()) + list(carried.values()))
    inputs = sorted([f.input for fs in feeds.values() for f in fs] + list(resets.values()))
    limit_of = {b.id: b.threshold_param for b in kept}
    state_of = {b.id: b.damage_state for b in kept}

    def value(self: Any, x: Any, pid: str) -> Any:
        return x[carried[pid]] if pid in carried else self.parameters[pid]

    def params_of(self: Any, x: Any, b: TemplateBinding) -> dict[str, Any]:
        values = {slot: value(self, x, pid) for slot, pid in b.params.items()}
        for slot in FAMILIES[b.template].params:
            values.setdefault(slot.name, 1.0)  # an f input no active condition uses: f stays 1
        return values

    def rates(self: Any, x: Any, u: Any) -> dict[str, Any]:
        out = {}
        for b in kept:
            conditions = {f.condition: np.asarray(u[f.input] * f.scale + f.offset) for f in feeds[b.id]}
            d = np.asarray(x[b.damage_state], dtype=np.float64)
            out[b.id] = FAMILIES[b.template]().rate(d, conditions, params_of(self, x, b))
        return out

    def next_state(self: Any, x: Any, u: Any, dt: float) -> Any:
        for bid, rate in rates(self, x, u).items():
            x[state_of[bid]] = np.asarray(x[state_of[bid]], dtype=np.float64) + dt * rate
        for r in rules:
            limit, state = value(self, x, limit_of[r.binding]), state_of[r.binding]
            if r.trigger == "threshold":
                if threshold_renewal:
                    hit = np.asarray(x[state]) >= limit
                    x[state] = np.where(hit, 0.0, x[state])
                    x[renewals[r.binding]] = x[renewals[r.binding]] + hit
            elif u[resets[r.id]] > 0.5:
                if r.action == "replace":
                    kept_fraction = 0.0
                elif r.recovery_param is None:
                    raise ValueError(f"maintenance rule {r.id}: a {r.action} needs a recovery parameter")
                else:
                    kept_fraction = 1.0 - value(self, x, r.recovery_param)
                x[state] = x[state] * kept_fraction
        return x

    def health(self: Any, x: Any, b: TemplateBinding) -> float:
        return float(np.clip(1.0 - x[b.damage_state] / value(self, x, b.threshold_param), 0.0, 1.0))

    def failed(self: Any, x: Any, b: TemplateBinding) -> bool:
        if b.id in renewals:
            return bool(x[renewals[b.id]] >= 1.0)
        return bool(x[b.damage_state] >= value(self, x, b.threshold_param))

    x0 = {s: 0.0 for s in states} | {state: float(defaults[pid]) for pid, state in carried.items()}
    attrs = {
        "inputs": inputs, "states": states, "outputs": sorted(f"{b.component}_health" for b in kept),
        "events": sorted(f"{b.component}_failure" for b in kept), "is_vectorized": True,
        "default_parameters": {**dict(sorted(defaults.items())), "x0": x0,
                               "process_noise": 0.0, "measurement_noise": 0.0},
        "next_state": next_state, "rates": rates,
        "output": lambda self, x: self.OutputContainer({f"{b.component}_health": health(self, x, b) for b in kept}),
        "event_state": lambda self, x: {f"{b.component}_failure": health(self, x, b) for b in kept},
        "threshold_met": lambda self, x: {f"{b.component}_failure": failed(self, x, b) for b in kept},
    }
    model_class = type(f"Twin_{twin.machine_id}", (PrognosticsModel,), attrs)
    return Rendered(model_class, dict(defaults), feeds, resets, renewals, carried)
