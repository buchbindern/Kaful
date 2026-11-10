"""Step d: the build gates (hard rule 2). Each failure is a line in the phase record, and the run stops.

- every full-model component of the ParamSet has exactly one binding, with k and D_limit;
- every param_id a binding, input or rule names exists in the ParamSet;
- units: D_limit's unit is D's; k's is "<D unit> / <time unit>"; a channel input has its condition's
  dimension (checked when wired); every connection's ends are pint-compatible;
- a clean or descale rule has its recovery parameter.
"""

from collections.abc import Sequence

import pint

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.units import UnitError, parse_unit


def _compatible(a: str, b: str) -> bool:
    try:
        return parse_unit(a).dimensionality == parse_unit(b).dimensionality
    except (UnitError, pint.errors.PintError):
        return False


def check(twin: TwinDefinition, ps: ParamSet, missing: Sequence[str]) -> tuple[str, ...]:
    by_id = {x.param_id: x for x in ps.parameters}
    failures = list(missing)
    bound = [b.component for b in twin.bindings]
    failures += [f"component {t.component}: not bound" for t in ps.templates if t.component not in bound]
    for b in twin.bindings:
        refs = [b.threshold_param, *b.params.values()]
        failures += [f"binding {b.id}: no parameter {r}" for r in refs if r not in by_id]
        limit, k = by_id.get(b.threshold_param), by_id.get(b.params.get("k", ""))
        if limit is not None and limit.unit != b.damage_unit:
            failures.append(f"binding {b.id}: D_limit unit {limit.unit!r} is not D's {b.damage_unit!r}")
        if k is not None and k.unit != f"{b.damage_unit} / {twin.time_unit}":
            failures.append(f"binding {b.id}: k unit {k.unit!r} is not '{b.damage_unit} / {twin.time_unit}'")
    for i in twin.external_inputs:
        if i.nominal_param is not None and i.nominal_param not in by_id:
            failures.append(f"input {i.name}: no parameter {i.nominal_param}")
    failures += [f"connection {c.id}: {c.source_unit!r} does not convert to {c.target_unit!r}"
                 for c in twin.connections if not _compatible(c.source_unit, c.target_unit)]
    failures += [f"maintenance {r.id}: no parameter {r.recovery_param}" for r in twin.maintenance
                 if r.recovery_param is not None and r.recovery_param not in by_id]
    return tuple(failures)
