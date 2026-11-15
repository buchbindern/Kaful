"""Static checks on a TwinDefinition against its ParamSet, shared by build (its gates) and verify (which re-runs
them on the artifacts it loads, so a twin from a reused base run is checked too, and records each as a result).

Each problem names its check and target, and keeps the message build has always shown:
- `bound`: every ParamSet template's component has a binding;
- `params_exist`: every param_id a binding, input or maintenance rule names exists;
- `units`: D_limit's unit is D's; k's is "<D unit> / <time unit>";
- `connection_units`: both ends of every connection are pint-compatible;
- `provenance`: no parameter rests on an `example` value (rule, M7). A source and ref on every parameter are
  enforced by the model (kaful.artifacts.base.Provenanced), so a loaded ParamSet already has them.
"""

from dataclasses import dataclass

import pint

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.units import UnitError, parse_unit


@dataclass(frozen=True)
class Problem:
    check: str
    target: str  # binding id, input name, connection id, rule id, or param_id
    message: str


def _compatible(a: str, b: str) -> bool:
    try:
        return parse_unit(a).dimensionality == parse_unit(b).dimensionality
    except (UnitError, pint.errors.PintError):
        return False


def _binding_problems(twin: TwinDefinition, ps: ParamSet) -> list[Problem]:
    by_id = {x.param_id: x for x in ps.parameters}
    out: list[Problem] = []
    for b in twin.bindings:
        refs = [b.threshold_param, *b.params.values()]
        out += [Problem("params_exist", b.id, f"binding {b.id}: no parameter {r}") for r in refs if r not in by_id]
        limit, k = by_id.get(b.threshold_param), by_id.get(b.params.get("k", ""))
        if limit is not None and limit.unit != b.damage_unit:
            out.append(Problem("units", b.id, f"binding {b.id}: D_limit unit {limit.unit!r} is not D's "
                                              f"{b.damage_unit!r}"))
        if k is not None and k.unit != f"{b.damage_unit} / {twin.time_unit}":
            out.append(Problem("units", b.id, f"binding {b.id}: k unit {k.unit!r} is not "
                                              f"'{b.damage_unit} / {twin.time_unit}'"))
    return out


def static_problems(twin: TwinDefinition, ps: ParamSet,
                    quarantined: frozenset[str] = frozenset()) -> tuple[Problem, ...]:
    """The build gates on a built twin, in the order build has always listed them. A `quarantined` component
    (verify, M11) was left out of the twin on purpose, so it is not reported as unbound."""
    by_id = {x.param_id for x in ps.parameters}
    bound = {b.component for b in twin.bindings} | quarantined
    out = [Problem("bound", t.component, f"component {t.component}: not bound")
           for t in ps.templates if t.component not in bound]
    out += _binding_problems(twin, ps)
    out += [Problem("params_exist", i.name, f"input {i.name}: no parameter {i.nominal_param}")
            for i in twin.external_inputs if i.nominal_param is not None and i.nominal_param not in by_id]
    out += [Problem("connection_units", c.id, f"connection {c.id}: {c.source_unit!r} does not convert to "
                                              f"{c.target_unit!r}")
            for c in twin.connections if not _compatible(c.source_unit, c.target_unit)]
    out += [Problem("params_exist", r.id, f"maintenance {r.id}: no parameter {r.recovery_param}")
            for r in twin.maintenance if r.recovery_param is not None and r.recovery_param not in by_id]
    return tuple(out)


def provenance_problems(ps: ParamSet) -> tuple[Problem, ...]:
    return tuple(Problem("provenance", x.param_id, f"parameter {x.param_id}: rests on an example value")
                 for x in sorted(ps.parameters, key=lambda x: x.param_id)
                 if any(e.value_role == "example" for e in x.evidence))
