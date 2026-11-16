"""Static checks on a TwinDefinition against its ParamSet, shared by build (its gates) and verify (which re-runs
them on the artifacts it loads, so a twin from a reused base run is checked too, and records each as a result).

Each problem names its check and target, and keeps the message build has always shown:
- `bound`: every ParamSet template's component has a binding;
- `params_exist`: every param_id a binding, input or maintenance rule names exists;
- `units`: D_limit's unit is D's; k's is "<D unit> / <time unit>";
- connections (M13b; kaful.templates.couplings): `connection_endpoints`: a connection joins two different bindings
  and the connections form no cycle; `connection_form`: the form exists at the recorded version, its slots are bound
  exactly, each to a `coupling` parameter whose prior lies inside the slot's bounds; `params_exist`: each slot
  parameter exists; `connection_units`: the from-binding's D and D_limit share a unit (x = D / D_limit is a pure
  number) and each slot parameter has its slot's dimension;
- observation models (M12; `observation_problems`, which also needs the SensorContract): `observation_channel`: the
  channel is in the contract; `observation_form`: the form exists and suits the channel (linear: a numeric channel;
  threshold_event: a bool channel, or a category channel with an `event_value` among its categories);
  `params_exist`: the form's slots, exactly, each an existing parameter; `observation_units`: every value parameter is
  in the channel's unit; `observation_shared`: the linear observations of one channel share y_healthy and noise_sd;
- `provenance`: no parameter rests on an `example` value (rule, M7). A source and ref on every parameter are
  enforced by the model (kaful.artifacts.base.Provenanced), so a loaded ParamSet already has them.
"""

from dataclasses import dataclass

import pint

from kaful.artifacts.distributions import Delta, Distribution, Empirical, Lognormal, Normal, TruncatedNormal, Uniform
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.artifacts.twin_definition import Connection, ObservationBinding, TwinDefinition
from kaful.observation.forms import FORMS
from kaful.templates.couplings import COUPLINGS
from kaful.units import UnitError, parse_unit, registry


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
    out += connection_problems(twin, ps)
    out += [Problem("params_exist", r.id, f"maintenance {r.id}: no parameter {r.recovery_param}")
            for r in twin.maintenance if r.recovery_param is not None and r.recovery_param not in by_id]
    return tuple(out)


def support(prior: Distribution) -> tuple[float, float]:
    """The closed interval a prior's draws lie in."""
    if isinstance(prior, Delta):
        return prior.value, prior.value
    if isinstance(prior, (Uniform, TruncatedNormal)):
        return prior.low, prior.high
    if isinstance(prior, Lognormal):
        return 0.0, float("inf")
    if isinstance(prior, Empirical):
        return min(prior.samples), max(prior.samples)
    assert isinstance(prior, Normal)
    return float("-inf"), float("inf")


def _cycle(twin: TwinDefinition) -> list[str]:
    """The bindings on a connection cycle, sorted (empty: none)."""
    feeds: dict[str, set[str]] = {}
    for c in twin.connections:
        feeds.setdefault(c.from_binding, set()).add(c.to_binding)
    on_cycle = set()
    for start in sorted(feeds):
        seen, todo = set(), sorted(feeds[start])
        while todo:
            nxt = todo.pop()
            if nxt == start:
                on_cycle.add(start)
                break
            if nxt not in seen:
                seen.add(nxt)
                todo += sorted(feeds.get(nxt, set()))
    return sorted(on_cycle)


def _slot_problems(c: Connection, twin: TwinDefinition, ps: ParamSet) -> list[Problem]:
    form, by_id = COUPLINGS[c.form], {x.param_id: x for x in ps.parameters}
    slots = {s.name: s for s in form.params}
    if sorted(c.params) != sorted(slots):
        return [Problem("connection_form", c.id, f"connection {c.id}: slots {sorted(c.params)}, the form has "
                                                 f"{sorted(slots)}")]
    out = [Problem("params_exist", c.id, f"connection {c.id}: no parameter {pid}")
           for pid in sorted(c.params.values()) if pid not in by_id]
    for slot, pid in sorted(c.params.items()):
        x = by_id.get(pid)
        if x is None:
            continue
        lo, hi = form.bounds[slot]
        a, b = support(x.prior)
        if x.role != "coupling":
            out.append(Problem("connection_form", c.id, f"connection {c.id}: {pid} has role {x.role!r}, not coupling"))
        if a < lo or b > hi:
            out.append(Problem("connection_form", c.id, f"connection {c.id}: {pid}'s prior spans [{a:g}, {b:g}], "
                                                        f"outside the slot's [{lo:g}, {hi:g}]"))
        try:
            ok = parse_unit(x.unit).dimensionality == registry().get_dimensionality(slots[slot].dimension)
        except (UnitError, pint.errors.PintError):
            ok = False
        if not ok:
            out.append(Problem("connection_units", c.id, f"connection {c.id}: {pid} is in {x.unit!r}, the slot needs "
                                                         f"{slots[slot].dimension}"))
    return out


def connection_problems(twin: TwinDefinition, ps: ParamSet) -> list[Problem]:
    """The connection gates (M13b), sorted by connection id."""
    by_id, bindings = {x.param_id: x for x in ps.parameters}, {b.id: b for b in twin.bindings}
    out: list[Problem] = []
    for c in sorted(twin.connections, key=lambda c: c.id):
        if c.from_binding == c.to_binding:
            out.append(Problem("connection_endpoints", c.id, f"connection {c.id}: joins {c.from_binding} to itself"))
        form = COUPLINGS.get(c.form)
        if form is None or form.version != c.form_version:
            out.append(Problem("connection_form", c.id, f"connection {c.id}: no coupling form {c.form!r} version "
                                                        f"{c.form_version!r}"))
            continue
        source = bindings[c.from_binding]
        limit = by_id.get(source.threshold_param)
        if limit is not None and not _compatible(limit.unit, source.damage_unit):
            out.append(Problem("connection_units", c.id, f"connection {c.id}: {source.id}'s D ({source.damage_unit!r}) "
                                                         f"and D_limit ({limit.unit!r}) do not share a unit"))
        out += _slot_problems(c, twin, ps)
    cycle = _cycle(twin)
    if cycle:
        out.append(Problem("connection_endpoints", "composite", f"connections form a cycle through {', '.join(cycle)}"))
    return out


def _form_problem(o: ObservationBinding, channel: SensorChannel) -> str | None:
    form = FORMS.get(o.form)
    if form is None:
        return f"no observation form {o.form!r}"
    numeric = channel.dtype in ("float", "int")
    if form.channel_kind == "numeric" and not numeric:
        return f"{o.form} needs a numeric channel, {channel.name} is {channel.dtype}"
    if form.channel_kind == "event":
        if numeric:
            return f"{o.form} needs a bool or category channel, {channel.name} is {channel.dtype}"
        if channel.dtype == "bool" and o.event_value is not None:
            return f"a bool channel reports True, not {o.event_value!r}"
        if channel.dtype == "category" and o.event_value not in (channel.categories or ()):
            return f"event_value {o.event_value!r} is not one of {channel.name}'s categories"
    return None


def _observation_params(o: ObservationBinding, channel: SensorChannel, ps: ParamSet) -> list[Problem]:
    by_id = {x.param_id: x for x in ps.parameters}
    slots = sorted(s.name for s in FORMS[o.form].params)
    if sorted(o.params) != slots:
        return [Problem("params_exist", o.id, f"observation {o.id}: slots {sorted(o.params)}, the form has {slots}")]
    out = [Problem("params_exist", o.id, f"observation {o.id}: no parameter {pid}")
           for pid in sorted(o.params.values()) if pid not in by_id]
    out += [Problem("observation_units", o.id, f"observation {o.id}: {pid} is in {by_id[pid].unit!r}, the channel "
                                               f"in {channel.unit!r}")
            for pid in sorted(o.params.values()) if pid in by_id and by_id[pid].unit != channel.unit]
    return out


def observation_problems(twin: TwinDefinition, ps: ParamSet, contract: SensorContract) -> tuple[Problem, ...]:
    """The observation-model gates (M12), sorted by observation id."""
    channels = {c.name: c for c in contract.channels}
    out: list[Problem] = []
    shared: dict[str, set[tuple[str, str]]] = {}
    for o in sorted(twin.observations, key=lambda o: o.id):
        channel = channels.get(o.channel)
        if channel is None:
            out.append(Problem("observation_channel", o.id, f"observation {o.id}: no channel {o.channel} in the "
                                                            f"SensorContract"))
            continue
        why = _form_problem(o, channel)
        if why is not None:
            out.append(Problem("observation_form", o.id, f"observation {o.id}: {why}"))
            continue
        out += _observation_params(o, channel, ps)
        if o.form == "linear":
            shared.setdefault(o.channel, set()).add((o.params.get("y_healthy", ""), o.params.get("noise_sd", "")))
    out += [Problem("observation_shared", ch, f"channel {ch}: its linear observations use different y_healthy or "
                                              f"noise_sd parameters {sorted(pairs)}")
            for ch, pairs in sorted(shared.items()) if len(pairs) > 1]
    return tuple(out)


def provenance_problems(ps: ParamSet) -> tuple[Problem, ...]:
    return tuple(Problem("provenance", x.param_id, f"parameter {x.param_id}: rests on an example value")
                 for x in sorted(ps.parameters, key=lambda x: x.param_id)
                 if any(e.value_role == "example" for e in x.evidence))
