"""Bindings no observation's likelihood reads (M13b, decided 2026-11-17).

In an observed group, the likelihood depends on the bindings with an observation and on everything upstream of them
through connections (a source's damage drives its target's rate, never the reverse). Any other binding in the group
(the eversys boilers, joined to W-014's group only through the water filter's connection) is unread: no row's
likelihood depends on its damage or on the parameters only it reads, so their posterior is their prior.

Carrying such a binding through resampling would make its parameters a clump of a few ancestors' prior draws
(particle impoverishment: in run fd6bf0 the KS test rejected 'posterior = prior' in 9-11 of 20 units). Instead the
filter steps only the read bindings and keeps, per particle, the integrals of the coupling form's basis functions
of the source's damage fraction since the target's last replace (m is linear in them: m = sum_i w_i g_i). At export
the unread parameters are drawn from the prior and the damage from the path, exactly in distribution for the Wiener
damage model (cloud.py):

    S = sum_i w_i(params) * integral g_i(x) dt,   D = max(k S + sqrt(rho^2 D_limit k S) z, 0).

A binding is reconstructed only when that is exact: one incoming connection, from a simulated binding, with a
basis-linear form; it feeds nothing; no operating input feeds its conditions (f = 1); its maintenance is at most
scheduled replaces (they restart the integrals); and no read binding shares its parameters. Otherwise it stays
simulated, and `reasons` says why. The split is structural: it reads the twin and the observation bindings, never a
component's name.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Delta, Distribution
from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import Connection, TwinDefinition
from kaful.estimation.groups import Group, read_by
from kaful.templates.couplings import COUPLINGS


@dataclass(frozen=True)
class Reconstructed:
    binding: str
    connection: Connection  # its one incoming connection
    replaces: tuple[str, ...]  # its scheduled replace rule ids


@dataclass(frozen=True)
class Split:
    simulated: tuple[str, ...]  # bindings the filter steps, sorted
    reconstructed: tuple[Reconstructed, ...]  # sorted by binding
    unread_params: tuple[str, ...]  # uncertain parameters only the reconstructed bindings read, sorted
    reasons: dict[str, str] = field(default_factory=dict)  # an unread binding kept simulated -> why


def likelihood_bindings(twin: TwinDefinition, group: Group) -> set[str]:
    """The bindings with an observation in the group, and every binding upstream of them through connections."""
    out = {o.binding for o in group.observations}
    todo = sorted(out)
    while todo:
        b = todo.pop()
        for c in twin.connections:
            if c.to_binding == b and c.from_binding not in out:
                out.add(c.from_binding)
                todo.append(c.from_binding)
    return out


def _params_of(twin: TwinDefinition, b: str) -> set[str]:
    binding = next(x for x in twin.bindings if x.id == b)
    out = {binding.threshold_param, *binding.params.values()}
    out |= {p for c in twin.connections if c.to_binding == b for p in c.params.values()}
    return out


def _why_not(twin: TwinDefinition, b: str, read: set[str]) -> str | None:
    feeds = sorted(c.to_binding for c in twin.connections if c.from_binding == b)
    incoming = [c for c in twin.connections if c.to_binding == b]
    if feeds:
        return f"it feeds {', '.join(feeds)}"
    if len(incoming) != 1:
        return f"{len(incoming)} connections feed it (one is needed)"
    if incoming[0].from_binding not in read:
        return f"its source {incoming[0].from_binding} is not simulated"
    if not hasattr(COUPLINGS[incoming[0].form], "basis"):
        return f"coupling form {incoming[0].form} is not linear in a basis"
    if any(i.binding == b and i.provenance == "channel" for i in twin.external_inputs):
        return "an operating input feeds its conditions"
    other = sorted(f"{r.action} ({r.trigger})" for r in twin.maintenance
                   if r.binding == b and not (r.action == "replace" and r.trigger == "schedule"))
    if other:
        return f"its maintenance is not only scheduled replaces: {', '.join(other)}"
    return None


def split(twin: TwinDefinition, ps: ParamSet, group: Group) -> Split:
    if not group.observed:
        return Split(simulated=group.bindings, reconstructed=(), unread_params=())
    read = likelihood_bindings(twin, group)
    reasons: dict[str, str] = {}
    chosen: list[str] = []
    for b in sorted(set(group.bindings) - read):
        why = _why_not(twin, b, read)
        if why is None:
            chosen.append(b)
        else:
            reasons[b] = why
    simulated = tuple(sorted(set(group.bindings) - set(chosen)))
    shared = set(read_by(twin, Group(bindings=simulated, observations=group.observations)))
    for b in list(chosen):
        overlap = sorted(_params_of(twin, b) & shared)
        if overlap:
            chosen.remove(b)
            reasons[b] = f"a simulated binding also reads {', '.join(overlap)}"
    simulated = tuple(sorted(set(group.bindings) - set(chosen)))
    recon = tuple(Reconstructed(binding=b, connection=next(c for c in twin.connections if c.to_binding == b),
                                replaces=tuple(sorted(r.id for r in twin.maintenance if r.binding == b)))
                  for b in sorted(chosen))
    unread = sorted({p for r in recon for p in _params_of(twin, r.binding)
                     if not isinstance(ps.get(p).prior, Delta)})
    return Split(simulated=simulated, reconstructed=recon, unread_params=tuple(unread), reasons=reasons)


@dataclass(frozen=True)
class PathSpec:
    """What the filter integrates per particle for one reconstructed binding."""

    binding: str
    source: str
    source_limit: str  # param_id of the source's D_limit
    form: str
    resets: tuple[float, ...]  # times of the binding's scheduled replaces (the integrals restart)

    def increment(self, fraction: FloatArray, h: float) -> FloatArray:
        return h * np.stack(COUPLINGS[self.form]().basis(fraction))


@dataclass(frozen=True)
class Reconstruction:
    """Everything export needs to rebuild the unread bindings: their paths, priors and fixed values."""

    paths: tuple[PathSpec, ...]
    priors: dict[str, Distribution]  # unread param_id -> prior
    fixed: dict[str, float]  # param_ids the reconstructed bindings read with a delta prior
    damage_of: dict[str, tuple[str, str, Connection]]  # binding -> (k param, D_limit param, connection)


def reconstruction(twin: TwinDefinition, ps: ParamSet, s: Split, profile: OperatingProfile) -> Reconstruction:
    by_id = {b.id: b for b in twin.bindings}
    rules = {r.id: r for r in twin.maintenance}
    paths, damage_of = [], {}
    for r in s.reconstructed:
        times = tuple(sorted(e.t for e in profile.events for rid in r.replaces
                             if (e.channel, e.value) == (rules[rid].trigger_channel, rules[rid].trigger_value)))
        src = by_id[r.connection.from_binding]
        paths.append(PathSpec(binding=r.binding, source=src.id, source_limit=src.threshold_param,
                              form=r.connection.form, resets=times))
        damage_of[r.binding] = (by_id[r.binding].params["k"], by_id[r.binding].threshold_param, r.connection)
    reads = {p for r in s.reconstructed for p in _params_of(twin, r.binding)}
    return Reconstruction(paths=tuple(paths), priors={p: ps.get(p).prior for p in s.unread_params},
                          fixed={p: float(ps.get(p).prior.value) for p in sorted(reads)  # type: ignore[union-attr]
                                 if isinstance(ps.get(p).prior, Delta)},
                          damage_of=damage_of)


def rebuild(rec: Reconstruction, paths: Mapping[str, FloatArray], rho: float,
            rng: np.random.Generator) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    """(damage per reconstructed binding, unread parameters), one value per particle of `paths`."""
    n = next(iter(paths.values())).shape[1]
    params = {pid: np.asarray(prior.sample(rng, n), dtype=np.float64) for pid, prior in sorted(rec.priors.items())}

    def value(pid: str) -> FloatArray:
        return params[pid] if pid in params else np.full(n, rec.fixed[pid])

    damage = {}
    for b, (k_pid, limit_pid, c) in sorted(rec.damage_of.items()):
        slots = {slot: value(pid) for slot, pid in c.params.items()}
        w = COUPLINGS[c.form]().weights(slots)
        s = sum(wi * gi for wi, gi in zip(w, paths[b]))
        mean = value(k_pid) * s
        sd = np.sqrt(rho * rho * value(limit_pid) * np.maximum(mean, 0.0))
        damage[b] = np.maximum(mean + sd * rng.standard_normal(n), 0.0)
    return damage, params


def zero_paths(rec: Reconstruction | None, n: int) -> dict[str, FloatArray]:
    if rec is None:
        return {}
    return {p.binding: np.zeros((len(COUPLINGS[p.form]().basis(np.zeros(1))), n)) for p in rec.paths}


def advance_paths(paths: dict[str, FloatArray], specs: Sequence[PathSpec], damage: Mapping[str, FloatArray],
                  limit_of: Mapping[str, FloatArray | float], t: float, h: float) -> None:
    """Add one step [t, t + h) of each path integral (the source's fraction at the step's start, as the model step
    reads it), then restart it if a scheduled replace fires in the step (as the model step resets D)."""
    for p in specs:
        paths[p.binding] = paths[p.binding] + p.increment(np.asarray(damage[p.source]) / limit_of[p.source], h)
        if any(t <= te < t + h for te in p.resets):
            paths[p.binding] = np.zeros_like(paths[p.binding])
