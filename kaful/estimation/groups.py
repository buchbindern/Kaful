"""Split a twin's bindings into groups the estimators can treat independently.

Bindings are independent a priori (no connections yet; maintenance is a known input), so data on one binding says
nothing about another unless both act on the same channel. Bindings linked through shared channels form one
observed group, filtered jointly. Bindings with no observation model form one unobserved group: their posterior is
their prior moved forward, so every estimator runs it open-loop. Once connections exist (M13b), a connection joins
groups the same way a shared channel does.
"""

from dataclasses import dataclass

from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import ObservationBinding, TwinDefinition


@dataclass(frozen=True)
class Group:
    bindings: tuple[str, ...]  # sorted binding ids
    observations: tuple[ObservationBinding, ...]  # sorted by id; empty for the unobserved group

    @property
    def observed(self) -> bool:
        return bool(self.observations)

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(sorted({o.channel for o in self.observations}))


def groups(twin: TwinDefinition) -> tuple[Group, ...]:
    """Observed groups (sorted by first binding id), then the unobserved group if any binding has no channel."""
    parent = {b.id: b.id for b in twin.bindings}

    def root(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    by_channel: dict[str, list[str]] = {}
    for o in sorted(twin.observations, key=lambda o: o.id):
        by_channel.setdefault(o.channel, []).append(o.binding)
    for members in by_channel.values():
        for other in members[1:]:
            a, b = sorted((root(members[0]), root(other)))
            parent[b] = a
    observed_ids = {o.binding for o in twin.observations}
    clusters: dict[str, list[str]] = {}
    for bid in sorted(observed_ids):
        clusters.setdefault(root(bid), []).append(bid)
    out = [Group(bindings=tuple(sorted(ids)),
                 observations=tuple(sorted((o for o in twin.observations if o.binding in ids), key=lambda o: o.id)))
           for ids in clusters.values()]
    out.sort(key=lambda g: g.bindings)
    rest = tuple(sorted(b.id for b in twin.bindings if b.id not in observed_ids))
    if rest:
        out.append(Group(bindings=rest, observations=()))
    return tuple(out)


def read_by(twin: TwinDefinition, group: Group) -> tuple[str, ...]:
    """Every param_id the group's bindings and observations read, sorted."""
    chosen = set(group.bindings)
    ids = set()
    for b in twin.bindings:
        if b.id in chosen:
            ids |= {b.threshold_param, *b.params.values()}
    ids |= {r.recovery_param for r in twin.maintenance if r.binding in chosen and r.recovery_param is not None}
    ids |= {i.nominal_param for i in twin.external_inputs if i.binding in chosen and i.nominal_param is not None}
    ids |= {p for o in group.observations for p in o.params.values()}
    return tuple(sorted(ids))


def carried(twin: TwinDefinition, ps: ParamSet, group: Group) -> tuple[str, ...]:
    """The uncertain parameters the group reads (any prior but a delta): carried per particle, sorted."""
    return tuple(pid for pid in read_by(twin, group) if not isinstance(ps.get(pid).prior, Delta))
