"""Which parts the data can tell apart (no truth here): the label-invariant groups (one channel, several parts) and
the parts joined to an observed group only through a connection."""

from kaful.artifacts.twin_definition import TwinDefinition
from kaful.estimation.groups import groups


def shared_groups(twin: TwinDefinition) -> list[tuple[str, tuple[str, ...]]]:
    """(channel, bindings) for each channel several parts are observed on linearly: not identifiable per part."""
    out = []
    for g in groups(twin):
        for ch in g.channels:
            on = tuple(sorted({o.binding for o in g.observations if o.channel == ch and o.form == "linear"}))
            if len(on) > 1:
                out.append((ch, on))
    return out


def joined_bindings(twin: TwinDefinition) -> tuple[str, ...]:
    """Bindings in an observed group with no observation of their own (joined through a connection), sorted."""
    own = {o.binding for o in twin.observations}
    return tuple(sorted(b for g in groups(twin) if g.observed for b in g.bindings if b not in own))
