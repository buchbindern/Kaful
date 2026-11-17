"""What the data can say about each part (used by f_calibration and g_sensitivity).

observed: the part has an observation of its own; joined: in an observed group only through a connection (its
posterior equals its prior); unobserved: no observation reaches it; not_identifiable: one of several parts read on one
channel (judged through the group's first failure).
"""

from kaful.artifacts.evaluation import PartClass
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.estimation.groups import groups
from kaful.scoring.groups import shared_groups


def part_classes(twin: TwinDefinition) -> dict[str, PartClass]:
    """Binding id -> its class, sorted by binding id."""
    shared = {b for _, bs in shared_groups(twin) for b in bs}
    own = {o.binding for o in twin.observations}
    observed = {b for g in groups(twin) if g.observed for b in g.bindings}
    out: dict[str, PartClass] = {}
    for b in sorted(x.id for x in twin.bindings):
        out[b] = ("not_identifiable" if b in shared else "observed" if b in own
                  else "joined" if b in observed else "unobserved")
    return out
