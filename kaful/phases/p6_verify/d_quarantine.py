"""Step d: quarantine each component that failed a dynamic check, and reduce the twin to the rest (M11).

Only a component's own dynamic checks quarantine it (c_check_dynamics.DYNAMIC_CHECKS). A static or composite
failure (provenance, units, the backend cannot build or run the twin) is never quarantined: it stops the run.
A component fed by a quarantined one through a connection is quarantined with it, as a dependent (no
component-to-component connection exists until M13). The reduced twin drops the quarantined bindings and
their events, inputs, maintenance rules, scheduled profile events and flags; phase.py re-runs every check on it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult, Quarantine
from kaful.phases.p6_verify.c_check_dynamics import DYNAMIC_CHECKS


def _component(port: str) -> str:
    return port.split(".", 1)[0]


def _dependents(twin: TwinDefinition, failed: set[str]) -> dict[str, set[str]]:
    """Each failed component -> the components it feeds, directly or not."""
    feeds: dict[str, set[str]] = {}
    for c in twin.connections:
        feeds.setdefault(_component(c.source_port), set()).add(_component(c.target_port))
    out = {}
    for root in sorted(failed):
        seen: set[str] = set()
        todo = [root]
        while todo:
            for nxt in sorted(feeds.get(todo.pop(), set()) - seen - {root}):
                seen.add(nxt)
                todo.append(nxt)
        out[root] = seen & set(twin.components)
    return out


def quarantine(twin: TwinDefinition, results: Sequence[CheckResult]) -> tuple[Quarantine, ...]:
    failing: dict[str, list[CheckResult]] = {}
    for r in results:
        if r.status == "fail" and r.check in DYNAMIC_CHECKS and r.target in twin.components:
            failing.setdefault(r.target, []).append(r)
    dependents = _dependents(twin, set(failing))
    out = [Quarantine(component=c, checks=tuple(sorted(r.check for r in rs)),
                      reason="; ".join(f"{r.check}: {r.message}" for r in sorted(rs, key=lambda r: r.check)),
                      dependents=tuple(sorted(dependents[c] - set(failing))))
           for c, rs in failing.items()]
    for d in sorted(set[str]().union(*dependents.values()) - set(failing)):
        roots = sorted(c for c, deps in dependents.items() if d in deps)
        out.append(Quarantine(component=d, checks=("depends_on_quarantined",),
                              reason=f"depends on quarantined {', '.join(roots)}"))
    return tuple(sorted(out, key=lambda q: q.component))


def reduce(twin: TwinDefinition, quarantined: Sequence[Quarantine], base: Mapping[str, Any]) -> TwinDefinition:
    """The twin without the quarantined components, as an artifact of this run (`base`)."""
    out = {q.component for q in quarantined}
    kept = tuple(b for b in twin.bindings if b.component not in out)
    ids = {b.id for b in kept}
    removed_rules = {(r.trigger_channel, r.trigger_value) for r in twin.maintenance if r.binding not in ids}
    profiles = tuple(p.model_copy(update={"events": tuple(e for e in p.events
                                                          if (e.channel, e.value) not in removed_rules)})
                     for p in twin.profiles)
    return TwinDefinition(
        **base, time_unit=twin.time_unit, components=tuple(c for c in twin.components if c not in out),
        bindings=kept,
        connections=tuple(c for c in twin.connections
                          if _component(c.source_port) not in out and _component(c.target_port) not in out),
        external_inputs=tuple(i for i in twin.external_inputs if i.binding is None or i.binding in ids),
        events=tuple(e for e in twin.events if e.binding in ids),
        maintenance=tuple(r for r in twin.maintenance if r.binding in ids), profiles=profiles,
        flags={c: f for c, f in twin.flags.items() if c not in out})
