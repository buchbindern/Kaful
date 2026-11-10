"""Step c: maintenance rules and the stored operating profiles (M10, approved 2026-11-10).

Each bound component's documented or declared action (kaful.lifetime.documented_actions) becomes a rule:
replace sets D = 0; clean and descale set D *= 1 - recovery (the ParamSet's `recovery_<action>`). A declared
channel and category trigger it in data; a documented interval triggers it in the documented-maintenance
profile on the pseudo channel `<schedule_channel>_<component>` (one per component, so events never collide;
when two actions fall at the same time, replace wins). Profiles run `profile_horizon_lives` times the longest
median life: `nominal` (no maintenance) and `documented_maintenance`. A component with no rule is flagged; a
calendar interval converted through the default duty cycle is flagged.
"""

import math

from kaful.artifacts.forecast import OperatingProfile, ProfileSegment, ScheduledEvent
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import MaintenanceRule
from kaful.config import BuildSettings
from kaful.lifetime import Action

FLAG_NO_MAINTENANCE = "no maintenance action documented or declared ([[maintenance]] can add one)"
FLAG_CALENDAR = "maintenance interval {rule}: a calendar period converted through the default duty cycle"
ORDER = {"replace": 0, "descale": 1, "clean": 2}  # at the same time, the stronger action wins


def rules(actions: list[Action], bound: set[str], b: BuildSettings) -> tuple[list[MaintenanceRule], dict[str, list[str]]]:
    out: list[MaintenanceRule] = []
    flags: dict[str, list[str]] = {}
    for a in (a for a in actions if a.component in bound):
        rid = f"{a.component}_{a.action}"
        out.append(MaintenanceRule(
            id=rid, binding=a.component, action=a.action,
            recovery_param=f"{a.component}.recovery_{a.action}" if a.action != "replace" else None,
            trigger_channel=a.channel or f"{b.schedule_channel}_{a.component}",
            trigger_value=a.category or rid, interval=a.interval, source=a.source,
            ref=a.ref))
        if a.calendar:
            flags.setdefault(a.component, []).append(FLAG_CALENDAR.format(rule=rid))
    for component in sorted(bound - {r.binding for r in out}):
        flags.setdefault(component, []).append(FLAG_NO_MAINTENANCE)
    return sorted(out, key=lambda r: r.id), flags


def horizon(ps: ParamSet, bound: set[str], b: BuildSettings) -> float:
    lives = [x.prior.quantile(0.5) for x in ps.parameters if x.name == "life_nominal" and x.component in bound]
    return b.profile_horizon_lives * max(lives, default=1.0)


def profiles(maintenance: list[MaintenanceRule], end: float) -> tuple[OperatingProfile, ...]:
    segment = (ProfileSegment(start=0.0, end=end, values={}),)
    events: dict[tuple[float, str], tuple[int, str]] = {}
    for r in (r for r in maintenance if r.interval is not None):
        assert r.interval is not None
        for n in range(1, math.floor(end / r.interval) + 1):
            key, value = (round(n * r.interval, 9), r.trigger_channel), (ORDER[r.action], r.trigger_value)
            events[key] = min(events.get(key, value), value)
    scheduled = tuple(ScheduledEvent(t=t, channel=ch, value=v[1]) for (t, ch), v in sorted(events.items()))
    return (OperatingProfile(name="documented_maintenance", segments=segment, events=scheduled),
            OperatingProfile(name="nominal", segments=segment))
