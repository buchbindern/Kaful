"""Lives, intervals and maintenance actions in the machine's time unit, shared by parameterize (life priors,
recovery priors) and build (maintenance rules and schedules).

- A calendar interval (day ... year) becomes time units through a duty cycle: the default per time unit
  (settings.parameterize.duty_defaults, wide, flagged) until machine.toml declares one.
- A documented interval's maintenance action comes from its quote: the first action of
  settings.build.action_words (replace, descale, clean) any of its word prefixes appears in; check or inspect
  alone is no action. machine.toml [[maintenance]] adds actions the documents do not attach.
"""

from dataclasses import dataclass

from kaful.artifacts.base import SourceKind
from kaful.artifacts.param_set import LifeStatement
from kaful.config import BuildSettings, DutyDefault, MaintenanceAction, MachineConfig, ParameterizeSettings
from kaful.doctext import words
from kaful.units import UnitError, parse_unit, registry


def duty(p: ParameterizeSettings, time_unit: str) -> DutyDefault:
    """The default duty cycle for the time unit, by name or by unit ('h' finds 'hour')."""
    if time_unit in p.duty_defaults:
        return p.duty_defaults[time_unit]
    for name, d in sorted(p.duty_defaults.items()):
        try:
            if parse_unit(name) == parse_unit(time_unit):
                return d
        except UnitError:
            continue
    raise KeyError(f"no default duty cycle for time unit {time_unit!r}")


def years(s: LifeStatement, p: ParameterizeSettings) -> float:
    unit = s.unit.lower()
    days = p.calendar_days.get(unit, p.calendar_days.get(unit.rstrip("s"), 0.0))
    return s.value * days / p.calendar_days["year"]


def comparable(s: LifeStatement, p: ParameterizeSettings, time_unit: str) -> float:
    """The statement in time units; a calendar one at the default duty cycle's median."""
    return s.in_time_unit if s.in_time_unit is not None else years(s, p) * duty(p, time_unit).per_year_median


def declared_interval(value: float, unit: str, time_unit: str, p: ParameterizeSettings) -> tuple[float, bool]:
    """(interval in time units, converted from a calendar unit through the default duty cycle)."""
    u = unit.lower().rstrip("s")
    if u in p.calendar_days:
        return value * p.calendar_days[u] / p.calendar_days["year"] * duty(p, time_unit).per_year_median, True
    if u == time_unit.lower().rstrip("s"):
        return value, False
    return float(registry().Quantity(value, parse_unit(unit)).to(parse_unit(time_unit)).magnitude), False


def action_of(text: str, b: BuildSettings) -> MaintenanceAction | None:
    said = words(text)
    for action, prefixes in b.action_words.items():
        if any(w.startswith(tuple(prefixes)) for w in said):
            return action
    return None


@dataclass(frozen=True)
class Action:
    component: str
    action: MaintenanceAction
    interval: float | None  # time units; None when only a channel reports it
    calendar: bool  # converted through the default duty cycle
    source: SourceKind  # "manual" (a life statement) or "deployment" ([[maintenance]])
    ref: str  # the statement id, or machine.toml [[maintenance]]
    channel: str | None = None
    category: str | None = None
    on_threshold: bool = False


def documented_actions(statements: tuple[LifeStatement, ...], machine: MachineConfig, p: ParameterizeSettings,
                       b: BuildSettings) -> list[Action]:
    """One action per (component, action): a declaration wins; else the most frequent documented interval."""
    time_unit = machine.confirmed_time_unit()
    found: dict[tuple[str, str], Action] = {}
    for s in sorted((s for s in statements if s.usable and s.evidence), key=lambda s: s.id):
        action = action_of(s.evidence[0].quote, b)
        if action is None:
            continue
        a = Action(s.component, action, comparable(s, p, time_unit), s.in_time_unit is None, "manual", s.id)
        old = found.get((s.component, action))
        if old is None or (a.interval or 0) < (old.interval or 0):
            found[(s.component, action)] = a
    for d in machine.maintenance:
        interval, calendar = None, False
        if d.interval is not None and d.unit is not None:
            interval, calendar = declared_interval(d.interval, d.unit, time_unit, p)
        found[(d.component, d.action)] = Action(d.component, d.action, interval, calendar, "deployment",
                                                f"machine.toml [[maintenance]] {d.component} {d.action}",
                                                d.channel, d.category, d.on_threshold)
    return [found[k] for k in sorted(found)]
