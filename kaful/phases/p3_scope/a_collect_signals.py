"""Step a: the signals understand already produced, per component (G, O, D, declared).

- G: fault modes on the component that count: a source other than llm_judgement or default, not downgraded
  (their quotes were verified by understand). The others are listed as uncounted.
- O: channels in a counted fault mode's `observable_by` whose role is observation, or that have a secondary
  observation role. A range whose value role is in `observation_limit_roles` is a documented limit.
- O named (kept apart, lower confidence): an observation channel that no counted fault mode of the component
  links, whose own name or description names the component as the thing measured (helpers/text.measures).
  Only components with G get one: it decides nothing for the rest.
- O threshold events (their own kind): channels in a counted fault mode's `observable_by` whose role is event
  (a warning or limit message: the machine reports when the limit is reached).
- D: channels in a counted fault mode's `driven_by` whose role is operating_input (or secondary).
Link confidence is computed by code: a fault-mode link has the lower of the channel's and its best fault
mode's confidence; a named channel has the channel's confidence times `named_channel_factor`.
- declared: machine.toml [[components]] declares it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from kaful.artifacts.machine_model import Component, FaultMode, MachineModel
from kaful.artifacts.scope import ChannelLink, NamedChannel
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.config import MachineConfig, ScopeSettings
from kaful.phases.p3_scope.helpers.text import measures, names, stem

UNCOUNTED_SOURCES = ("llm_judgement", "default")


@dataclass
class Collected:
    fault_modes: list[str] = field(default_factory=list)
    uncounted: list[str] = field(default_factory=list)
    observed: dict[str, ChannelLink] = field(default_factory=dict)
    driven: dict[str, ChannelLink] = field(default_factory=dict)
    named: dict[str, NamedChannel] = field(default_factory=dict)
    events: dict[str, ChannelLink] = field(default_factory=dict)
    declared: bool = False


def counts(mode: FaultMode) -> bool:
    return mode.source not in UNCOUNTED_SOURCES and mode.downgraded_from is None


def _plays(channel: SensorChannel, role: str) -> bool:
    return channel.role == role or any(r.role == role for r in channel.secondary_roles)


def _link(links: dict[str, ChannelLink], channel: SensorChannel, mode: FaultMode, limit: bool) -> None:
    old = links.get(channel.name)
    modes = tuple(sorted({mode.id, *(old.fault_modes if old else ())}))
    confidence = max(min(channel.confidence, mode.confidence), old.confidence if old else 0.0)
    links[channel.name] = ChannelLink(channel=channel.name, fault_modes=modes, documented_limit=limit,
                                      confidence=confidence)


def _limit(channel: SensorChannel, roles: Sequence[str]) -> bool:
    ev = channel.range_evidence
    return channel.range is not None and ev is not None and ev.value_role in roles


def collect(model: MachineModel, contract: SensorContract, machine: MachineConfig,
            settings: ScopeSettings, heads: Sequence[str]) -> dict[str, Collected]:
    """The signals per component; `heads` are understand's generic head words (helpers/text.names)."""
    by_name = {c.name: c for c in contract.channels}
    out = {c.id: Collected(declared=c.source == "deployment") for c in model.components}
    for mode in sorted(model.fault_modes, key=lambda m: m.id):
        got = out[mode.component]
        if not counts(mode):
            got.uncounted.append(mode.id)
            continue
        got.fault_modes.append(mode.id)
        for name in mode.observable_by:
            channel = by_name.get(name)
            if channel is not None and _plays(channel, "observation"):
                _link(got.observed, channel, mode, _limit(channel, settings.observation_limit_roles))
            elif channel is not None and _plays(channel, "event"):
                _link(got.events, channel, mode, False)
        for name in mode.driven_by:
            channel = by_name.get(name)
            if channel is not None and _plays(channel, "operating_input"):
                _link(got.driven, channel, mode, False)
    declared = {d.id for d in machine.components}
    for cid, got in out.items():
        got.declared = got.declared or cid in declared
    _named(model, contract, settings, heads, out)
    return out


def _named(model: MachineModel, contract: SensorContract, settings: ScopeSettings, heads: Sequence[str],
           out: dict[str, Collected]) -> None:
    quantities = {stem(w) for w in settings.channel_quantity_words}
    observing = sorted((c for c in contract.channels if _plays(c, "observation")), key=lambda c: c.name)
    for component in sorted(model.components, key=lambda c: c.id):
        got = out[component.id]
        if got.fault_modes:
            for channel in (c for c in observing if c.name not in got.observed):
                hit = _names(channel, component, heads, quantities)
                if hit is not None:
                    got.named[channel.name] = NamedChannel(
                        channel=channel.name, named_in=hit[0], text=hit[1],
                        documented_limit=_limit(channel, settings.observation_limit_roles),
                        confidence=channel.confidence * settings.named_channel_factor)


def _names(channel: SensorChannel, component: Component, heads: Sequence[str],
           quantities: set[str]) -> tuple[Literal["name", "description"], str] | None:
    forms = names(component, heads)
    if measures(channel.name.replace("_", " "), forms, quantities):
        return "name", channel.name
    if channel.description and measures(channel.description, forms, quantities):
        return "description", channel.description
    return None
