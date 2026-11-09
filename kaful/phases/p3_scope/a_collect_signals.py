"""Step a: the signals understand already produced, per component (G, O, D, declared).

- G: fault modes on the component that count: a source other than llm_judgement or default, not downgraded
  (their quotes were verified by understand). The others are listed as uncounted.
- O: channels in a counted fault mode's `observable_by` whose role is observation, or that have a secondary
  observation role. A range whose value role is in `observation_limit_roles` is a documented limit.
- D: channels in a counted fault mode's `driven_by` whose role is operating_input (or secondary).
- declared: machine.toml [[components]] declares it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from kaful.artifacts.machine_model import FaultMode, MachineModel
from kaful.artifacts.scope import ChannelLink
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.config import MachineConfig, ScopeSettings

UNCOUNTED_SOURCES = ("llm_judgement", "default")


@dataclass
class Collected:
    fault_modes: list[str] = field(default_factory=list)
    uncounted: list[str] = field(default_factory=list)
    observed: dict[str, ChannelLink] = field(default_factory=dict)
    driven: dict[str, ChannelLink] = field(default_factory=dict)
    declared: bool = False


def counts(mode: FaultMode) -> bool:
    return mode.source not in UNCOUNTED_SOURCES and mode.downgraded_from is None


def _plays(channel: SensorChannel, role: str) -> bool:
    return channel.role == role or any(r.role == role for r in channel.secondary_roles)


def _link(links: dict[str, ChannelLink], channel: SensorChannel, mode: str, limit: bool) -> None:
    old = links.get(channel.name)
    modes = tuple(sorted({mode, *(old.fault_modes if old else ())}))
    links[channel.name] = ChannelLink(channel=channel.name, fault_modes=modes, documented_limit=limit)


def _limit(channel: SensorChannel, roles: Sequence[str]) -> bool:
    ev = channel.range_evidence
    return channel.range is not None and ev is not None and ev.value_role in roles


def collect(model: MachineModel, contract: SensorContract, machine: MachineConfig,
            settings: ScopeSettings) -> dict[str, Collected]:
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
                _link(got.observed, channel, mode.id, _limit(channel, settings.observation_limit_roles))
        for name in mode.driven_by:
            channel = by_name.get(name)
            if channel is not None and _plays(channel, "operating_input"):
                _link(got.driven, channel, mode.id, False)
    declared = {d.id for d in machine.components}
    for cid, got in out.items():
        got.declared = got.declared or cid in declared
    return out
