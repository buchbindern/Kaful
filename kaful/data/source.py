"""The DataSource interface and observation checking against the SensorContract."""

from collections.abc import Iterator
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from kaful.artifacts.base import NonEmptyStr, StrictModel
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract

ChannelValue = float | int | bool | str | None


class Observation(StrictModel):
    """One timestamped reading.

    `t` is elapsed usage in the machine's time_unit, taken from the contract's usage
    clock (or derived from timestamps when time_unit is wall-clock time). `values`
    maps channel name to value; a channel that did not report is absent or None.
    """

    t: float
    timestamp: datetime | None = None
    values: dict[str, ChannelValue]


class ContractViolation(StrictModel):
    """One way an observation does not conform to the SensorContract."""

    channel: NonEmptyStr
    problem: Literal["unknown_channel", "wrong_dtype", "out_of_range", "unknown_category"]
    detail: str


@runtime_checkable
class DataSource(Protocol):
    """A stream of observations that conform to a SensorContract.

    FileSource (CSV/parquet) and future stream sources implement this same interface.
    Consumers process one observation at a time, so batch and streaming share one
    code path.
    """

    @property
    def contract(self) -> SensorContract:
        """The SensorContract every yielded observation conforms to."""
        ...

    def __iter__(self) -> Iterator[Observation]:
        """Yield observations in non-decreasing `t`.

        Channels outside the contract are never yielded. Out-of-range values are
        yielded as read and reported by `check_observation`.
        """
        ...

    def close(self) -> None:
        """Release files or connections. Safe to call more than once."""
        ...


def _dtype_ok(channel: SensorChannel, value: ChannelValue) -> bool:
    if channel.dtype == "bool":
        return isinstance(value, bool)
    if channel.dtype == "category":
        return isinstance(value, str)
    if isinstance(value, bool):
        return False
    if channel.dtype == "int":
        return isinstance(value, int)
    return isinstance(value, (int, float))


def check_observation(contract: SensorContract, obs: Observation) -> tuple[ContractViolation, ...]:
    """Every contract violation in `obs`, sorted by (channel, problem)."""
    known = {c.name: c for c in contract.channels}
    found: list[ContractViolation] = []
    for name, value in obs.values.items():
        channel = known.get(name)
        if channel is None:
            found.append(ContractViolation(channel=name, problem="unknown_channel", detail=""))
            continue
        if value is None:
            continue
        if not _dtype_ok(channel, value):
            found.append(ContractViolation(
                channel=name, problem="wrong_dtype", detail=f"expected {channel.dtype}, got {value!r}"
            ))
        elif channel.categories is not None and value not in channel.categories:
            found.append(ContractViolation(channel=name, problem="unknown_category", detail=repr(value)))
        elif channel.range is not None and isinstance(value, (int, float)):
            low, high = channel.range
            if not low <= value <= high:
                found.append(ContractViolation(
                    channel=name, problem="out_of_range", detail=f"{value} not in [{low}, {high}]"
                ))
    return tuple(sorted(found, key=lambda v: (v.channel, v.problem)))
