"""Phase 2 (understand) artifact: the single definition of what the machine reports.

The synthesizer, the observation models and every DataSource read this contract.

Channel roles:
- `operating_input`: a condition that drives degradation (load, speed, temperature).
- `observation`: a reading that reflects health.
- `event`: a discrete happening, such as a maintenance action. An event channel has
  dtype `category` (its categories are the action names, e.g. "descale", "replace")
  or `bool`, and is event-driven (`sample_period_s` is None). Templates react to
  maintenance events through `DegradationTemplate.apply_maintenance`.
- `usage_clock`: elapsed usage in the machine's `time_unit` (e.g. a brew-cycle
  counter or operating hours). `Observation.t` comes from this channel.

Measurement noise is not stored here: it is a ParamSet parameter with role `noise`.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Identifier,
    NonEmptyStr,
    Provenanced,
    by_attr,
)

ChannelRole = Literal["operating_input", "observation", "event", "usage_clock"]
ChannelDtype = Literal["float", "int", "bool", "category"]


class SensorChannel(Provenanced):
    """One reported channel. `range` keeps its sign; `unit` is a pint-parseable string."""

    name: Identifier
    description: NonEmptyStr
    role: ChannelRole
    dtype: ChannelDtype
    unit: str | None = Field(default=None, description="None only for bool/category; '1' = dimensionless.")
    range: tuple[float, float] | None = None
    categories: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] | None = None
    sample_period_s: float | None = Field(default=None, gt=0.0, description="None = event-driven.")

    @model_validator(mode="after")
    def _check_channel(self) -> "SensorChannel":
        numeric = self.dtype in ("float", "int")
        if numeric and not self.unit:
            raise ValueError(f"{self.name}: numeric channels need a unit ('1' if dimensionless)")
        if not numeric and (self.unit is not None or self.range is not None):
            raise ValueError(f"{self.name}: unit and range apply only to numeric channels")
        if (self.dtype == "category") != (self.categories is not None):
            raise ValueError(f"{self.name}: categories are required iff dtype is 'category'")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError(f"{self.name}: range low must be <= high")
        if self.role == "event":
            if self.dtype not in ("category", "bool"):
                raise ValueError(f"{self.name}: event channels must be 'category' or 'bool'")
            if self.sample_period_s is not None:
                raise ValueError(f"{self.name}: event channels are event-driven")
        if self.role == "usage_clock" and not numeric:
            raise ValueError(f"{self.name}: the usage clock must be numeric")
        return self


class SensorContract(ArtifactBase):
    """Channels, units, ranges and sample rates the machine reports."""

    kind: Literal["sensor_contract"] = "sensor_contract"
    time_column: Identifier = "timestamp"
    usage_clock: Identifier | None = Field(
        default=None, description="Channel giving elapsed usage in the machine's time_unit."
    )
    channels: Annotated[tuple[SensorChannel, ...], by_attr("name")]

    @model_validator(mode="after")
    def _check_contract(self) -> "SensorContract":
        clocks = [c.name for c in self.channels if c.role == "usage_clock"]
        if len(clocks) > 1:
            raise ValueError(f"at most one usage_clock channel, got {clocks}")
        if self.usage_clock != (clocks[0] if clocks else None):
            raise ValueError("usage_clock must name the channel whose role is 'usage_clock'")
        if any(c.name == self.time_column for c in self.channels):
            raise ValueError(f"time_column {self.time_column!r} clashes with a channel name")
        return self

    def channel(self, name: str) -> SensorChannel:
        """Return the channel called `name`; KeyError if absent."""
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise KeyError(name)
