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

A channel may also play a secondary role (`secondary_roles`, decided 2026-11-06): spindle
load is an operating input (the cut drives it) that also rises with bearing wear. Only
`operating_input` and `observation` combine; events and the usage clock are single-role.
M13 does not use such a channel twice: health is the residual of the measured value
against the value expected at the same operating point, and wear is driven by the
expected value (this needs a healthy reference period or a documented nominal value).

Sampling is explicit: `periodic` (with `sample_period_s`), `event_driven`, or `unknown`
when the documentation does not say. A periodic `manual` channel must carry the quote
that states its period, so a guessed rate never passes for a documented one.

Measurement noise is not stored here: it is a ParamSet parameter with role `noise`.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Evidence,
    Identifier,
    NonEmptyStr,
    Provenanced,
    by_attr,
)

ChannelRole = Literal["operating_input", "observation", "event", "usage_clock"]
DualRole = Literal["operating_input", "observation"]
DUAL_ROLES: tuple[str, ...] = ("operating_input", "observation")
ChannelDtype = Literal["float", "int", "bool", "category"]
Sampling = Literal["periodic", "event_driven", "unknown"]


class SecondaryRole(Provenanced):
    """A second role a channel plays, with its own provenance (a quote when the documentation states it)."""

    role: DualRole
    reason: NonEmptyStr = Field(description="Why the channel also plays this role.")


class SensorChannel(Provenanced):
    """One reported channel. `range` keeps its sign; `unit` is a pint-parseable string."""

    name: Identifier
    description: NonEmptyStr
    role: ChannelRole
    dtype: ChannelDtype
    unit: str | None = Field(default=None, description="None for bool/category, and for a numeric channel whose "
                             "documentation prints no unit (unit_status 'not_printed'); '1' = dimensionless.")
    unit_status: Literal["not_printed"] | None = Field(
        default=None, description="'not_printed': numeric, but the documentation prints no unit; no range is "
        "stored, and the unit is supplied later (M8 or the user).")
    range: tuple[float, float] | None = None
    categories: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] | None = None
    sampling: Sampling
    sample_period_s: float | None = Field(default=None, gt=0.0, description="Set iff sampling is periodic.")
    sample_period_evidence: Evidence | None = Field(
        default=None, description="The quote stating the period; required for a periodic manual channel."
    )
    range_evidence: Evidence | None = Field(
        default=None, description="The quote stating the range, with its value_role; only with a range."
    )
    secondary_roles: Annotated[tuple[SecondaryRole, ...], by_attr("role")] = ()

    @model_validator(mode="after")
    def _check_channel(self) -> "SensorChannel":
        numeric = self.dtype in ("float", "int")
        if self.unit_status == "not_printed":
            if not numeric:
                raise ValueError(f"{self.name}: unit_status applies only to numeric channels")
            if self.unit is not None or self.range is not None:
                raise ValueError(f"{self.name}: unit_status 'not_printed' means no unit and no range")
        elif numeric and not self.unit:
            raise ValueError(f"{self.name}: numeric channels need a unit or unit_status 'not_printed'")
        if not numeric and (self.unit is not None or self.range is not None):
            raise ValueError(f"{self.name}: unit and range apply only to numeric channels")
        if (self.dtype == "category") != (self.categories is not None):
            raise ValueError(f"{self.name}: categories are required iff dtype is 'category'")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError(f"{self.name}: range low must be <= high")
        if self.role == "event":
            if self.dtype not in ("category", "bool"):
                raise ValueError(f"{self.name}: event channels must be 'category' or 'bool'")
            if self.sampling != "event_driven":
                raise ValueError(f"{self.name}: event channels are event-driven")
        if self.role == "usage_clock" and not numeric:
            raise ValueError(f"{self.name}: the usage clock must be numeric")
        if (self.sampling == "periodic") != (self.sample_period_s is not None):
            raise ValueError(f"{self.name}: sample_period_s is set iff sampling is 'periodic'")
        if self.sampling == "periodic" and self.source == "manual" and self.sample_period_evidence is None:
            raise ValueError(f"{self.name}: a documented (manual) period needs sample_period_evidence")
        if self.sampling != "periodic" and self.sample_period_evidence is not None:
            raise ValueError(f"{self.name}: sample_period_evidence applies only to periodic sampling")
        if self.range is None and self.range_evidence is not None:
            raise ValueError(f"{self.name}: range_evidence applies only to a stored range")
        if self.secondary_roles and self.role not in DUAL_ROLES:
            raise ValueError(f"{self.name}: only an operating_input or observation channel has a secondary role")
        if any(r.role == self.role for r in self.secondary_roles):
            raise ValueError(f"{self.name}: a secondary role must differ from its role")
        for ev in (self.range_evidence, self.sample_period_evidence):
            if ev is not None and ev.value_role == "example":
                raise ValueError(f"{self.name}: an example value is never stored as a range or period")
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
