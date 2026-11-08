"""Output model for understand.sensor_channels: the channels the machine reports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaful.artifacts.base import ValueRole
from kaful.prompts.understand.quotes import ID_PATTERN, Quote, duplicates


class StatedPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(gt=0.0)
    unit: str = Field(min_length=1, description="Time unit as printed, e.g. 's', 'seconds', 'min'.")
    evidence: Quote
    role: ValueRole = Field(description="What the stated period is (spec, variant, default, setting_range, example).")


class ExtractedChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=ID_PATTERN)
    printed_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: Literal["observation", "operating_input", "usage_clock", "event"]
    secondary_role: Literal["observation", "operating_input"] | None = Field(
        description="Only for an operating_input that also reflects condition, or an observation that also drives wear.")
    secondary_reason: str | None = Field(description="One sentence: why; given iff secondary_role is.")
    secondary_evidence: Quote | None = Field(description="The line stating the second role, if the documentation does.")
    dtype: Literal["float", "int", "bool", "category"]
    unit: str | None = Field(description="Unit as printed; null for bool and category channels.")
    range_min: float | None
    range_max: float | None
    range_evidence: Quote | None = Field(description="The line stating the range; null without a range.")
    range_role: ValueRole | None = Field(description="What the range is; null without a range.")
    categories: list[str] | None = Field(description="Only for category channels: the values as printed.")
    sample_period: StatedPeriod | None = Field(description="Only when the documentation states it.")
    evidence: list[Quote] = Field(min_length=1)


class ChannelExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channels: list[ExtractedChannel]

    @model_validator(mode="after")
    def _consistent(self) -> "ChannelExtraction":
        names = [c.name for c in self.channels]
        if duplicates(names):
            raise ValueError(f"duplicate channel names: {duplicates(names)}")
        clocks = [c.name for c in self.channels if c.role == "usage_clock"]
        if len(clocks) > 1:
            raise ValueError(f"at most one usage_clock channel, got {clocks}")
        for c in self.channels:
            if c.range_min is not None and c.range_max is not None and c.range_min > c.range_max:
                raise ValueError(f"{c.name}: range_min > range_max")
            has_range = c.range_min is not None or c.range_max is not None
            if has_range != (c.range_evidence is not None) or has_range != (c.range_role is not None):
                raise ValueError(f"{c.name}: range_evidence and range_role are given iff a range is")
            if c.secondary_role is not None and (c.role not in ("observation", "operating_input")
                                                 or c.secondary_role == c.role):
                raise ValueError(f"{c.name}: secondary_role is only for an operating_input or observation, "
                                 "and differs from role")
            if (c.secondary_role is None) != (not c.secondary_reason):
                raise ValueError(f"{c.name}: secondary_reason is given iff secondary_role is")
            if c.secondary_role is None and c.secondary_evidence is not None:
                raise ValueError(f"{c.name}: secondary_evidence needs a secondary_role")
            if c.dtype == "category" and not c.categories:
                raise ValueError(f"{c.name}: category channels need categories")
            # categories on another dtype are dropped (and noted) by d_build_artifacts, not retried (2026-11-08)
            if c.dtype in ("float", "int") and not c.unit:
                raise ValueError(f"{c.name}: numeric channels need a unit ('1' if dimensionless)")
        return self
