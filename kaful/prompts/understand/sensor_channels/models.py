"""Output model for understand.sensor_channels: the channels the machine reports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaful.prompts.understand.quotes import ID_PATTERN, Quote, duplicates


class StatedPeriod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float = Field(gt=0.0)
    unit: str = Field(min_length=1, description="Time unit as printed, e.g. 's', 'seconds', 'min'.")
    evidence: Quote


class ExtractedChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=ID_PATTERN)
    printed_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: Literal["observation", "operating_input", "usage_clock", "event"]
    dtype: Literal["float", "int", "bool", "category"]
    unit: str | None = Field(description="Unit as printed; null for bool and category channels.")
    range_min: float | None
    range_max: float | None
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
            if (c.dtype == "category") != (c.categories is not None):
                raise ValueError(f"{c.name}: categories are given iff dtype is 'category'")
            if c.dtype in ("float", "int") and not c.unit:
                raise ValueError(f"{c.name}: numeric channels need a unit ('1' if dimensionless)")
        return self
