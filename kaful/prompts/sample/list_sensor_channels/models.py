"""Output model for sample.list_sensor_channels (M5 sample; M7 owns the real sensor contract)."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SensorChannel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="Channel name as the document prints it.")
    unit: str = Field(min_length=1, description="Unit as printed, e.g. 'bar', 'm3/h', 'degC'.")
    range_min: float | None = Field(description="Lower end of the stated measuring range; null if none is stated.")
    range_max: float | None = Field(description="Upper end of the stated measuring range; null if none is stated.")
    evidence: str = Field(min_length=1, description="The document line that lists this channel, copied exactly.")
    page: int = Field(ge=1, description="Page number from the '--- page N ---' marker above the evidence.")


class SensorChannelList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    channels: list[SensorChannel] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> "SensorChannelList":
        names = [c.name for c in self.channels]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate channel names: {duplicates}")
        for c in self.channels:
            if c.range_min is not None and c.range_max is not None and c.range_min > c.range_max:
                raise ValueError(f"{c.name}: range_min > range_max")
        return self
