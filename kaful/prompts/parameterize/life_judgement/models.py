"""Output model for parameterize.life_judgement. No confidence field (hard rule 15): code computes it."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LifeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: str = Field(min_length=1, description="The given time unit, or the unit of an amount processed.")
    low: float = Field(gt=0, description="5th percentile of the life, in `unit`.")
    high: float = Field(gt=0, description="95th percentile of the life, in `unit`.")

    @model_validator(mode="after")
    def _ordered(self) -> "LifeRange":
        if not self.low < self.high:
            raise ValueError("low must be below high")
        return self


class LifeJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    life: LifeRange
    basis: str = Field(min_length=1)
