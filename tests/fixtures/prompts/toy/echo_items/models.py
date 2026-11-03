"""Output model for the toy.echo_items test prompt."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Items(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> "Items":
        if len(set(self.items)) != len(self.items):
            raise ValueError("items must be unique")
        return self
