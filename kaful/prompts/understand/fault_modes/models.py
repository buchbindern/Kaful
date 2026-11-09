"""Output model for understand.fault_modes. References to components and channels are
checked by code against the lists the prompt was given."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaful.prompts.understand.quotes import ID_PATTERN, Quote, duplicates


class ExtractedFaultMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    component: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    driven_by: list[str]
    observable_by: list[str]
    evidence: list[Quote] = Field(min_length=1)


class FaultModeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_modes: list[ExtractedFaultMode]

    @model_validator(mode="after")
    def _unique(self) -> "FaultModeExtraction":
        ids = [f.id for f in self.fault_modes]
        if duplicates(ids):
            raise ValueError(f"duplicate fault mode ids: {duplicates(ids)}")
        return self
