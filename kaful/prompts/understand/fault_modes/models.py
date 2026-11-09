"""Output model for understand.fault_modes. References to components and channels are
checked by code against the lists the prompt was given; a carrier (v4, M7b) is a part the
documentation says wears that the component list lacks, proposed with its own evidence and
checked by code like any component."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaful.prompts.understand.components.models import ExtractedComponent
from kaful.prompts.understand.quotes import ID_PATTERN, Quote, duplicates


class ExtractedFaultMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    component: str = Field(pattern=ID_PATTERN, description="A given component id, or the id of one of `carriers`.")
    name: str = Field(min_length=1)
    mechanism: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    driven_by: list[str]
    observable_by: list[str]
    evidence: list[Quote] = Field(min_length=1)


class FaultModeExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carriers: list[ExtractedComponent] = Field(description=(
        "Parts the documentation says wear or must be serviced that are not in the component list; "
        "empty when every fault mode sits on a given component."))
    fault_modes: list[ExtractedFaultMode]

    @model_validator(mode="after")
    def _unique(self) -> "FaultModeExtraction":
        ids = [f.id for f in self.fault_modes]
        if duplicates(ids):
            raise ValueError(f"duplicate fault mode ids: {duplicates(ids)}")
        carriers = [c.id for c in self.carriers]
        if duplicates(carriers):
            raise ValueError(f"duplicate carrier ids: {duplicates(carriers)}")
        bare = sorted(set(carriers) - {f.component for f in self.fault_modes})
        if bare:
            raise ValueError(f"carriers with no fault mode: {bare}; give each carrier its fault modes or leave it out")
        return self
