"""Output model for understand.components: components (a tree), boundaries and flow paths."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaful.prompts.understand.quotes import ID_PATTERN, Quote, duplicates


class ExtractedComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, description="The component's name as printed.")
    description: str = Field(description="What it does, in one sentence, from the documentation.")
    category: str = Field(min_length=1)
    parent: str | None = Field(description="Id of the component this one is part of; null for a top-level component.")
    evidence: list[Quote] = Field(min_length=1)


class ExtractedFlowPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_node: str = Field(pattern=ID_PATTERN, description="A component id or boundary id.")
    to_node: str = Field(pattern=ID_PATTERN, description="A component id or boundary id.")
    medium: Literal["fluid", "electrical", "mechanical", "thermal", "signal"]
    quantity: str | None = Field(description="What passes (e.g. 'water', 'torque'); null if not stated.")
    evidence: list[Quote] = Field(min_length=1)


class ComponentExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    components: list[ExtractedComponent] = Field(min_length=1)
    boundaries: list[str] = Field(description="Ids of where something enters or leaves the machine.")
    flow_paths: list[ExtractedFlowPath]

    @model_validator(mode="after")
    def _consistent(self) -> "ComponentExtraction":
        ids = [c.id for c in self.components]
        if duplicates(ids) or duplicates(self.boundaries):
            raise ValueError(f"duplicate ids: {duplicates(ids) + duplicates(self.boundaries)}")
        clash = sorted(set(ids) & set(self.boundaries))
        if clash:
            raise ValueError(f"ids used as both component and boundary: {clash}")
        bad_parents = sorted({c.parent for c in self.components if c.parent is not None and c.parent not in ids})
        if bad_parents:
            raise ValueError(f"parents that are not components: {bad_parents}")
        nodes = set(ids) | set(self.boundaries)
        unknown = sorted({n for f in self.flow_paths for n in (f.from_node, f.to_node) if n not in nodes})
        if unknown:
            raise ValueError(f"flow path ends that are neither components nor boundaries: {unknown}")
        return self
