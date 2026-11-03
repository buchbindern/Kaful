"""Phase 2 (understand) artifact: components, flow paths and fault modes."""

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


class Component(Provenanced):
    """A physical part of the machine."""

    id: Identifier
    name: NonEmptyStr
    description: str
    category: NonEmptyStr


class FlowPath(Provenanced):
    """Something that passes from one component to another."""

    id: Identifier
    from_component: Identifier
    to_component: Identifier
    medium: Literal["fluid", "electrical", "mechanical", "thermal", "signal"]
    quantity: str | None = None
    unit: str | None = None


class FaultMode(Provenanced):
    """A way a component degrades. Channel names refer to the SensorContract."""

    id: Identifier
    component: Identifier
    name: NonEmptyStr
    mechanism: NonEmptyStr = Field(description="e.g. wear, fatigue, scaling, contamination.")
    effect: str
    driven_by: Annotated[tuple[Identifier, ...], SORTED_STRS] = ()
    observable_by: Annotated[tuple[Identifier, ...], SORTED_STRS] = ()


class MachineModel(ArtifactBase):
    """What the machine is made of and how it fails, with evidence."""

    kind: Literal["machine_model"] = "machine_model"
    time_unit: NonEmptyStr = Field(description="The one declared time unit (hard rule 6).")
    components: Annotated[tuple[Component, ...], by_attr("id")]
    flow_paths: Annotated[tuple[FlowPath, ...], by_attr("id")] = ()
    fault_modes: Annotated[tuple[FaultMode, ...], by_attr("id")] = ()
    n_extractions: int = Field(ge=1, description="Self-consistency N; agreement feeds confidence.")

    @model_validator(mode="after")
    def _check_references(self) -> "MachineModel":
        ids = {c.id for c in self.components}
        for path in self.flow_paths:
            for end in (path.from_component, path.to_component):
                if end not in ids:
                    raise ValueError(f"flow path {path.id!r}: unknown component {end!r}")
        for mode in self.fault_modes:
            if mode.component not in ids:
                raise ValueError(f"fault mode {mode.id!r}: unknown component {mode.component!r}")
        return self
