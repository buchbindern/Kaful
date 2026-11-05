"""Phase 2 (understand) artifact: components, flow paths and fault modes.

Components form a tree through `parent` (e.g. an impeller inside a pump). Understand
records every documented part; scope (M8) decides which become full models. The time
unit is not stored here: it is declared in machine.toml and confirmed in M9.
"""

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
    parent: Identifier | None = Field(default=None, description="The component this one is part of.")


class FlowPath(Provenanced):
    """Something that passes from one component (or boundary) to another."""

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
    components: Annotated[tuple[Component, ...], by_attr("id")]
    boundaries: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="External sources and sinks flow paths may start or end at (e.g. supply_inlet)."
    )
    flow_paths: Annotated[tuple[FlowPath, ...], by_attr("id")] = ()
    fault_modes: Annotated[tuple[FaultMode, ...], by_attr("id")] = ()
    n_extractions: int = Field(ge=1, description="Self-consistency N; agreement feeds confidence.")

    @model_validator(mode="after")
    def _check_references(self) -> "MachineModel":
        ids = {c.id for c in self.components}
        parents = {c.id: c.parent for c in self.components}
        for component in self.components:
            if component.parent is not None and component.parent not in ids:
                raise ValueError(f"component {component.id!r}: unknown parent {component.parent!r}")
            seen, node = {component.id}, component.parent
            while node is not None:
                if node in seen:
                    raise ValueError(f"component {component.id!r}: parent chain has a cycle")
                seen.add(node)
                node = parents[node]
        if ids & set(self.boundaries):
            raise ValueError(f"boundaries reuse component ids: {sorted(ids & set(self.boundaries))}")
        for path in self.flow_paths:
            for end in (path.from_component, path.to_component):
                if end not in ids and end not in self.boundaries:
                    raise ValueError(f"flow path {path.id!r}: unknown component {end!r}")
        for mode in self.fault_modes:
            if mode.component not in ids:
                raise ValueError(f"fault mode {mode.id!r}: unknown component {mode.component!r}")
        return self
