"""Phase 5 (build) artifact: template bindings and composite wiring. First cut (M10 refines).

Each fault mode is a degradation template acting on a physical damage state D (e.g.
mm of wear, crack length, scale thickness). Health is derived as
health = 1 - D / D_limit, where D_limit is the binding's `threshold_param`.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
)
from kaful.paths import RelPath


class GeneratedCode(StrictModel):
    """Flagged escape hatch: free-form code, run only through the sandbox."""

    path: RelPath
    sha256: Sha256
    entrypoint: Identifier
    reason: NonEmptyStr


class TemplateBinding(StrictModel):
    """A fault mode bound to a degradation template and ParamSet entries."""

    id: Identifier
    component: Identifier
    fault_mode: Identifier
    template: Identifier
    template_version: NonEmptyStr
    damage_state: Identifier = Field(description="Name of the physical damage state D.")
    damage_unit: NonEmptyStr = Field(description="Unit of D, e.g. 'mm'.")
    threshold_param: NonEmptyStr = Field(description="param_id of D_limit (role 'threshold').")
    params: dict[Identifier, NonEmptyStr] = Field(description="Template slot -> param_id.")
    conditions: dict[Identifier, NonEmptyStr] = Field(
        default_factory=dict, description="Template condition -> channel or port."
    )
    justification: NonEmptyStr
    escape_hatch: GeneratedCode | None = None


class Connection(StrictModel):
    """One wire between component ports; units are checked for compatibility in verify."""

    id: Identifier
    source_port: NonEmptyStr
    target_port: NonEmptyStr
    source_unit: str
    target_unit: str
    match: Literal["exact", "fuzzy", "manual"]
    score: Fraction


class ExternalInput(StrictModel):
    """An unresolved input, fed from a channel or held at a nominal ParamSet value."""

    name: Identifier
    channel: Identifier | None = None
    nominal_param: NonEmptyStr = Field(description="param_id of the nominal value.")


class EventDef(StrictModel):
    """A failure event: the binding's damage reaching its D_limit."""

    name: Identifier
    binding: Identifier


class TwinDefinition(ArtifactBase):
    """Everything a TwinBackend needs to build the twin, without library specifics."""

    kind: Literal["twin_definition"] = "twin_definition"
    time_unit: NonEmptyStr
    components: Annotated[tuple[Identifier, ...], SORTED_STRS]
    bindings: Annotated[tuple[TemplateBinding, ...], by_attr("id")]
    connections: Annotated[tuple[Connection, ...], by_attr("id")] = ()
    external_inputs: Annotated[tuple[ExternalInput, ...], by_attr("name")] = ()
    events: Annotated[tuple[EventDef, ...], by_attr("name")] = ()

    @model_validator(mode="after")
    def _check_references(self) -> "TwinDefinition":
        binding_ids = {b.id for b in self.bindings}
        for binding in self.bindings:
            if binding.component not in self.components:
                raise ValueError(f"binding {binding.id!r}: unknown component {binding.component!r}")
        for event in self.events:
            if event.binding not in binding_ids:
                raise ValueError(f"event {event.name!r}: unknown binding {event.binding!r}")
        return self
