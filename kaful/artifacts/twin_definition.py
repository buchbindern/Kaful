"""Phase 5 (build) artifact: template bindings, wiring, maintenance and profiles.

Schema 2 (M10, approved 2026-11-10): one binding per full-model component (one template per component, M9),
acting on a physical damage state D in `damage_unit`; health = 1 - D / D_limit is derived, and the event fires
when D reaches D_limit (the binding's `threshold_param`, role threshold or capacity). Every template
condition is fed by a connection or is an ExternalInput with provenance (a channel, a nominal parameter, or
undocumented: f = 1). Maintenance rules reset D (replace) or reduce it (clean, descale) when a scheduled
event arrives; the stored profiles are the nominal one and the documented-maintenance one.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    SourceKind,
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
)
from kaful.artifacts.forecast import OperatingProfile
from kaful.paths import RelPath


class GeneratedCode(StrictModel):
    """Flagged escape hatch: free-form code, run only through the sandbox."""

    path: RelPath
    sha256: Sha256
    entrypoint: Identifier
    reason: NonEmptyStr


class TemplateBinding(StrictModel):
    """A component bound to its degradation template and ParamSet entries."""

    id: Identifier
    component: Identifier
    fault_mode: Identifier | None = Field(description="The fault mode whose words chose the family (M9).")
    other_modes: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="Counted fault modes this binding does not model separately.")
    template: Identifier
    template_version: NonEmptyStr
    damage_state: Identifier = Field(description="Name of the physical damage state D.")
    damage_unit: NonEmptyStr = Field(description="Unit of D, e.g. 'mm'.")
    threshold_param: NonEmptyStr = Field(description="param_id of D_limit (role threshold or capacity).")
    params: dict[Identifier, NonEmptyStr] = Field(description="Template slot -> param_id.")
    conditions: dict[Identifier, NonEmptyStr] = Field(
        default_factory=dict, description="Template condition -> connection id or external input name."
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
    """A template condition no connection feeds: from a channel, held at a nominal value, or undocumented."""

    name: Identifier
    binding: Identifier | None = None
    condition: Identifier | None = None
    unit: str = Field(default="", description="The channel's unit, or the nominal parameter's.")
    channel: Identifier | None = None
    nominal_param: NonEmptyStr | None = Field(default=None, description="param_id of the nominal value.")
    provenance: Literal["channel", "nominal_param", "undocumented"] = "nominal_param"
    note: str = ""


class MaintenanceRule(StrictModel):
    """A maintenance action on one binding: replace sets D = 0; clean and descale set D *= 1 - recovery."""

    id: Identifier
    binding: Identifier
    action: Literal["replace", "clean", "descale"]
    recovery_param: NonEmptyStr | None = Field(default=None, description="param_id of the recovery (clean, descale).")
    trigger_channel: Identifier = Field(description="The event channel that carries it ('maintenance' for a schedule).")
    trigger_value: NonEmptyStr = Field(description="The channel category, or the rule id for a schedule.")
    interval: float | None = Field(default=None, gt=0.0, description="Documented interval, in time units.")
    source: SourceKind
    ref: NonEmptyStr


class EventDef(StrictModel):
    """A failure event: the binding's damage reaching its D_limit."""

    name: Identifier
    binding: Identifier


class TwinDefinition(ArtifactBase):
    """Everything a TwinBackend needs to build the twin, without library specifics."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["twin_definition"] = "twin_definition"
    time_unit: NonEmptyStr
    components: Annotated[tuple[Identifier, ...], SORTED_STRS]
    bindings: Annotated[tuple[TemplateBinding, ...], by_attr("id")]
    connections: Annotated[tuple[Connection, ...], by_attr("id")] = ()
    external_inputs: Annotated[tuple[ExternalInput, ...], by_attr("name")] = ()
    events: Annotated[tuple[EventDef, ...], by_attr("name")] = ()
    maintenance: Annotated[tuple[MaintenanceRule, ...], by_attr("id")] = ()
    profiles: Annotated[tuple[OperatingProfile, ...], by_attr("name")] = ()
    flags: dict[Identifier, tuple[NonEmptyStr, ...]] = Field(default_factory=dict, description=(
        "Per binding: what the review should know (undocumented inputs, maintenance gaps)."))

    @model_validator(mode="after")
    def _check_references(self) -> "TwinDefinition":
        binding_ids = {b.id for b in self.bindings}
        for rule in self.maintenance:
            if rule.binding not in binding_ids:
                raise ValueError(f"maintenance {rule.id!r}: unknown binding {rule.binding!r}")
        for binding in self.bindings:
            if binding.component not in self.components:
                raise ValueError(f"binding {binding.id!r}: unknown component {binding.component!r}")
        for event in self.events:
            if event.binding not in binding_ids:
                raise ValueError(f"event {event.name!r}: unknown binding {event.binding!r}")
        return self
