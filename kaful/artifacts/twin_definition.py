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
    Provenanced,
    SourceKind,
    ArtifactBase,
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


CouplingFormName = Literal["hardness_passthrough"]


class Connection(Provenanced):
    """A coupling between two bindings (M13b; kaful.templates.couplings has the forms): the `from_binding`'s damage
    fraction x = clip(D / D_limit, 0, 1) multiplies the `to_binding`'s template rate by the form's m(x).

    Provenance is the link's existence: `manual` when its quote is in the cited page's raw text (checked by code,
    rule 4), else `deployment` (downgraded) for a declaration alone. Its strength is a parameter with its own
    provenance (role `coupling`). Units: x is D / D_limit, so the from-binding's D and D_limit units must agree, and
    every slot parameter is in the slot's dimension (verify, `connection_units`).
    """

    id: Identifier
    from_binding: Identifier
    to_binding: Identifier
    form: CouplingFormName
    form_version: NonEmptyStr
    params: dict[Identifier, NonEmptyStr] = Field(default_factory=dict, description="Form slot -> param_id.")
    basis: Literal["declared"] = Field(description="declared: machine.toml [[connection]] (no flow path found it).")
    justification: NonEmptyStr


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
    trigger: Literal["schedule", "channel", "threshold"] = Field(description=(
        "schedule: every `interval` in a stored profile; channel: when the channel reports the category; "
        "threshold: replace when D reaches D_limit (condition-based; the event still fires)."))
    trigger_channel: Identifier | None = Field(default=None, description="The event channel (schedule: a pseudo channel).")
    trigger_value: NonEmptyStr | None = Field(default=None, description="The channel category, or the rule id.")
    interval: float | None = Field(default=None, gt=0.0, description="Documented interval, in time units.")
    source: SourceKind
    ref: NonEmptyStr


def renewal_state(binding: str) -> str:
    """The state counting a threshold rule's renewals of `binding` (every backend names it so)."""
    return f"{binding}_renewals"


class EventDef(StrictModel):
    """A failure event: the binding's damage reaching its D_limit."""

    name: Identifier
    binding: Identifier


ObservationFormName = Literal["linear", "threshold_event"]
ObservationBasis = Literal["observable_by", "named_by", "declared"]


class ObservationBinding(StrictModel):
    """How one binding's damage shows on one SensorContract channel (M12; kaful.observation has the forms).

    `linear`: the channel reads y_healthy + delta * D / D_limit (D / D_limit clipped to [0, 1]) plus Gaussian noise
    (slots y_healthy, delta, noise_sd). `threshold_event`: an event channel reports `event_value` (None for a bool
    channel: it reports True) at the step where D reaches D_limit, e.g. the eversys W-014 at each filter renewal.
    """

    id: Identifier
    binding: Identifier
    channel: Identifier
    form: ObservationFormName
    form_version: NonEmptyStr
    params: dict[Identifier, NonEmptyStr] = Field(default_factory=dict, description="Form slot -> param_id.")
    event_value: NonEmptyStr | None = None
    basis: ObservationBasis = Field(description=(
        "observable_by: a fault mode names the channel; named_by: scope's channel-name link (M8); declared: "
        "machine.toml [[observation]]."))
    ref: NonEmptyStr = Field(description="The fault mode id, or the machine.toml entry.")
    justification: NonEmptyStr


class TwinDefinition(ArtifactBase):
    """Everything a TwinBackend needs to build the twin, without library specifics.

    Schema 3 (M12, approved 2026-11-14) adds `observations`: the observation models of the twin's bindings.
    Schema 4 (M13b, approved 2026-11-16) gives `connections` their typed form: couplings between bindings.
    """

    schema_version: Literal[4] = 4  # type: ignore[assignment]
    kind: Literal["twin_definition"] = "twin_definition"
    time_unit: NonEmptyStr
    components: Annotated[tuple[Identifier, ...], SORTED_STRS]
    bindings: Annotated[tuple[TemplateBinding, ...], by_attr("id")]
    connections: Annotated[tuple[Connection, ...], by_attr("id")] = ()
    external_inputs: Annotated[tuple[ExternalInput, ...], by_attr("name")] = ()
    events: Annotated[tuple[EventDef, ...], by_attr("name")] = ()
    maintenance: Annotated[tuple[MaintenanceRule, ...], by_attr("id")] = ()
    profiles: Annotated[tuple[OperatingProfile, ...], by_attr("name")] = ()
    observations: Annotated[tuple[ObservationBinding, ...], by_attr("id")] = ()
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
        for obs in self.observations:
            if obs.binding not in binding_ids:
                raise ValueError(f"observation {obs.id!r}: unknown binding {obs.binding!r}")
        for c in self.connections:
            for end in (c.from_binding, c.to_binding):
                if end not in binding_ids:
                    raise ValueError(f"connection {c.id!r}: unknown binding {end!r}")
        return self
