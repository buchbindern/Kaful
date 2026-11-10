"""Phase 3 (scope) artifact: which components get full models, and the signals each decision rests on.

Schema 2 (M8, approved 2026-11-09) stores the signals:
- G `fault_modes`: documented gradual fault modes (a verified quote; not downgraded, not llm_judgement);
- O `observed_by`: observation channels (or ones with a secondary observation role) that a counted
  fault mode is observable by; `documented_limit` when the channel's range is a spec or setting range;
- O `named_by` (kept apart, lower confidence; approved 2026-11-09): observation channels whose own name or
  description names the component as the thing measured ("Coffee boiler temperature"), when no fault mode
  links them;
- D `driven_by`: operating-input channels that drive a counted fault mode;
- L `life`: a stated life or interval naming the component in a machine document, a life rating or life
  model on a reference-document page for it, or a life declared in machine.toml.
Machine-wide service intervals (a life sentence that names no component) are kept once per scope, linked
to the components whose fault modes cite that page, for M9.
"""

from typing import Annotated, Literal

from pydantic import Field

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Evidence,
    Fraction,
    Identifier,
    NonEmptyStr,
    StrictModel,
    by_attr,
    sorted_unique,
)

Decision = Literal["full_model", "connection_only", "excluded"]
LifeBasis = Literal["stated", "life_model", "declared"]


class ChannelLink(StrictModel):
    """A channel that observes (O) or drives (D) some of the component's counted fault modes."""

    channel: Identifier
    fault_modes: Annotated[tuple[Identifier, ...], SORTED_STRS]
    documented_limit: bool = Field(default=False, description="The channel's range is a spec or setting range.")
    confidence: Fraction = Field(description="Computed by code: the lower of the channel's and its best fault mode's.")


class NamedChannel(StrictModel):
    """An observation channel whose own name or description names the component; no fault mode links it."""

    channel: Identifier
    named_in: Literal["name", "description"]
    text: NonEmptyStr = Field(description="The channel name or description that names the component.")
    documented_limit: bool = False
    confidence: Fraction = Field(description="Computed by code: the channel's confidence times named_channel_factor.")


class LifeEvidence(StrictModel):
    """One life signal (L): its basis, where it is, and the quote (none for a machine.toml declaration)."""

    basis: LifeBasis
    ref: NonEmptyStr = Field(description="e.g. 'nsk_bearing_selection_mounting p51 (label 50)' or 'machine.toml [[life]]'.")
    evidence: Evidence | None = None
    note: str = Field(default="", description="Why it counts, e.g. which name the page prints.")
    phrase: str = Field(default="", description="life_model: the life-rating phrase the quote has, e.g. 'L10'.")


def _life_key(e: LifeEvidence) -> tuple[str, str, str]:
    return e.basis, e.ref, e.evidence.quote if e.evidence else ""


class ScopeSignals(StrictModel):
    fault_modes: Annotated[tuple[Identifier, ...], SORTED_STRS] = ()
    uncounted_fault_modes: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="Fault modes on the component that do not count for G (downgraded, llm_judgement).")
    observed_by: Annotated[tuple[ChannelLink, ...], by_attr("channel")] = ()
    named_by: Annotated[tuple[NamedChannel, ...], by_attr("channel")] = ()
    driven_by: Annotated[tuple[ChannelLink, ...], by_attr("channel")] = ()
    life: Annotated[tuple[LifeEvidence, ...], sorted_unique(_life_key, "life evidence")] = ()
    declared: bool = Field(default=False, description="machine.toml declares the component.")


class ScopeDecision(StrictModel):
    """The decision for one component, with its reason, the signals, and flags for the onboarding review."""

    component: Identifier
    decision: Decision
    reason: NonEmptyStr
    decided_by: Literal["llm", "rule", "override"]
    signals: ScopeSignals = ScopeSignals()
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()


class MachineLifeStatement(StrictModel):
    """A service interval or life that names no component (eversys: "12 months or every 50,000 products")."""

    evidence: Evidence
    linked: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="Components whose fault-mode evidence cites the same page.")


def _statement_key(s: MachineLifeStatement) -> tuple[str, int, str]:
    return s.evidence.doc_id, s.evidence.page, s.evidence.quote


class Scope(ArtifactBase):
    """One decision per MachineModel component, made before per-component work."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["scope"] = "scope"
    decisions: Annotated[tuple[ScopeDecision, ...], by_attr("component")]
    machine_life: Annotated[tuple[MachineLifeStatement, ...], sorted_unique(_statement_key, "statement")] = ()

    def in_scope(self) -> tuple[str, ...]:
        """Components that get a full model, sorted."""
        return tuple(d.component for d in self.decisions if d.decision == "full_model")
