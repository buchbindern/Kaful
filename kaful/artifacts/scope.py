"""Phase 3 (scope) artifact: which components get full models."""

from typing import Annotated, Literal

from kaful.artifacts.base import ArtifactBase, Identifier, NonEmptyStr, StrictModel, by_attr


class ScopeDecision(StrictModel):
    """The decision for one component, with its reason."""

    component: Identifier
    decision: Literal["full_model", "connection_only", "excluded"]
    reason: NonEmptyStr
    decided_by: Literal["llm", "rule", "override"]


class Scope(ArtifactBase):
    """One decision per MachineModel component, made before per-component work."""

    kind: Literal["scope"] = "scope"
    decisions: Annotated[tuple[ScopeDecision, ...], by_attr("component")]

    def in_scope(self) -> tuple[str, ...]:
        """Components that get a full model, sorted."""
        return tuple(d.component for d in self.decisions if d.decision == "full_model")
