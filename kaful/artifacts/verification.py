"""Phase 6 (verify) artifact: saved check results. First cut (M11 refines)."""

from typing import Annotated, Literal

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    NonEmptyStr,
    StrictModel,
    sorted_unique,
)


class CheckResult(StrictModel):
    """One check on one target (component id, connection id, or 'composite')."""

    check: NonEmptyStr
    target: NonEmptyStr
    status: Literal["pass", "fail", "warn", "skipped"]
    measured: dict[str, float] = {}
    limits: dict[str, float] = {}
    message: str = ""


class VerificationReport(ArtifactBase):
    """All check results; a failure stops the run unless quarantined or allowed."""

    kind: Literal["verification_report"] = "verification_report"
    results: Annotated[
        tuple[CheckResult, ...], sorted_unique(lambda r: (r.target, r.check), "results")
    ]
    quarantined: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    allowed_failures: bool = False

    @property
    def passed(self) -> bool:
        """No failing check outside quarantine, or failures explicitly allowed."""
        failing = {r.target for r in self.results if r.status == "fail"}
        return self.allowed_failures or failing <= set(self.quarantined)
