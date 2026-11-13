"""Phase 6 (verify) artifact: saved check results, the simulations they ran on, and the quarantine.

Schema 2 (M11, approved 2026-11-13): every check is a result per target (a component, or a twin-level target
such as a connection, an input or 'composite'). A failing component is quarantined: verify writes a reduced
TwinDefinition without it and re-runs every check on that twin (`recheck`). The report passes when every
failing target is quarantined and the recheck has no failure, or when failures were explicitly allowed.
"""

from typing import Annotated, Literal

from pydantic import Field

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    StrictModel,
    by_attr,
    sorted_unique,
)

CheckStatus = Literal["pass", "fail", "warn", "skipped"]


class CheckResult(StrictModel):
    """One check on one target (component id, connection id, or 'composite')."""

    check: NonEmptyStr
    target: NonEmptyStr
    status: CheckStatus
    measured: dict[str, float] = {}
    limits: dict[str, float] = {}
    message: str = ""


RESULTS = sorted_unique(lambda r: (r.target, r.check), "results")


class SimulationRun(StrictModel):
    """One simulation the dynamic checks ran on: a stored profile with every parameter at one prior quantile."""

    name: NonEmptyStr = Field(description="'<profile>@q<quantile>', e.g. 'nominal@q0.5'.")
    profile: NonEmptyStr
    quantile: Fraction
    dt: float = Field(gt=0.0, description="Step, in the twin's time unit.")
    horizon: float = Field(gt=0.0)
    steps: int = Field(ge=1)


class Quarantine(StrictModel):
    """A component left out of the verified twin, and why."""

    component: Identifier
    checks: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS]
    reason: NonEmptyStr
    dependents: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="Components quarantined because they depend on this one (none until M13).")


class VerificationReport(ArtifactBase):
    """All check results; a failure stops the run unless quarantined or allowed."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["verification_report"] = "verification_report"
    results: Annotated[tuple[CheckResult, ...], RESULTS]
    runs: Annotated[tuple[SimulationRun, ...], by_attr("name")] = ()
    quarantined: Annotated[tuple[Quarantine, ...], by_attr("component")] = ()
    recheck: Annotated[tuple[CheckResult, ...], RESULTS] = Field(
        default=(), description="The checks re-run on the reduced twin (empty when nothing was quarantined).")
    allowed_failures: bool = False

    @property
    def quarantined_components(self) -> tuple[str, ...]:
        return tuple(q.component for q in self.quarantined)

    @property
    def passed(self) -> bool:
        """No failing check outside quarantine and none on the reduced twin, or failures explicitly allowed."""
        failing = {r.target for r in self.results if r.status == "fail"}
        rechecked = all(r.status != "fail" for r in self.recheck)
        return self.allowed_failures or (failing <= set(self.quarantined_components) and rechecked)
