"""The run manifest: everything needed to know what a run did and reproduce it.

It is rewritten after every phase, so a crashed run still shows how far it got.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    ArtifactRef,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
    sorted_unique,
)
from kaful.paths import RelPath
from kaful.phases import PHASE_ORDER, PhaseName

PhaseStatus = Literal[
    "pending",
    "reused",
    "skipped",
    "passed",
    "failed",
    "quarantined",
    "not_implemented",
    "budget_exceeded",
]
RunStatus = Literal["running", "passed", "failed", "budget_exceeded", "aborted", "not_implemented"]


class CodeVersion(StrictModel):
    git_commit: NonEmptyStr
    git_dirty: bool
    kaful_version: NonEmptyStr


class InputFile(StrictModel):
    """A machine input (machine.toml or a document), relative to the machine directory."""

    path: RelPath
    sha256: Sha256


class Override(StrictModel):
    """A recorded deviation from the default behaviour (hard rule 2)."""

    kind: Literal["allow_failures", "base_run", "scope", "param", "other"]
    target: NonEmptyStr
    value: str
    reason: NonEmptyStr
    recorded_at: datetime


class PhaseRecord(StrictModel):
    """Status, artifacts and cost of one phase in this run."""

    phase: PhaseName
    status: PhaseStatus = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifacts: Annotated[
        tuple[ArtifactRef, ...], sorted_unique(lambda r: (r.kind, r.path), "artifacts")
    ] = ()
    cost_usd: float = Field(default=0.0, ge=0.0)
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = Field(default=(), description="Facts the run report must show (e.g. reused responses).")


class BaseRun(StrictModel):
    """The earlier run whose artifacts this run reuses for phases before `--from`."""

    run_id: NonEmptyStr
    selected_by: Literal["auto", "explicit"]
    inputs_match: bool


class RunManifest(StrictModel):
    """Inputs, code version, config, overrides, cost and per-phase status of one run."""

    schema_version: Literal[1] = 1
    kind: Literal["run_manifest"] = "run_manifest"
    run_id: NonEmptyStr
    machine_id: Identifier
    created_at: datetime
    finished_at: datetime | None = None
    code: CodeVersion
    argv: tuple[str, ...]
    from_phase: PhaseName | None = None
    only: tuple[PhaseName, ...] = ()
    base_run: BaseRun | None = None
    budget_usd: float = Field(gt=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    allow_failures: bool = False
    strict: bool = Field(default=False, description="--strict: index-hash mismatches and unlisted documents are errors.")
    config: dict[str, Any]
    config_hash: Sha256
    inputs: Annotated[tuple[InputFile, ...], by_attr("path")]
    prompt_versions: dict[str, str] = {}
    index_hash: str | None = None
    python: NonEmptyStr
    pythonhashseed: str | None = None
    overrides: tuple[Override, ...] = ()
    phases: tuple[PhaseRecord, ...]
    status: RunStatus = "running"

    @model_validator(mode="after")
    def _check_phases(self) -> "RunManifest":
        if tuple(r.phase for r in self.phases) != PHASE_ORDER:
            raise ValueError("phases must list every phase once, in pipeline order")
        return self

    def phase(self, name: PhaseName) -> PhaseRecord:
        return self.phases[PHASE_ORDER.index(name)]
