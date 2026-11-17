"""Phase 9 (evaluate) artifact: one evaluation report per run.

Schema 2 (M14, approved 2026-11-17). Everything is read from saved artifacts and files, never recomputed upstream:
each section names the files its numbers come from (`FileRef`, path relative to the repository or the machine
directory, with its sha256). A section whose phase the lineage never reached is listed in `not_reached` with the
reason, never left silently empty (CDV: understand failed its gate in M7c).

Sections:
- lineage: per phase, its status and the run that produced its artifacts;
- grounding: evidence quotes by verification class, code downgrades, parameters by source and role, ingest page
  coverage (the M6 quote set);
- prompts: per prompt version the lineage used, its lock and the per-run eval scores the lock rests on;
- twin quality: the verification results against the M11 legacy baseline;
- cost: per phase through the lineage (billed, unbilled = what cache hits would have cost, cold = billed + unbilled),
  the machine's whole development spend, and the M1 legacy baseline where one exists;
- calibration (M14 checkpoint 2) and sensitivity (checkpoint 3).
The fixed disclaimers (rule 5) are part of the report.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    ArtifactRef,
    EvidenceClass,
    Fraction,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
    sorted_unique,
)
from kaful.phases import PHASE_ORDER, PhaseName

EVALUATION_DISCLAIMERS: tuple[str, ...] = (
    "Absolute rates are checked for consistency with the documentation and physics, not for accuracy: without real "
    "failure data, no number here says how long a real part will last.",
    "Calibration is scored on synthetic data generated from the twin's own ParamSet: a pipeline sanity check, partly "
    "circular, not validation.",
    "Coverage of parts with no observation of their own (unobserved, or whose posterior equals the prior) is the "
    "prior predictive scored against truth drawn from that prior: circular by construction, reported separately.",
)

CALIBRATION_TARGET_90: tuple[float, float] = (0.85, 0.95)  # PLAN M14: 90% intervals hold the truth 85-95% of the time

Section = Literal["grounding", "prompts", "twin_quality", "calibration", "sensitivity", "cost"]


class FileRef(StrictModel):
    """A file a number came from: repository-relative (`repo:`) or machine-relative, with its sha256."""

    path: NonEmptyStr
    sha256: Sha256


class PhaseLineage(StrictModel):
    """One phase of the evaluated run: its status there, and the run that produced its artifacts."""

    phase: PhaseName
    status: NonEmptyStr = Field(description="The phase's status in the evaluated run (reused, passed, failed, ...).")
    produced_by: str | None = Field(description="Run id of its artifacts; None: no artifact (never reached).")
    artifacts: Annotated[tuple[ArtifactRef, ...], sorted_unique(lambda r: (r.kind, r.path), "artifacts")] = ()
    failures: tuple[str, ...] = ()


class NotReached(StrictModel):
    section: Section
    reason: NonEmptyStr


# ── Grounding ────────────────────────────────────────────────────────────────

class EvidenceCount(StrictModel):
    """Evidence quotes in one artifact by verification class, and the provenance code downgraded (rule 4)."""

    kind: NonEmptyStr
    quotes: dict[EvidenceClass, int] = Field(description="Every class, zeros included.")
    provenanced: int = Field(ge=0, description="Items carrying a source (parameters, connections, statements, ...).")
    downgraded: dict[str, int] = Field(default_factory=dict, description="'<claimed> -> <now>' -> count.")


class SourceRoleCount(StrictModel):
    source: NonEmptyStr
    role: NonEmptyStr
    count: int = Field(ge=1)


class PageCoverage(StrictModel):
    """Ingest grounding on the fixed M6 quote set, for the index this lineage used."""

    report: FileRef
    matcher_version: NonEmptyStr
    n_quotes: int = Field(ge=0)
    verbatim: int = Field(ge=0)
    near: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    page_share: Fraction = Field(description="Pages with text in at least one chunk / pages with text.")


class Grounding(StrictModel):
    evidence: Annotated[tuple[EvidenceCount, ...], by_attr("kind")]
    params_by_source: dict[str, int] = Field(description="Every source kind, zeros included (pipeline order).")
    params_by_source_role: Annotated[tuple[SourceRoleCount, ...],
                                     sorted_unique(lambda c: (c.source, c.role), "params_by_source_role")] = ()
    life_statements_by_source: dict[str, int] = Field(default_factory=dict, description=(
        "Life and interval statements behind calibrated lives, by source: a calibrated life on a `deployment` "
        "statement (declared in machine.toml) is not documented."))
    declared_params: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = Field(default=(), description=(
        "Parameters whose source is `deployment` (declared in machine.toml): not documented, listed by name."))
    page_coverage: PageCoverage | None = None
    notes: tuple[str, ...] = ()


# ── Prompts ──────────────────────────────────────────────────────────────────

class RunScore(StrictModel):
    result: NonEmptyStr
    sample: int = Field(ge=0)
    metrics: dict[str, float]


class PromptUse(StrictModel):
    """A prompt version the lineage called, and the evals behind it."""

    prompt_id: NonEmptyStr
    version: int = Field(ge=1)
    phase: NonEmptyStr
    models: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS]
    calls: int = Field(ge=0)
    lock: Literal["locked", "older_than_lock", "not_locked"]
    locked_version: int | None = None
    run_scores: tuple[RunScore, ...] = Field(default=(), description="The per-run scores the lock records (locked) "
                                             "or the saved eval results of the version used (older_than_lock).")


class MachineScore(StrictModel):
    """A machine-specific score of a lineage artifact (e.g. understand against its golden labels)."""

    tool: NonEmptyStr
    file: FileRef
    metrics: dict[str, float]
    note: str = ""


class Prompts(StrictModel):
    uses: Annotated[tuple[PromptUse, ...], sorted_unique(lambda u: (u.prompt_id, u.version), "uses")]
    machine_scores: Annotated[tuple[MachineScore, ...], sorted_unique(lambda s: (s.tool, s.file.path), "scores")] = ()
    notes: tuple[str, ...] = ()


# ── Twin quality ─────────────────────────────────────────────────────────────

class LegacyBaselineRef(StrictModel):
    file: FileRef
    components: int = Field(ge=0)
    dynamic_pass_rate: Fraction
    full_pass_rate: Fraction


class TwinQuality(StrictModel):
    """The lineage's verification report, per component, against the M11 legacy baseline (same definitions)."""

    report: ArtifactRef
    components: int = Field(ge=0, description="Components the checks ran on (before quarantine).")
    dynamic_pass_rate: Fraction | None = Field(description=(
        "Components with no failing dynamic check (every check but life and provenance): the M11 comparison."))
    full_pass_rate: Fraction | None = Field(description="Components with no failing check.")
    passed: bool
    quarantined: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    failing_checks: dict[str, int] = Field(default_factory=dict, description="check -> failing targets (before "
                                           "quarantine).")
    legacy: LegacyBaselineRef | None = None


# ── Cost ─────────────────────────────────────────────────────────────────────

class PhaseCost(StrictModel):
    phase: PhaseName
    produced_by: str | None
    calls: int = Field(ge=0, description="Ledger rows of this phase in the producing run (LLM, extraction, embedding).")
    billed_usd: float = Field(ge=0.0)
    unbilled_usd: float = Field(ge=0.0, description="What the cache hits would have cost (the ledger's unbilled).")

    @property
    def cold_usd(self) -> float:
        return self.billed_usd + self.unbilled_usd


class M1Comparison(StrictModel):
    """The legacy cold-run cost (docs/m1_cost_baseline.md), its steps mapped onto the new phases."""

    source: FileRef
    cold_total_usd: float = Field(ge=0.0)
    rerun_total_usd: float = Field(ge=0.0, description="Legacy `--from phase2` rerun (no ingest).")
    by_phase_usd: dict[str, float] = Field(description="New phase -> the summed legacy steps mapped onto it.")
    mapping: dict[str, NonEmptyStr] = Field(description="Legacy step -> new phase (settings.evaluate).")


class Cost(StrictModel):
    by_phase: Annotated[tuple[PhaseCost, ...], sorted_unique(lambda c: PHASE_ORDER.index(c.phase), "by_phase")]
    ledgers: Annotated[tuple[FileRef, ...], sorted_unique(lambda f: f.path, "ledgers")]
    development_billed_usd: float = Field(ge=0.0, description="Billed over every run of this machine (all ledgers).")
    development_runs: int = Field(ge=0, description="Runs with a ledger (the others made no API call).")
    runs_total: int = Field(ge=0, description="Every run directory of the machine.")
    m1: M1Comparison | None = None
    notes: tuple[str, ...] = ()

    @property
    def billed_usd(self) -> float:
        return sum(p.billed_usd for p in self.by_phase)

    @property
    def cold_usd(self) -> float:
        return sum(p.cold_usd for p in self.by_phase)


# ── Report ───────────────────────────────────────────────────────────────────

class EvaluationReport(ArtifactBase):
    """Grounding, prompt quality, twin quality, calibration, sensitivity and cost of one lineage."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["evaluation_report"] = "evaluation_report"
    evaluated_run: NonEmptyStr = Field(description="The run whose lineage is evaluated.")
    complete: bool = Field(description="Every section reached (False: see not_reached).")
    time_unit: str | None = None
    lineage: tuple[PhaseLineage, ...]
    grounding: Grounding | None = None
    prompts: Prompts
    twin_quality: TwinQuality | None = None
    cost: Cost
    not_reached: Annotated[tuple[NotReached, ...], by_attr("section")] = ()
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    disclaimers: tuple[NonEmptyStr, ...] = EVALUATION_DISCLAIMERS

    @model_validator(mode="after")
    def _check(self) -> "EvaluationReport":
        missing = {n.section for n in self.not_reached}
        for section in ("grounding", "twin_quality"):
            if (getattr(self, section) is None) != (section in missing):
                raise ValueError(f"{section}: absent exactly when listed in not_reached")
        if self.complete == bool(self.not_reached):
            raise ValueError("complete must be True exactly when nothing is in not_reached")
        if self.disclaimers != EVALUATION_DISCLAIMERS:
            raise ValueError("the disclaimers are fixed text (rule 5)")
        return self
