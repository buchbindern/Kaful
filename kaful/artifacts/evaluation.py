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
- calibration: 90% interval coverage on synthetic truth with explicit censoring, alpha-lambda accuracy, prognostic
  horizon, RMSE of the median RUL; the 85-95% target is judged on observed parts of the primary estimator only, and
  parts with no observation of their own are reported apart, labelled circular (decided 2026-11-17);
- sensitivity: which priors dominate RUL uncertainty (M9's computation on the lineage's ParamSet), how far the data
  narrowed each carried parameter, and which data would narrow each part most (rules in settings.evaluate).
The fixed disclaimers (rule 5) are part of the report.
"""

from typing import Annotated, Literal, get_args

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
    lock: Literal["locked", "equivalent_to_lock", "older_than_lock", "not_locked"] = Field(description=(
        "equivalent_to_lock (M15): the reuse rule serves this version's normally ended replies under the lock, which "
        "differs from it only in max_tokens"))
    locked_version: int | None = None
    run_scores: tuple[RunScore, ...] = Field(default=(), description="The per-run scores the lock records (locked, "
                                             "equivalent_to_lock) or the saved eval results of the version used "
                                             "(older_than_lock).")


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
    unbilled_usd: float = Field(ge=0.0, description=(
        "What the cache hits would have cost at synchronous prices (the ledger's unbilled)."))
    unbilled_if_batched_usd: float = Field(ge=0.0, description=(
        "The same cache hits with every LLM call through the Batch API (pricing.toml batch multiplier): a cold run "
        "costs between billed + this and billed + unbilled, depending on what it batches."))

    @property
    def cold_usd(self) -> float:
        """Upper bound: every cached call at synchronous prices."""
        return self.billed_usd + self.unbilled_usd

    @property
    def cold_low_usd(self) -> float:
        """Lower bound: every cached LLM call batched."""
        return self.billed_usd + self.unbilled_if_batched_usd


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

    @property
    def cold_low_usd(self) -> float:
        return sum(p.cold_low_usd for p in self.by_phase)


# ── Calibration ──────────────────────────────────────────────────────────────

# observed: the part has an observation of its own; joined: in an observed group only through a connection (its
# posterior equals its prior); unobserved: no observation reaches it; not_identifiable: one of several parts on one
# channel, judged instead through a label-invariant quantity (the first failure of the group).
PartClass = Literal["observed", "joined", "unobserved", "not_identifiable"]
# Per (unit, issue): covered / missed (truth before the horizon); censored_covered / censored_missed (no true crossing
# before the horizon: the interval must reach past it); already_triggered (the part is at its limit at the issue time,
# no renewal: counted apart); undecidable (the truth file stops before the forecast horizon).
PairClass = Literal["covered", "missed", "censored_covered", "censored_missed", "already_triggered", "undecidable"]
PAIR_CLASSES: tuple[PairClass, ...] = get_args(PairClass)
CIRCULAR: frozenset[str] = frozenset({"joined", "unobserved"})


class Bounds(StrictModel):
    low: Fraction
    high: Fraction


class CoverageCell(StrictModel):
    """Coverage of the 90% interval for one quantity and estimator, at one issue fraction (None: all pooled)."""

    estimator: NonEmptyStr
    quantity: NonEmptyStr = Field(description="An event, or 'first_failure(<a>, <b>)' for a label-invariant group.")
    part_class: PartClass
    issue_fraction: float | None
    classes: dict[PairClass, int] = Field(description="Every class, zeros included.")
    pairs: int = Field(ge=0, description="Scored (unit, issue) pairs: covered, missed and both censored classes.")
    covered: int = Field(ge=0)
    units: int = Field(ge=0, description="Units with at least one scored pair.")
    triggered_agree: int = Field(default=0, ge=0, description=(
        "Already-triggered pairs where the forecast agreed: at least half its samples at or before the issue time."))
    share: Fraction | None = None
    ci_pairs: Bounds | None = Field(default=None, description="Wilson 95% on the pairs (treats them as independent).")
    ci_units: Bounds | None = Field(default=None, description=(
        "Wilson 95% with the units as the sample size: pairs within a unit are correlated, so this is the honest one."))


class AccuracyScore(StrictModel):
    """Point accuracy of one quantity and estimator, over every (unit, issue) with a true crossing ahead."""

    estimator: NonEmptyStr
    quantity: NonEmptyStr
    part_class: PartClass
    alpha_lambda_pass: int = Field(ge=0)
    alpha_lambda_n: int = Field(ge=0)
    alpha_lambda: Fraction | None = None
    alpha_mass_median: Fraction | None = Field(default=None, description=(
        "Median over pairs of the RUL mass inside (1 +- alpha) x the true RUL (the pass mark is beta)."))
    horizon_units: int = Field(ge=0, description="Units whose prognostic horizon is defined (a true crossing ahead).")
    horizon_median: float | None = Field(default=None, description="Median prognostic horizon, in the time unit.")
    horizon_share_median: Fraction | None = Field(default=None, description=(
        "Median of PH / (true crossing - first issue time): the share of the forecastable span with alpha-lambda held."))
    rmse: float | None = Field(default=None, ge=0.0, description="RMSE of the median RUL, in the time unit.")
    rmse_relative: float | None = Field(default=None, ge=0.0, description="RMSE of (median - true) / true RUL.")
    rmse_n: int = Field(ge=0)
    rmse_excluded: int = Field(ge=0, description="Pairs with a true crossing ahead whose median RUL is censored.")
    abs_relative_error_median: float | None = Field(default=None, ge=0.0, description=(
        "Median |median RUL - true RUL| / true RUL: robust to the near-crossing pairs that dominate the RMSE."))
    abs_log_error_median: float | None = Field(default=None, ge=0.0, description=(
        "Median |log(median RUL / true RUL)| over pairs whose median RUL is positive."))
    width_relative_median: float | None = Field(default=None, ge=0.0, description=(
        "Sharpness: median of the 90% interval's width / true RUL (None when that median is censored)."))
    width_censored: int = Field(default=0, ge=0, description="Pairs whose 95% point is past the horizon (open width).")
    width_ratio_open_loop: float | None = Field(default=None, ge=0.0, description=(
        "Median, over the same (unit, issue) pairs with both finite, of this estimator's 90% width / open-loop's: "
        "how much the data narrowed the forecast (1: not at all)."))


class Verdict(StrictModel):
    """The calibration target, judged on one observed quantity of the primary estimator (pooled over issues)."""

    quantity: NonEmptyStr
    coverage: Fraction
    units: int = Field(ge=0)
    ci_units: Bounds
    verdict: Literal["within", "above", "below"]
    ci_overlaps_target: bool


class Calibration(StrictModel):
    forecast_set: ArtifactRef
    dataset: ArtifactRef
    target: tuple[float, float] = CALIBRATION_TARGET_90
    interval: tuple[float, float]
    alpha: float
    beta: float
    primary_estimator: NonEmptyStr
    process_noise_cv: float = Field(description="rho: an estimator assumption; not tuned on synthetic truth.")
    cells: Annotated[tuple[CoverageCell, ...], sorted_unique(
        lambda c: (c.estimator, c.quantity, -1.0 if c.issue_fraction is None else c.issue_fraction), "cells")]
    accuracy: Annotated[tuple[AccuracyScore, ...], sorted_unique(lambda a: (a.estimator, a.quantity), "accuracy")]
    verdicts: Annotated[tuple[Verdict, ...], by_attr("quantity")]
    notes: tuple[NonEmptyStr, ...] = ()
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()


# ── Sensitivity ──────────────────────────────────────────────────────────────

class PriorEffect(StrictModel):
    """One prior's first-order share of var(log RUL) at t = 0 (kaful.sensitivity, as M9's report)."""

    param: NonEmptyStr
    source: NonEmptyStr
    role: NonEmptyStr
    share: Fraction
    low_median: float = Field(description="Median RUL with the prior at its 16th percentile.")
    high_median: float = Field(description="Median RUL with the prior at its 84th percentile.")


class PosteriorParam(StrictModel):
    """What one carried parameter looks like after the data (primary estimator, last issue, median over units)."""

    param: NonEmptyStr
    contraction: float | None = Field(default=None, ge=0.0, description=(
        "Posterior 90% width / prior 90% width (on the log scale for a positive parameter): 1 = the data did not "
        "narrow it."))
    remaining_share: Fraction | None = Field(default=None, description=(
        "Share of the remaining log time-to-limit spread it explains (correlation ratio over quantile bins)."))
    units: int = Field(ge=0)


class ComponentSensitivity(StrictModel):
    component: NonEmptyStr
    part_class: PartClass
    life_source: NonEmptyStr
    sigma_log_rul: float = Field(ge=0.0, description="sd of log RUL at t = 0 from the priors.")
    effects: tuple[PriorEffect, ...] = Field(description="Most variance first.")
    posterior: Annotated[tuple[PosteriorParam, ...], by_attr("param")] = ()
    life_contraction: float | None = Field(default=None, ge=0.0, description=(
        "Posterior 90% width of its life D_limit / k (the identifiable quantity; M13) / its life_nominal prior's, log "
        "scale, median over units (None: no samples carry k)."))
    remaining_spread: float = Field(ge=0.0, description="sigma_log_rul x life contraction: what the data left open.")
    recommendations: tuple[NonEmptyStr, ...] = ()


class Sensitivity(StrictModel):
    param_set: ArtifactRef
    samples: int = Field(ge=1)
    seed: int
    components: tuple[ComponentSensitivity, ...] = Field(description="Most remaining spread first.")
    variance_by_source: dict[str, Fraction] = Field(description=(
        "Mean over components of each source's part of the first-order shares (normalised per component)."))
    estimator_assumptions: tuple[NonEmptyStr, ...] = ()
    notes: tuple[NonEmptyStr, ...] = ()


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
    calibration: Calibration | None = None
    sensitivity: Sensitivity | None = None
    cost: Cost
    not_reached: Annotated[tuple[NotReached, ...], by_attr("section")] = ()
    summary: tuple[NonEmptyStr, ...] = Field(default=(), description="Computed from the sections (helpers/summary).")
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    disclaimers: tuple[NonEmptyStr, ...] = EVALUATION_DISCLAIMERS

    @model_validator(mode="after")
    def _check(self) -> "EvaluationReport":
        missing = {n.section for n in self.not_reached}
        for section in ("grounding", "twin_quality", "calibration", "sensitivity"):
            if (getattr(self, section) is None) != (section in missing):
                raise ValueError(f"{section}: absent exactly when listed in not_reached")
        if self.complete == bool(self.not_reached):
            raise ValueError("complete must be True exactly when nothing is in not_reached")
        if self.disclaimers != EVALUATION_DISCLAIMERS:
            raise ValueError("the disclaimers are fixed text (rule 5)")
        return self
