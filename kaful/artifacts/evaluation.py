"""Phase 9 (evaluate) artifact: one evaluation report per run. First cut (M14 refines)."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    ArtifactBase,
    Fraction,
    NonEmptyStr,
    StrictModel,
    by_attr,
    sorted_unique,
)

CALIBRATION_TARGET_90: tuple[float, float] = (0.85, 0.95)


class PromptScore(StrictModel):
    """One eval metric for one prompt version."""

    prompt_id: NonEmptyStr
    version: NonEmptyStr
    metric: NonEmptyStr
    value: float


class CalibrationMetrics(StrictModel):
    """Forecast calibration against synthetic truth (no legacy baseline exists)."""

    coverage_90: Fraction = Field(description="Share of true failure times inside 90% intervals.")
    coverage_target: tuple[float, float] = CALIBRATION_TARGET_90
    alpha_lambda: Fraction | None = None
    prognostic_horizon: float | None = None
    rmse_median: float | None = Field(default=None, ge=0.0)

    @property
    def meets_target(self) -> bool:
        low, high = self.coverage_target
        return low <= self.coverage_90 <= high


class SensitivityEntry(StrictModel):
    """Share of RUL variance attributed to one parameter prior."""

    param_id: NonEmptyStr
    share: Fraction


class EvaluationReport(ArtifactBase):
    """Grounding, prompt quality, twin quality, calibration, sensitivity and cost."""

    kind: Literal["evaluation_report"] = "evaluation_report"
    grounding: dict[str, int] = Field(description="Evidence verification class -> count.")
    params_by_source: dict[str, int]
    prompt_scores: Annotated[
        tuple[PromptScore, ...],
        sorted_unique(lambda s: (s.prompt_id, s.version, s.metric), "prompt_scores"),
    ] = ()
    verification_pass_rate: Fraction
    calibration: CalibrationMetrics | None = None
    sensitivity: Annotated[tuple[SensitivityEntry, ...], by_attr("param_id")] = ()
    cost_by_phase: dict[str, float]
    disclaimer: NonEmptyStr = Field(
        description="States that rates are checked for consistency, not accuracy (rule 5)."
    )

    @model_validator(mode="after")
    def _check_counts(self) -> "EvaluationReport":
        for name, counts in (("grounding", self.grounding), ("params_by_source", self.params_by_source)):
            if any(v < 0 for v in counts.values()):
                raise ValueError(f"{name} counts must be >= 0")
        return self
