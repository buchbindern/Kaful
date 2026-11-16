"""Phase 8 (estimate) artifacts and the operating profile.

Schema 2 (M13, approved 2026-11-16). One `forecast` per (data source, estimator): the forecasts issued at several
times in the source's span, each with, per event, the censoring counts and percentiles of the next time the part
reaches its limit; posterior summaries of the carried parameters (k as the reported posterior, the others as
nuisance); and health quantiles. Every sample (each event's time, each carried parameter) is in a CSV sidecar,
named with its sha256, so M14 can pair times of event with parameters. A `forecast_set` indexes one estimate run.

All times are in the machine's time_unit and absolute (from t = 0), the horizon included.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
    sorted_unique,
)
from kaful.paths import RelPath


class ProfileSegment(StrictModel):
    """Constant operating conditions over [start, end)."""

    start: float = Field(ge=0.0)
    end: float
    values: dict[Identifier, float] = Field(description="operating_input channel -> value.")

    @model_validator(mode="after")
    def _check_span(self) -> "ProfileSegment":
        if not self.end > self.start:
            raise ValueError("segment end must be > start")
        return self


class ScheduledEvent(StrictModel):
    """A discrete event at time t on an event channel, e.g. a maintenance action."""

    t: float = Field(ge=0.0)
    channel: Identifier
    value: NonEmptyStr = Field(description="One of the channel's categories, e.g. 'descale'.")


class OperatingProfile(StrictModel):
    """Future operating conditions and scheduled events, for forecasts and what-ifs."""

    name: NonEmptyStr
    segments: tuple[ProfileSegment, ...] = Field(min_length=1)
    events: Annotated[
        tuple[ScheduledEvent, ...], sorted_unique(lambda e: (e.t, e.channel), "events")
    ] = ()

    @model_validator(mode="after")
    def _check_contiguous(self) -> "OperatingProfile":
        for before, after in zip(self.segments, self.segments[1:]):
            if before.end != after.start:
                raise ValueError("segments must be contiguous and in time order")
        return self


Percentile = Literal["p05", "p50", "p95"]
EstimatorKind = Literal["particle_filter", "ukf", "open_loop"]


class EventForecast(StrictModel):
    """The next time one binding reaches its limit after t_issue, over every sample, with explicit censoring."""

    event: Identifier
    binding: Identifier
    n_samples: int = Field(ge=1)
    n_hits: int = Field(ge=0, description="Samples reaching the limit before the horizon (already triggered included).")
    n_censored: int = Field(ge=0, description="Samples with no crossing before the horizon.")
    n_already_triggered: int = Field(ge=0, description=(
        "Samples at or past the limit at t_issue with no renewal rule (failed, awaiting maintenance): RUL 0."))
    p_event_before_horizon: Fraction
    percentiles: dict[Percentile, float | None] = Field(description=(
        "Time of event over all samples, a censored one counting as after the horizon; None: past the horizon."))

    @model_validator(mode="after")
    def _check_counts(self) -> "EventForecast":
        if self.n_hits + self.n_censored != self.n_samples:
            raise ValueError("n_hits + n_censored must equal n_samples")
        if self.n_already_triggered > self.n_hits:
            raise ValueError("n_already_triggered cannot exceed n_hits")
        return self


class ParamSummary(StrictModel):
    """A carried parameter's prior and posterior 5/50/95% (the posterior from this source's data)."""

    param_id: NonEmptyStr
    role: Literal["rate", "nuisance"] = Field(description="rate: k, the reported posterior; nuisance: the others.")
    prior: dict[Percentile, float]
    posterior: dict[Percentile, float]


class HealthSummary(StrictModel):
    """1 - D / D_limit at t_issue (clipped to [0, 1]), 5/50/95% over samples."""

    binding: Identifier
    quantiles: dict[Percentile, float]


class Issue(StrictModel):
    """A forecast issued at one time."""

    t_issue: float = Field(ge=0.0)
    events: Annotated[tuple[EventForecast, ...], by_attr("event")]
    params: Annotated[tuple[ParamSummary, ...], by_attr("param_id")] = ()
    health: Annotated[tuple[HealthSummary, ...], by_attr("binding")] = ()
    samples: RelPath = Field(description="CSV sidecar: per sample, each event's time (empty: censored) and each "
                                         "carried parameter; relative to the machine directory.")
    samples_sha256: Sha256


class GroupRecord(StrictModel):
    """What one group's estimator did over the source (kaful.estimation.filter.Diagnostics)."""

    bindings: Annotated[tuple[Identifier, ...], SORTED_STRS]
    estimator: EstimatorKind = Field(description="What actually ran on this group.")
    shared_from_open_loop: bool = Field(default=False, description=(
        "An unobserved group, filtered and forecast once in the open-loop run (identical by construction)."))
    updates: int = Field(ge=0)
    resamples: int = Field(ge=0)
    degenerate_updates: int = Field(ge=0)
    min_ess: float | None = None
    channels_used: dict[str, int] = Field(default_factory=dict)
    reports: dict[str, int] = Field(default_factory=dict)
    unmonitored_events: Annotated[tuple[Identifier, ...], SORTED_STRS] = ()
    note: str = ""


class Forecast(ArtifactBase):
    """RUL forecasts for one data source and one estimator, issued at several times, under one operating profile."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["forecast"] = "forecast"
    source: Identifier = Field(description="The data source, e.g. a synthetic unit id.")
    estimator: EstimatorKind
    seed: int
    spawn_key: int = Field(ge=0)
    horizon: float = Field(gt=0.0)
    process_noise_cv: float = Field(gt=0.0, description="rho: an estimator assumption, not a physical parameter.")
    n_particles: int = Field(ge=1)
    profile: OperatingProfile
    ignored_profile_channels: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = Field(default=(), description=(
        "Profile channels no active condition reads: they change nothing in this twin."))
    groups: tuple[GroupRecord, ...]
    issues: tuple[Issue, ...] = Field(min_length=1)
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()

    @model_validator(mode="after")
    def _check_issues(self) -> "Forecast":
        times = [i.t_issue for i in self.issues]
        if times != sorted(times):
            raise ValueError("issues must be in time order")
        if any(t > self.horizon for t in times):
            raise ValueError("an issue time is past the horizon")
        return self


class ForecastEntry(StrictModel):
    source: Identifier
    estimator: EstimatorKind
    path: RelPath
    sha256: Sha256


class SkippedEstimator(StrictModel):
    source: Identifier
    estimator: EstimatorKind
    reason: NonEmptyStr


class ForecastSet(ArtifactBase):
    """One estimate run: every forecast it wrote, and every estimator it skipped and why."""

    kind: Literal["forecast_set"] = "forecast_set"
    time_unit: NonEmptyStr
    settings: dict[str, float | int | str | list[float] | list[str]] = Field(description=(
        "The estimate settings used (process_noise_cv is an estimator assumption, listed in the flags too)."))
    forecasts: Annotated[tuple[ForecastEntry, ...], sorted_unique(lambda e: (e.source, e.estimator), "forecasts")]
    skipped: Annotated[tuple[SkippedEstimator, ...], sorted_unique(lambda e: (e.source, e.estimator), "skipped")] = ()
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    disclaimer: NonEmptyStr | None = Field(default=None, description="The dataset's, when the data is synthetic.")
