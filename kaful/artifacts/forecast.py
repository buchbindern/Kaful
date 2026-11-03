"""Phase 8 (estimate) artifact and the operating profile. First cut (M13 refines).

All times are in the machine's time_unit.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    StrictModel,
    by_attr,
    sorted_unique,
)


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


class EventForecast(StrictModel):
    """Time-of-event samples for one event, with explicit censoring."""

    event: Identifier
    n_samples: int = Field(ge=1)
    toe_samples: tuple[float, ...] = Field(description="Every uncensored sample, in sample order.")
    n_censored: int = Field(ge=0, description="Samples with no event before the horizon.")
    horizon: float = Field(gt=0.0)
    p_event_before_horizon: Fraction
    percentiles: dict[Percentile, float | None]
    already_triggered_fraction: Fraction

    @model_validator(mode="after")
    def _check_counts(self) -> "EventForecast":
        if len(self.toe_samples) + self.n_censored != self.n_samples:
            raise ValueError("len(toe_samples) + n_censored must equal n_samples")
        return self


class Forecast(ArtifactBase):
    """RUL forecast for one data source under one operating profile."""

    kind: Literal["forecast"] = "forecast"
    source: NonEmptyStr
    estimator: Literal["particle_filter", "ukf", "open_loop"]
    seed: int
    t_issue: float
    profile: OperatingProfile
    events: Annotated[tuple[EventForecast, ...], by_attr("event")]
