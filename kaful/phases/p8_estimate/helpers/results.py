"""What one source produces: per-issue results (built by d_forecast's callers in phase.py) and everything e_write
gathers for the phase to write in source order."""

from dataclasses import dataclass, field
from pathlib import Path

from kaful.artifacts.forecast import EventForecast, Forecast, HealthSummary, ParamSummary, SkippedEstimator


@dataclass(frozen=True)
class IssueResult:
    t_issue: float
    events: list[EventForecast]
    params: list[ParamSummary]
    health: list[HealthSummary]
    samples: str  # CSV text


@dataclass
class SourceResult:
    """Everything one source produced: forecasts (artifact, name under the phase dir), files to write, skips."""

    forecasts: list[tuple[Forecast, str]] = field(default_factory=list)
    files: dict[Path, str] = field(default_factory=dict)
    skipped: list[SkippedEstimator] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
