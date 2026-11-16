"""Step e: one source's forecast artifacts and sidecar files, ready to write.

Per estimator: `samples/<source>.<estimator>.<issue>.csv` (every sample of each issue), `traces/<source>.<estimator>.csv`
(the per-row trace), and the `forecast` artifact `forecasts/<source>.<estimator>.json`, which names each sidecar with
its sha256. The phase writes them in source order, so the outputs never depend on how many workers ran.
"""

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaful.artifacts.forecast import (
    EventForecast,
    Forecast,
    GroupRecord,
    HealthSummary,
    Issue,
    OperatingProfile,
    ParamSummary,
    SkippedEstimator,
)
from kaful.config import EstimateSettings, EstimatorName
from kaful.paths import to_rel
from kaful.runs import sha256_bytes

TraceRow = dict[str, float]


@dataclass
class SourceResult:
    """Everything one source produced: forecasts (artifact, name under the phase dir), files to write, skips."""

    forecasts: list[tuple[Forecast, str]] = field(default_factory=list)
    files: dict[Path, str] = field(default_factory=dict)
    skipped: list[SkippedEstimator] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


def trace_csv(rows: Sequence[TraceRow]) -> str:
    names = sorted({k for r in rows for k in r} - {"t"})
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["t", *names])
    for r in rows:
        writer.writerow([repr(float(r["t"])), *(repr(float(r[n])) if n in r else "" for n in names)])
    return buf.getvalue()


def group_record(f: Any, estimator: EstimatorName, shared: bool) -> GroupRecord:
    d = f.diagnostics
    ran: EstimatorName = "open_loop" if (d.note or not f.group.observed or shared) else estimator
    return GroupRecord(bindings=f.group.bindings, estimator=ran, shared_from_open_loop=shared, updates=d.updates,
                       resamples=getattr(d, "resamples", 0), degenerate_updates=getattr(d, "degenerate", 0),
                       min_ess=None if d.min_ess == float("inf") else float(d.min_ess),
                       channels_used=dict(sorted(d.channels_used.items())), reports=dict(sorted(d.reports.items())),
                       unmonitored_events=d.unmonitored_events, note=d.note)


@dataclass(frozen=True)
class IssueResult:
    t_issue: float
    events: list[EventForecast]
    params: list[ParamSummary]
    health: list[HealthSummary]
    samples: str  # CSV text


def build(base: Mapping[str, Any], source: str, spawn_key: int, estimator: EstimatorName, settings: EstimateSettings,
          profile: OperatingProfile, horizon: float, ignored: Sequence[str], groups: Sequence[GroupRecord],
          issues: Sequence[IssueResult], trace: str, flags: Sequence[str], machine_dir: Path,
          phase_dir: Path, out: SourceResult) -> None:
    built_issues = []
    for i, issue in enumerate(issues):
        path = phase_dir / "samples" / f"{source}.{estimator}.{i}.csv"
        out.files[path] = issue.samples
        built_issues.append(Issue(t_issue=issue.t_issue, events=tuple(issue.events), params=tuple(issue.params),
                                  health=tuple(issue.health), samples=to_rel(machine_dir, path),
                                  samples_sha256=sha256_bytes(issue.samples.encode())))
    trace_path = phase_dir / "traces" / f"{source}.{estimator}.csv"
    out.files[trace_path] = trace
    forecast = Forecast(**base, source=source, estimator=estimator, seed=settings.seed, spawn_key=spawn_key,
                        horizon=horizon, process_noise_cv=settings.process_noise_cv,
                        n_particles=settings.n_particles, profile=profile, ignored_profile_channels=tuple(ignored),
                        groups=tuple(groups), issues=tuple(built_issues), trace=to_rel(machine_dir, trace_path),
                        trace_sha256=sha256_bytes(trace.encode()), flags=tuple(sorted(set(flags))))
    out.forecasts.append((forecast, f"forecasts/{source}.{estimator}.json"))
