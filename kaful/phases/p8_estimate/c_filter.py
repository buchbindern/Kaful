"""Step c: feed one source's rows to an estimator one at a time; keep a trace and the beliefs at the issue times.

Issue times: for each settings.issue_fractions f, the last row at or before f x the time of the last row (a fraction
that falls on the same row as another gives one issue). The trace has, per row, each binding's health (mean and
5/95%) and each linear channel's expected reading (mean): what M14 and the scoring tool compare with truth.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import Estimator, StateBelief
from kaful.data.source import Observation
from kaful.estimation.summaries import channel_readings, health

TraceRow = dict[str, float]


@dataclass(frozen=True)
class Filtered:
    beliefs: tuple[StateBelief, ...]  # at each issue time
    trace: tuple[TraceRow, ...]


def issue_rows(times: Sequence[float], fractions: Sequence[float]) -> tuple[int, ...]:
    """Row index of each issue time, distinct and increasing."""
    if not times:
        return ()
    end, tol = times[-1], 1e-9 * max(1.0, abs(times[-1]))
    chosen = {max((i for i, t in enumerate(times) if t <= f * end + tol), default=0) for f in fractions}
    return tuple(sorted(chosen))


def _trace_row(twin: TwinDefinition, belief: StateBelief, fixed: dict[str, float]) -> TraceRow:
    row: TraceRow = {"t": belief.t}
    for b in sorted(twin.bindings, key=lambda b: b.id):
        if b.damage_state not in belief.particles:
            continue
        h = health(belief, b, fixed)
        row[f"{b.id}.health"] = float(np.mean(h))
        row[f"{b.id}.health_p05"] = float(np.quantile(h, 0.05, method="inverted_cdf"))
        row[f"{b.id}.health_p95"] = float(np.quantile(h, 0.95, method="inverted_cdf"))
    readings: dict[str, FloatArray] = channel_readings(twin, belief, fixed)
    for ch, y in readings.items():
        row[f"{ch}.expected"] = float(np.mean(y))
    return row


def run_filter(estimator: Estimator, rows: Sequence[Observation], issues: Sequence[int], twin: TwinDefinition,
               fixed: dict[str, float]) -> Filtered:
    beliefs, trace = [], []
    wanted = set(issues)
    for i, row in enumerate(rows):
        belief = estimator.update(row)
        trace.append(_trace_row(twin, belief, fixed))
        if i in wanted:
            beliefs.append(belief)
    return Filtered(beliefs=tuple(beliefs), trace=tuple(trace))
