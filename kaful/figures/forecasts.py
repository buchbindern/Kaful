"""Forecast samples and traces for figures (no truth here). Used by f2_tracking and f3_failure_time.

- `quantity_bindings`: the bindings a quantity covers (an event's own, or a first_failure group's);
- `issue_rul`: one issue's sampled RUL for a quantity (time of event - t_issue; +inf censored), sha-checked;
- `rul_quantiles`: the fan's percentiles over those samples (inverted CDF, as calibration: never interpolated
  through infinity);
- `read_trace`: a forecast's trace CSV as columns of floats.
"""

import re

import numpy as np

from kaful.artifacts.forecast import Forecast, Issue
from kaful.figures.sources import LoadedLineage
from kaful.runs import ArtifactMismatch, sha256_file
from kaful.scoring.samples import read_table, toe_samples

FIRST_FAILURE = re.compile(r"^first_failure\((?P<bindings>[a-z0-9_, ]+)\)$")
FAN_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def quantity_bindings(quantity: str, issue: Issue) -> tuple[str, ...]:
    m = FIRST_FAILURE.match(quantity)
    if m:
        return tuple(b.strip() for b in m.group("bindings").split(","))
    found = [e.binding for e in issue.events if e.event == quantity]
    if not found:
        raise ValueError(f"no event {quantity!r} in the forecast")
    return (found[0],)


def issue_rul(ll: LoadedLineage, issue: Issue, quantity: str) -> np.ndarray:
    path = ll.file(issue.samples)
    if sha256_file(path) != issue.samples_sha256:
        raise ArtifactMismatch(f"{issue.samples}: content does not match the forecast's samples_sha256")
    rows = read_table(path)
    bindings = quantity_bindings(quantity, issue)
    toes = [toe_samples(rows, e.event) for e in issue.events if e.binding in bindings]
    return np.asarray(np.min(np.stack(toes), axis=0) - issue.t_issue, dtype=float)


def rul_quantiles(rul: np.ndarray) -> tuple[float, ...]:
    """The fan's percentiles; +inf where the quantile lies past the horizon (censored)."""
    return tuple(float(np.quantile(rul, q, method="inverted_cdf")) for q in FAN_QUANTILES)


def read_trace(ll: LoadedLineage, f: Forecast) -> dict[str, np.ndarray]:
    path = ll.file(f.trace)
    if sha256_file(path) != f.trace_sha256:
        raise ArtifactMismatch(f"{f.trace}: content does not match the forecast's trace_sha256")
    rows = read_table(path)
    cols = list(rows[0]) if rows else []
    return {c: np.array([float(r[c]) if r[c] != "" else np.nan for r in rows]) for c in cols}
