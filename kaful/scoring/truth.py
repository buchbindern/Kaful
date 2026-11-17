"""Synthetic truth, read for scoring (M13's scorer, M14's calibration): the only shared code that opens truth files.

- `load_units`: each unit's truth artifact and true damage states;
- `true_next_event`: the true next time a part reaches its limit after t_issue (the forecast's event definition);
- `toe_samples` / `interval`: a forecast issue's sampled times of event (censored as +inf) and their 5-95% interval
  as sample quantiles (inverted CDF, the forecast's own percentiles), never interpolated through infinity;
- `shared_groups` / `joined_bindings`: the label-invariant pairs (one channel, several parts) and the parts joined to
  an observed group only through a connection.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.estimation.groups import groups
from kaful.paths import resolve_in

Table = list[dict[str, str]]
INTERVAL = (0.05, 0.95)


@dataclass(frozen=True)
class Unit:
    truth: SyntheticTruth
    states: Table
    t_end: float


def read_table(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_units(machine_dir: Path, dataset: SyntheticDataset) -> dict[str, Unit]:
    out = {}
    for u in dataset.units:
        truth = SyntheticTruth.model_validate_json(resolve_in(machine_dir, u.truth.path).read_text())
        out[u.unit_id] = Unit(truth=truth, states=read_table(resolve_in(machine_dir, u.truth_states.path)),
                              t_end=u.t_end)
    return out


def true_next_event(unit: Unit, event: str, damage_state: str, limit_param: str, t_issue: float,
                    renews: bool = False) -> float | None:
    """The true next time the part reaches its limit after t_issue: t_issue itself when it is already at or past
    its limit then (failed, no renewal rule), as the forecast's "already triggered"; None: not before the horizon."""
    limit_times = next(e for e in unit.truth.events if e.event == event).limit_times
    if not renews:
        before = [r for r in unit.states if float(r["t"]) <= t_issue]
        last_t = float(before[-1]["t"]) if before else 0.0
        failed = bool(before) and float(before[-1][damage_state]) >= unit.truth.true_params[limit_param]
        if failed or any(last_t < t <= t_issue for t in limit_times):
            return t_issue
    later = [t for t in limit_times if t > t_issue]
    return later[0] if later else None


def toe_samples(rows: Sequence[Mapping[str, str]], event: str) -> np.ndarray:
    """Each sample's time of event (+inf: censored, no crossing before the horizon)."""
    return np.array([float(r[f"{event}.toe"]) if r[f"{event}.toe"] else np.inf for r in rows])


def interval(samples: np.ndarray, qs: tuple[float, float] = INTERVAL) -> tuple[float, float]:
    lo, hi = (float(np.quantile(samples, q, method="inverted_cdf")) for q in qs)
    return lo, hi


def covers(samples: np.ndarray, truth: float) -> bool:
    """Whether the 5-95% interval holds the truth; samples may be +inf (censored)."""
    lo, hi = interval(samples)
    return bool(lo <= truth <= hi)


def shared_groups(twin: TwinDefinition) -> list[tuple[str, tuple[str, ...]]]:
    """(channel, bindings) for each channel several parts are observed on linearly: not identifiable per part."""
    out = []
    for g in groups(twin):
        for ch in g.channels:
            on = tuple(sorted({o.binding for o in g.observations if o.channel == ch and o.form == "linear"}))
            if len(on) > 1:
                out.append((ch, on))
    return out


def joined_bindings(twin: TwinDefinition) -> tuple[str, ...]:
    """Bindings in an observed group with no observation of their own (joined through a connection), sorted."""
    own = {o.binding for o in twin.observations}
    return tuple(sorted(b for g in groups(twin) if g.observed for b in g.bindings if b not in own))
