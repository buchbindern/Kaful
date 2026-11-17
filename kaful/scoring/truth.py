"""Synthetic truth, read for scoring (M13's scorer, M14's calibration): the only shared code that opens truth files.

- `load_units`: each unit's truth artifact and true damage states;
- `true_next_event`: the true next time a part reaches its limit after t_issue (the forecast's event definition);
"""

from dataclasses import dataclass
from pathlib import Path

from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth
from kaful.paths import resolve_in
from kaful.scoring.samples import Table, read_table



@dataclass(frozen=True)
class Unit:
    truth: SyntheticTruth
    states: Table
    t_end: float


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
