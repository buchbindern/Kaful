"""The only figures module that reads synthetic truth (allow-listed in the truth guard, M16).

Truth is read through `kaful.scoring.truth`, and the forecast/truth pairing is evaluate's own (`collect_forecast`),
so a figure's true RUL is exactly the one calibration scored. Used by figures f2_tracking and f3_failure_time.
- `load_truth`: every unit's truth, recording the truth files as figure inputs;
- `rul_records`: per (estimator, quantity), one record per unit and issue, as evaluate's calibration builds them;
- `true_health`: a unit's true health 1 - D / D_limit (clipped to [0, 1]) per binding, on the truth grid.
"""

from kaful.config import EvaluateSettings
from kaful.figures.sources import LoadedLineage
from kaful.phases.p9_evaluate.f_calibration import Collected, Record, collect_forecast
from kaful.phases.p9_evaluate.helpers.parts import part_classes
from kaful.scoring.truth import Unit, load_units

Series = tuple[tuple[float, ...], tuple[float, ...]]


def load_truth(ll: LoadedLineage) -> dict[str, Unit]:
    dataset = ll.dataset()
    for u in dataset.units:
        ll.file(u.truth.path)
        ll.file(u.truth_states.path)
    return load_units(ll.machine_dir, dataset)


def rul_records(ll: LoadedLineage, units: dict[str, Unit], estimators: tuple[str, ...],
                settings: EvaluateSettings) -> dict[tuple[str, str], list[Record]]:
    twin = ll.twin()
    c = Collected(part=part_classes(twin))
    for f in ll.forecasts(estimators):
        for issue in f.issues:
            ll.file(issue.samples)
        collect_forecast(f, units[f.source], twin, ll.machine_dir, settings, c)
    return {k: c.records[k] for k in sorted(c.records)}


def true_health(ll: LoadedLineage, unit: Unit, bindings: tuple[str, ...]) -> dict[str, Series]:
    by_id = {b.id: b for b in ll.twin().bindings}
    t = tuple(float(r["t"]) for r in unit.states)
    out = {}
    for b in bindings:
        limit = unit.truth.true_params[by_id[b].threshold_param]
        h = tuple(min(1.0, max(0.0, 1.0 - float(r[by_id[b].damage_state]) / limit)) for r in unit.states)
        out[b] = (t, h)
    return out
