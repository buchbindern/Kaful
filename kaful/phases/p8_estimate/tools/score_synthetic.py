"""Score an estimate run on synthetic data against the hidden truth (M13). Not used during a run (a tool); the only
code in estimate that reads truth. A pipeline sanity check, not validation: the data came from the twin's own priors.

Per source and estimator:
- state error: RMSE of the trace's mean health against true health (1 - D / D_limit, clipped) at the truth grid
  rows, per binding; and, per linear channel, of the expected reading against the true noise-free reading;
- life: at the last issue time, |log(posterior median life / true life)| against the prior median's, and whether the
  90% interval covers the true life (life = D_limit / k: for water_filter the identifiable quantity, otherwise the
  same as judging k);
- a forecast preview (M14 owns calibration): whether each event's 5-95% covers the true next limit time.
Shared-channel groups (CNC's bearings) are judged label-invariantly (approved 2026-11-16): the channel's reading,
the sorted pair of rates, and the time to the group's first failure.

A binding seen only through alarms is scored on life only in units where it reached its limit inside the window
("informative"); its state error counts every unit.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field

from kaful.artifacts.base import StrictModel
from kaful.artifacts.forecast import Forecast, ForecastSet
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.estimation.groups import groups
from kaful.observation.forms import linear_channel
from kaful.observation.protocol import DamageView
from kaful.paths import resolve_in

Table = list[dict[str, str]]


class Comparison(StrictModel):
    """One quantity over the units it applies to: estimator vs open-loop, and vs the prior."""

    target: str = Field(description="A binding, a channel, or a group's label-invariant quantity.")
    kind: str = Field(description="state_rmse | channel_rmse | life | pair_min_k | pair_max_k | first_failure")
    units: int
    estimator_mean: float | None = None
    open_loop_mean: float | None = None
    wins: int | None = Field(default=None, description="Units where the estimator beat open-loop (state error).")
    prior_mean: float | None = Field(default=None, description="Mean |log error| of the prior median.")
    closer: int | None = Field(default=None, description="Units where the posterior median beat the prior median.")
    covers: int | None = Field(default=None, description="Units whose 90% interval covers the truth.")


class EstimatorScore(StrictModel):
    estimator: str
    comparisons: tuple[Comparison, ...]
    forecast_cover: dict[str, list[int]] = Field(default_factory=dict, description=(
        "Preview (M14 owns calibration): event -> [units whose 5-95% covers the true next limit time, units]."))


class EstimateScore(StrictModel):
    machine_id: str
    run_id: str
    disclaimer: str
    scores: tuple[EstimatorScore, ...]
    runtime_note: str = ""


@dataclass(frozen=True)
class Unit:
    truth: SyntheticTruth
    states: Table
    t_end: float


def _read(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _true_health(twin: TwinDefinition, unit: Unit, binding: str, row: Mapping[str, str]) -> float:
    b = next(x for x in twin.bindings if x.id == binding)
    return float(np.clip(1.0 - float(row[b.damage_state]) / unit.truth.true_params[b.threshold_param], 0.0, 1.0))


def _true_reading(twin: TwinDefinition, unit: Unit, channel: str, row: Mapping[str, str]) -> float:
    on = sorted((o for o in twin.observations if o.channel == channel and o.form == "linear"), key=lambda o: o.id)
    by_id, p = {b.id: b for b in twin.bindings}, unit.truth.true_params
    terms = []
    for o in on:
        b = by_id[o.binding]
        frac = np.array([float(row[b.damage_state]) / p[b.threshold_param]])
        terms.append((DamageView(frac, np.zeros(1, dtype=bool)), p[o.params["y_limit"]]))
    return float(linear_channel(p[on[0].params["y_healthy"]], terms)[0])


def _rmse(trace: Table, unit: Unit, column: str, truth_of: object, interval: float) -> float | None:
    grid = np.array([float(r["t"]) for r in unit.states])
    errors = []
    for r in trace:
        if not r.get(column):
            continue
        k = int(np.argmin(np.abs(grid - float(r["t"]))))
        if abs(grid[k] - float(r["t"])) <= 0.5 * interval:
            errors.append(float(r[column]) - truth_of(unit.states[k]))  # type: ignore[operator]
    return float(np.sqrt(np.mean(np.square(errors)))) if errors else None


def _life_samples(samples: Table, twin: TwinDefinition, ps: ParamSet, binding: str) -> np.ndarray | None:
    b = next(x for x in twin.bindings if x.id == binding)
    k, lim = b.params.get("k"), b.threshold_param
    if k is None or not samples or k not in samples[0]:
        return None
    ks = np.array([float(r[k]) for r in samples])
    limits = np.array([float(r[lim]) for r in samples]) if lim in samples[0] else np.full(ks.size, _median(ps, lim))
    return np.asarray(np.log(limits) - np.log(ks))


def _median(ps: ParamSet, pid: str) -> float:
    return float(ps.get(pid).prior.quantile(0.5))


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


def _covers(samples: np.ndarray, truth: float) -> bool:
    """Whether the 5-95% interval holds the truth; samples may be +inf (censored), so the quantiles are sample values
    (inverted CDF, as the forecast's percentiles), never an interpolation through infinity."""
    lo, hi = (float(np.quantile(samples, q, method="inverted_cdf")) for q in (0.05, 0.95))
    return bool(lo <= truth <= hi)


def _informative(twin: TwinDefinition, unit: Unit, binding: str) -> bool:
    linear = {o.binding for o in twin.observations if o.form == "linear"}
    if binding in linear:
        return True
    event = next(e.event for e in unit.truth.events if e.event == f"{_component(twin, binding)}_failure")
    return any(t <= unit.t_end for t in next(e for e in unit.truth.events if e.event == event).limit_times)


def _component(twin: TwinDefinition, binding: str) -> str:
    return next(b.component for b in twin.bindings if b.id == binding)


@dataclass
class Collected:
    """Per (target, kind), per unit: (estimator value, open-loop value, prior value, covered)."""

    rows: dict[tuple[str, str], list[tuple[float | None, float | None, float | None, bool | None]]]
    cover: dict[str, list[int]]


def _shared_groups(twin: TwinDefinition) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for g in groups(twin):
        for ch in g.channels:
            on = tuple(sorted({o.binding for o in g.observations if o.channel == ch and o.form == "linear"}))
            if len(on) > 1:
                out.append((ch, on))
    return out


def _unit_rows(twin: TwinDefinition, ps: ParamSet, unit: Unit, own: Forecast, ol: Forecast, machine_dir: Path,
               interval: float, c: Collected) -> None:
    trace, ol_trace = _read(resolve_in(machine_dir, own.trace)), _read(resolve_in(machine_dir, ol.trace))
    shared = _shared_groups(twin)
    in_shared = {b for _, bs in shared for b in bs}
    observed = sorted({b for g in groups(twin) if g.observed for b in g.bindings})
    for b in observed:
        col = f"{b}.health"
        fn = (lambda row, b=b: _true_health(twin, unit, b, row))
        c.rows.setdefault((b, "state_rmse"), []).append(
            (_rmse(trace, unit, col, fn, interval), _rmse(ol_trace, unit, col, fn, interval), None, None))
    for ch in sorted({o.channel for o in twin.observations if o.form == "linear"}):
        fn = (lambda row, ch=ch: _true_reading(twin, unit, ch, row))
        col = f"{ch}.expected"
        c.rows.setdefault((ch, "channel_rmse"), []).append(
            (_rmse(trace, unit, col, fn, interval), _rmse(ol_trace, unit, col, fn, interval), None, None))
    last = own.issues[-1]
    samples = _read(resolve_in(machine_dir, last.samples))
    for b in observed:
        if b in in_shared or not _informative(twin, unit, b):
            continue
        life = _life_samples(samples, twin, ps, b)
        if life is None:
            continue
        bd = next(x for x in twin.bindings if x.id == b)
        true = np.log(unit.truth.true_params[bd.threshold_param]) - np.log(unit.truth.true_params[bd.params["k"]])
        prior = np.log(_median(ps, bd.threshold_param)) - np.log(_median(ps, bd.params["k"]))
        c.rows.setdefault((b, "life"), []).append(
            (abs(float(np.median(life)) - true), None, abs(prior - true), _covers(life, true)))
    for ch, bs in shared:
        _pair_rows(twin, ps, unit, samples, ch, bs, last, c)
    by_id = {b.id: b for b in twin.bindings}
    renewing = {m.binding for m in twin.maintenance if m.trigger == "threshold"}
    for e in last.events:
        bd = by_id[e.binding]
        truth = true_next_event(unit, e.event, bd.damage_state, bd.threshold_param, last.t_issue, bd.id in renewing)
        lo, hi = e.percentiles["p05"], e.percentiles["p95"]
        ok = (hi is None if truth is None else (lo is not None and lo <= truth and (hi is None or truth <= hi)))
        hit = c.cover.setdefault(e.event, [0, 0])
        hit[0] += int(ok)
        hit[1] += 1


def _pair_rows(twin: TwinDefinition, ps: ParamSet, unit: Unit, samples: Table, ch: str, bs: Sequence[str],
               last: object, c: Collected) -> None:
    by_id = {b.id: b for b in twin.bindings}
    ks = np.stack([np.log([float(r[by_id[b].params["k"]]) for r in samples]) for b in bs])
    true = np.sort([np.log(unit.truth.true_params[by_id[b].params["k"]]) for b in bs])
    prior = np.sort([np.log(_median(ps, by_id[b].params["k"])) for b in bs])
    ordered = np.sort(ks, axis=0)
    for i, kind in ((0, "pair_min_k"), (len(bs) - 1, "pair_max_k")):
        c.rows.setdefault((ch, kind), []).append(
            (abs(float(np.median(ordered[i])) - true[i]), None, abs(prior[i] - true[i]), _covers(ordered[i], true[i])))
    t_issue = getattr(last, "t_issue")
    events = [f"{by_id[b].component}_failure" for b in bs]
    toe = np.stack([np.array([float(r[f"{e}.toe"]) if r[f"{e}.toe"] else np.inf for r in samples]) for e in events])
    first = np.min(toe, axis=0)
    renewing = {m.binding for m in twin.maintenance if m.trigger == "threshold"}
    truths = [t for t in (true_next_event(unit, e, by_id[b].damage_state, by_id[b].threshold_param, t_issue,
                                          b in renewing) for e, b in zip(events, bs)) if t is not None]
    true_first = min(truths) if truths else np.inf
    covered = _covers(first, true_first)
    c.rows.setdefault((ch, "first_failure"), []).append((None, None, None, covered))


def _summaries(c: Collected) -> tuple[Comparison, ...]:
    out = []
    for (target, kind), rows in sorted(c.rows.items()):
        est = [r[0] for r in rows if r[0] is not None]
        ol = [r[1] for r in rows if r[1] is not None]
        pri = [r[2] for r in rows if r[2] is not None]
        cov = [r[3] for r in rows if r[3] is not None]
        pairs = [(r[0], r[1]) for r in rows if r[0] is not None and r[1] is not None]
        closer = [(r[0], r[2]) for r in rows if r[0] is not None and r[2] is not None]
        out.append(Comparison(
            target=target, kind=kind, units=len(rows), estimator_mean=float(np.mean(est)) if est else None,
            open_loop_mean=float(np.mean(ol)) if ol else None,
            wins=sum(a < b for a, b in pairs) if pairs else None, prior_mean=float(np.mean(pri)) if pri else None,
            closer=sum(a < b for a, b in closer) if closer else None, covers=sum(cov) if cov else None))
    return tuple(out)


def score(machine_dir: Path, run_id: str, fs: ForecastSet, dataset: SyntheticDataset, twin: TwinDefinition,
          ps: ParamSet) -> EstimateScore:
    units = {}
    for u in dataset.units:
        truth = SyntheticTruth.model_validate_json(resolve_in(machine_dir, u.truth.path).read_text())
        units[u.unit_id] = Unit(truth=truth, states=_read(resolve_in(machine_dir, u.truth_states.path)), t_end=u.t_end)
    forecasts = {(e.source, e.estimator): Forecast.model_validate_json(resolve_in(machine_dir, e.path).read_text())
                 for e in fs.forecasts}
    scores = []
    for method in sorted({m for _, m in forecasts}):
        c = Collected(rows={}, cover={})
        for unit_id, unit in sorted(units.items()):
            if (unit_id, method) in forecasts and (unit_id, "open_loop") in forecasts:
                _unit_rows(twin, ps, unit, forecasts[(unit_id, method)], forecasts[(unit_id, "open_loop")],
                           machine_dir, dataset.sample_interval, c)
        scores.append(EstimatorScore(estimator=method, comparisons=_summaries(c),
                                     forecast_cover=dict(sorted(c.cover.items()))))
    return EstimateScore(machine_id=dataset.machine_id, run_id=run_id, disclaimer=dataset.disclaimer,
                         scores=tuple(scores))
