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

Joined parts (M13b): a binding in an observed group with no observation of its own, joined only through a connection
(the eversys boilers, through the water filter's W-014). No row's likelihood reads its own parameters, so the
particle filter's posterior on them should equal their prior: a two-sample KS test per unit against prior draws
(settings.scoring). What the data does give it is the source's state: the report compares its 25-75% forecast
width with open-loop's (the 95% points are mostly censored), and how much of its time-to-limit spread each carried parameter explains (squared
Spearman correlation, samples that cross after the issue time).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from pydantic import Field

from kaful.artifacts.base import StrictModel
from kaful.artifacts.forecast import Forecast, ForecastSet
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import ScoringSettings
from kaful.estimation.groups import groups
from kaful.observation.forms import linear_channel
from kaful.observation.protocol import DamageView
from kaful.paths import resolve_in
from kaful.scoring.samples import covers
from kaful.scoring.groups import joined_bindings, shared_groups
from kaful.scoring.samples import Table, read_table
from kaful.scoring.truth import Unit, load_units, true_next_event

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


class PriorCheck(StrictModel):
    """Posterior vs prior for one parameter of a joined part, over units (particle filter, last issue)."""

    param: str
    units: int
    ks_mean: float
    ks_max: float
    ks_critical: float
    rejected: int = Field(description="Units where the KS test rejects 'posterior = prior' at settings.scoring.ks_alpha.")
    alpha: float


class JoinedPart(StrictModel):
    binding: str
    priors: tuple[PriorCheck, ...]
    width_ratio: float | None = Field(description=(
        "Median over (unit, issue) of the particle filter's 25-75% time-to-limit width / open-loop's, where both are "
        "finite and the part has not already reached its limit at the 25% point."))
    width_pairs: int
    spread_share: dict[str, float | None] = Field(description=(
        "Mean over (unit, issue) of the squared Spearman correlation of the time to limit with each carried "
        "parameter (particle filter)."))


class EstimateScore(StrictModel):
    machine_id: str
    run_id: str
    disclaimer: str
    scores: tuple[EstimatorScore, ...]
    joined: tuple[JoinedPart, ...] = ()
    runtime_note: str = ""


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


def _unit_rows(twin: TwinDefinition, ps: ParamSet, unit: Unit, own: Forecast, ol: Forecast, machine_dir: Path,
               interval: float, c: Collected) -> None:
    trace, ol_trace = read_table(resolve_in(machine_dir, own.trace)), read_table(resolve_in(machine_dir, ol.trace))
    shared = shared_groups(twin)
    in_shared = {b for _, bs in shared for b in bs}
    observed = sorted({b for g in groups(twin) if g.observed for b in g.bindings})
    for b in observed:
        col = f"{b}.health"
        fn = partial(_true_health, twin, unit, b)
        c.rows.setdefault((b, "state_rmse"), []).append(
            (_rmse(trace, unit, col, fn, interval), _rmse(ol_trace, unit, col, fn, interval), None, None))
    for ch in sorted({o.channel for o in twin.observations if o.form == "linear"}):
        fn = partial(_true_reading, twin, unit, ch)
        col = f"{ch}.expected"
        c.rows.setdefault((ch, "channel_rmse"), []).append(
            (_rmse(trace, unit, col, fn, interval), _rmse(ol_trace, unit, col, fn, interval), None, None))
    last = own.issues[-1]
    samples = read_table(resolve_in(machine_dir, last.samples))
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
            (abs(float(np.median(life)) - true), None, abs(prior - true), covers(life, true)))
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
            (abs(float(np.median(ordered[i])) - true[i]), None, abs(prior[i] - true[i]), covers(ordered[i], true[i])))
    t_issue = getattr(last, "t_issue")
    events = [f"{by_id[b].component}_failure" for b in bs]
    toe = np.stack([np.array([float(r[f"{e}.toe"]) if r[f"{e}.toe"] else np.inf for r in samples]) for e in events])
    first = np.min(toe, axis=0)
    renewing = {m.binding for m in twin.maintenance if m.trigger == "threshold"}
    truths = [t for t in (true_next_event(unit, e, by_id[b].damage_state, by_id[b].threshold_param, t_issue,
                                          b in renewing) for e, b in zip(events, bs)) if t is not None]
    true_first = min(truths) if truths else np.inf
    covered = covers(first, true_first)
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


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                               - np.searchsorted(b, grid, side="right") / b.size)))


def ks_critical(n: int, m: int, alpha: float) -> float:
    return float(np.sqrt(-np.log(alpha / 2.0) / 2.0) * np.sqrt((n + m) / (n * m)))


def _spearman2(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1] ** 2)


WIDTH_QUANTILES = (0.25, 0.75)  # the boilers' 95% points are mostly past the horizon (censored): use the quartiles


def _width(f: Forecast, event: str, t_issue: float, machine_dir: Path) -> float | None:
    issue = next(i for i in f.issues if i.t_issue == t_issue)
    rows = read_table(resolve_in(machine_dir, issue.samples))
    toe = np.array([float(r[f"{event}.toe"]) if r[f"{event}.toe"] else np.inf for r in rows])
    lo, hi = (float(np.quantile(toe, q, method="inverted_cdf")) for q in WIDTH_QUANTILES)
    return None if lo <= t_issue or not np.isfinite(hi) else hi - lo


def _joined_part(b: str, twin: TwinDefinition, ps: ParamSet, pf: Sequence[Forecast], ol: Mapping[str, Forecast],
                 machine_dir: Path, s: ScoringSettings) -> JoinedPart:
    bd = next(x for x in twin.bindings if x.id == b)
    event = f"{bd.component}_failure"
    pids = sorted({bd.params["k"], *(p for c in twin.connections if c.to_binding == b for p in c.params.values())})
    rng = np.random.default_rng(s.prior_reference_seed)
    reference = {pid: np.asarray(ps.get(pid).prior.sample(rng, s.prior_reference_samples)) for pid in pids}
    ks: dict[str, list[float]] = {pid: [] for pid in pids}
    shares: dict[str, list[float]] = {pid: [] for pid in pids}
    ratios: list[float] = []
    n = 0
    for f in pf:
        last = read_table(resolve_in(machine_dir, f.issues[-1].samples))
        n = len(last)
        for pid in pids:
            ks[pid].append(ks_distance(np.array([float(r[pid]) for r in last]), reference[pid]))
        for issue in f.issues:
            rows = read_table(resolve_in(machine_dir, issue.samples))
            toe = np.array([float(r[f"{event}.toe"]) if r[f"{event}.toe"] else np.inf for r in rows])
            live = np.isfinite(toe) & (toe > issue.t_issue)
            for pid in pids:
                v = _spearman2(toe[live], np.array([float(r[pid]) for r in rows])[live])
                if v is not None:
                    shares[pid].append(v)
            mine = _width(f, event, issue.t_issue, machine_dir)
            theirs = _width(ol[f.source], event, issue.t_issue, machine_dir)
            if mine is not None and theirs:
                ratios.append(mine / theirs)
    crit = ks_critical(n, s.prior_reference_samples, s.ks_alpha) if n else 0.0
    checks = tuple(PriorCheck(param=pid, units=len(v), ks_mean=float(np.mean(v)), ks_max=float(np.max(v)),
                              ks_critical=crit, rejected=sum(x > crit for x in v), alpha=s.ks_alpha)
                   for pid, v in sorted(ks.items()) if v)
    return JoinedPart(binding=b, priors=checks, width_ratio=float(np.median(ratios)) if ratios else None,
                      width_pairs=len(ratios),
                      spread_share={pid: (float(np.mean(v)) if v else None) for pid, v in sorted(shares.items())})


def score(machine_dir: Path, run_id: str, fs: ForecastSet, dataset: SyntheticDataset, twin: TwinDefinition,
          ps: ParamSet, scoring: ScoringSettings | None = None) -> EstimateScore:
    units = load_units(machine_dir, dataset)
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
    pf = [f for (src, m), f in sorted(forecasts.items()) if m == "particle_filter"]
    ol = {src: f for (src, m), f in forecasts.items() if m == "open_loop"}
    joined = tuple(_joined_part(b, twin, ps, pf, ol, machine_dir, scoring or ScoringSettings())
                   for b in joined_bindings(twin)) if pf else ()
    return EstimateScore(machine_id=dataset.machine_id, run_id=run_id, disclaimer=dataset.disclaimer,
                         scores=tuple(scores), joined=joined)
