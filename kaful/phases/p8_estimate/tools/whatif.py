"""What-if forecasts (M13b): how a maintenance change moves the parts a connection couples, against the documented
schedule. Offline; no API calls; reads no truth.

`kaful estimate whatif <machine> --renew-at water_filter=1.5 [--run RUN]` replaces each named binding (a part
replaced on threshold) at that multiple of its D_limit instead of at it, and forecasts the bindings it feeds (and
itself) twice from the same belief with the same seed: under the dataset's documented profile and under the what-if.
Common random numbers make the per-sample difference the effect of the change alone. Three views:

- from new: a new machine (the prior, t = 0);
- per unit: each synthetic unit's belief at each issue time: the particle filter's when every reported binding is in
  an observed group (eversys: the boilers join W-014's group), else open-loop; the same seeds as the estimate phase,
  so the documented forecast is the estimate run's;
- long run: the coupling form's closed-form mean multiplier over many source cycles (`delayed_mean`), over r's
  distribution (W-014 does not inform r, so its posterior is its prior).

Each view has a one-line conclusion computed from its numbers. How much of a change's spread each uncertain input
explains is the squared Spearman correlation between the per-sample change and that input (r, the target's k).
The report says what information would narrow it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import Field

from kaful.arrays import FloatArray
from kaful.artifacts.base import Identifier, NonEmptyStr, StrictModel
from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import Connection, TwinDefinition
from kaful.backends import load_backend
from kaful.config import EstimatorName, KafulSettings
from kaful.dynamics.step import simulation_step
from kaful.estimation.predict import EventSamples, Prediction
from kaful.estimation.summaries import medians
from kaful.phases.p8_estimate.a_open_sources import Source, read_rows, sources
from kaful.phases.p8_estimate.b_start_estimators import observed_bindings, predict_seed, start
from kaful.phases.p8_estimate.c_filter import issue_rows, run_filter
from kaful.phases.p8_estimate.d_forecast import forecast_issue
from kaful.phases.p8_estimate.helpers.workers import run_in_order
from kaful.templates.couplings import COUPLINGS

NEW_MACHINE_KEY = 1_000_000  # spawn key of the from-new view (no synthetic unit uses it)
R_GRID = 20_001  # quantiles of r's prior evaluated for the long-run view
DISCLAIMER = ("What-if on the twin's own assumptions: W-014 does not identify a boiler's k or the coupling strength r "
              "(their posteriors are their priors), so the change shown comes from the coupling form, r's default "
              "prior and the boilers' judged lives; it is not validated against failure data.")


class TimeSummary(StrictModel):
    p05: float | None
    p50: float | None
    p95: float | None
    p_before_horizon: float
    n_already_triggered: int


class Change(StrictModel):
    n_samples: int
    n_changed: int = Field(description="Samples whose time to limit moves (both reach it before the horizon).")
    n_both_hit: int
    relative: dict[str, float | None] = Field(description="p05/p50/p95 of delayed / documented - 1, moved samples.")
    spread_share: dict[str, float | None] = Field(description=(
        "Squared Spearman correlation of the relative change with each uncertain input (param_id), changed samples."))


class EventWhatIf(StrictModel):
    binding: Identifier
    documented: TimeSummary
    delayed: TimeSummary
    change: Change


class FromNew(StrictModel):
    events: tuple[EventWhatIf, ...]
    conclusion: NonEmptyStr


class UnitIssue(StrictModel):
    source: NonEmptyStr
    t_issue: float
    events: tuple[EventWhatIf, ...]


class PerUnit(StrictModel):
    issues: tuple[UnitIssue, ...]
    moved_share: dict[Identifier, float] = Field(description=(
        "Per binding: the median over (unit, issue) forecasts of the share of samples whose time to limit moves."))
    failed_share: dict[Identifier, float] = Field(description=(
        "Per binding: the median over (unit, issue) forecasts of the share of samples already at the limit at issue."))
    conclusion: NonEmptyStr


class LongRun(StrictModel):
    factor: float
    rate_change: dict[str, float] = Field(description="p05/p50/p95 over r of the long-run mean multiplier - 1.")
    rate_change_r_known: dict[str, float] = Field(description=(
        "The same if r were known to +-narrow_r_halfwidth around its median: low, high."))
    conclusion: NonEmptyStr


class WhatIfReport(StrictModel):
    machine_id: Identifier
    run_id: NonEmptyStr
    disclaimer: NonEmptyStr
    renew_at: dict[Identifier, float]
    horizon: float
    time_unit: NonEmptyStr
    from_new: FromNew
    per_unit: PerUnit
    long_run: dict[Identifier, LongRun] = Field(description="Per connection id.")
    narrowing: NonEmptyStr


def _summary(e: EventSamples, horizon: float) -> TimeSummary:
    hits = ~np.isnan(e.toe)
    ordered = np.where(hits, e.toe, np.inf)
    q = {k: float(np.quantile(ordered, p, method="inverted_cdf")) for k, p in (("p05", .05), ("p50", .5), ("p95", .95))}
    return TimeSummary(**{k: (v if np.isfinite(v) else None) for k, v in q.items()},
                       p_before_horizon=float(hits.mean()), n_already_triggered=int(e.triggered.sum()))


def _spearman2(a: FloatArray, b: FloatArray) -> float | None:
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1] ** 2)


def _change(doc: EventSamples, late: EventSamples, inputs: Mapping[str, FloatArray]) -> Change:
    both = ~np.isnan(doc.toe) & ~np.isnan(late.toe) & ~doc.triggered
    rel = np.full(doc.toe.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel[both] = late.toe[both] / np.where(doc.toe[both] > 0, doc.toe[both], np.nan) - 1.0
    moved = both & (np.abs(late.toe - doc.toe) > 0)
    finite = moved & np.isfinite(rel)
    pct = {k: (float(np.quantile(rel[finite], p)) if finite.any() else None)
           for k, p in (("p05", .05), ("p50", .5), ("p95", .95))}
    share = {pid: _spearman2(rel[moved & finite], v[moved & finite]) for pid, v in sorted(inputs.items())}
    return Change(n_samples=int(doc.toe.size), n_changed=int(moved.sum()), n_both_hit=int(both.sum()),
                  relative=pct, spread_share=share)


def _events(doc: Prediction, late: Prediction, wanted: Sequence[str], inputs: Mapping[str, Sequence[str]],
            horizon: float) -> tuple[EventWhatIf, ...]:
    a, b = {e.binding: e for e in doc.events}, {e.binding: e for e in late.events}
    out = []
    for bid in wanted:
        params = {pid: doc.params[pid] for pid in inputs.get(bid, ()) if pid in doc.params}
        out.append(EventWhatIf(binding=bid, documented=_summary(a[bid], horizon), delayed=_summary(b[bid], horizon),
                               change=_change(a[bid], b[bid], params)))
    return tuple(out)


@dataclass(frozen=True)
class Job:
    twin: TwinDefinition
    ps: ParamSet
    contract: SensorContract
    dataset: SyntheticDataset
    settings: KafulSettings
    renew_at: dict[str, float]
    wanted: tuple[str, ...]
    inputs: dict[str, tuple[str, ...]]
    source: Source | None  # None: the from-new view


def _pair(job: Job, backend: object, built: object, belief: object, dt: float, key: int, i: int) -> tuple[EventWhatIf, ...]:
    s, d = job.settings.estimate, job.dataset
    late = d.profile.model_copy(update={"renew_at": dict(job.renew_at)})
    preds = [forecast_issue(backend, built, belief, p, dt, d.horizon, s, predict_seed(s, key, i), job.wanted)  # type: ignore[arg-type]
             for p in (d.profile, late)]
    assert preds[0] is not None and preds[1] is not None
    return _events(preds[0], preds[1], job.wanted, job.inputs, d.horizon)


def run_job(job: Job) -> list[UnitIssue]:
    s, d = job.settings, job.dataset
    backend = load_backend(s.build.backend)
    built = backend.build(job.twin, job.ps, job.contract)
    dt = simulation_step(job.twin, job.ps, s.verify.steps_per_shortest_life)
    if job.source is None:
        est = start(backend, built, job.ps, d.profile, "open_loop", dt, s.estimate, NEW_MACHINE_KEY, ())
        return [UnitIssue(source="new", t_issue=0.0,
                          events=_pair(job, backend, built, getattr(est, "belief"), dt, NEW_MACHINE_KEY, 0))]
    src = job.source
    rows = read_rows(src, job.contract, d)
    method: EstimatorName = "particle_filter" if set(job.wanted) <= set(observed_bindings(job.twin)) else "open_loop"
    est = start(backend, built, job.ps, d.profile, method, dt, s.estimate, src.spawn_key, src.monitored)
    issues = issue_rows([r.t for r in rows], s.estimate.issue_fractions)
    filtered = run_filter(est, rows, issues, job.twin, medians(job.ps))
    return [UnitIssue(source=src.source_id, t_issue=b.t, events=_pair(job, backend, built, b, dt, src.spawn_key, i))
            for i, b in enumerate(filtered.beliefs)]


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.0f}%"


def _new_conclusion(events: Sequence[EventWhatIf], targets: Sequence[str], unit: str) -> str:
    parts = []
    for e in (e for e in events if e.binding in targets):
        c = e.change
        share = c.n_changed / c.n_samples
        drivers = ", ".join(f"{pid} {v:.0%}" for pid, v in c.spread_share.items() if v is not None)
        parts.append(f"{e.binding}: median time to limit {e.documented.p50 or float('nan'):,.0f} -> "
                     f"{e.delayed.p50 or float('nan'):,.0f} {unit}; {share:.0%} of samples move (the others reach the "
                     f"limit before the filter is first left past W-014), those by {_pct(c.relative['p50'])} at the "
                     f"median ({_pct(c.relative['p05'])} to {_pct(c.relative['p95'])})"
                     + (f"; the change's spread explained by {drivers}" if drivers else ""))
    return "New machine: " + "; ".join(parts) + "."


def _unit_conclusion(moved: Mapping[str, float], failed: Mapping[str, float], targets: Sequence[str]) -> str:
    parts = [f"{b}: {moved.get(b, 0.0):.0%} of samples move, {failed.get(b, 0.0):.0%} have already reached the limit"
             for b in targets]
    return ("Synthetic units, in the median (unit, issue) forecast: " + "; ".join(parts)
            + " (a part already at its limit, with no maintenance rule, cannot be moved by the delay).")


def long_run(c: Connection, ps: ParamSet, factor: float, halfwidth: float) -> LongRun:
    form = COUPLINGS[c.form]()
    [(slot, pid)] = sorted(c.params.items())
    prior = ps.get(pid).prior
    qs = np.array([prior.quantile(p) for p in (np.arange(R_GRID) + 0.5) / R_GRID])
    change = np.array([form.delayed_mean({slot: float(r)}, factor) - 1.0 for r in qs])
    lo, hi = form.bounds[slot]
    med = prior.quantile(0.5)
    known = sorted(form.delayed_mean({slot: min(max(med + d, lo), hi)}, factor) - 1.0 for d in (-halfwidth, halfwidth))
    pct = {k: float(np.quantile(change, p)) for k, p in (("p05", .05), ("p50", .5), ("p95", .95))}
    return LongRun(factor=factor, rate_change=pct, rate_change_r_known={"low": known[0], "high": known[1]},
                   conclusion=(f"Over many filter cycles, replacing {c.from_binding} at {factor:g}x its limit raises "
                               f"{c.to_binding}'s scaling rate by {_pct(pct['p50'])} (90%: {_pct(pct['p05'])} to "
                               f"{_pct(pct['p95'])}); the range comes from {pid} alone, which W-014 does not inform."))


def narrowing(conns: Sequence[Connection], runs: Mapping[str, LongRun], halfwidth: float) -> str:
    c = conns[0]
    lr = runs[c.id]
    pid = next(iter(c.params.values()))
    return (f"{pid} (the residual hardness fraction: the share of inlet hardness a fresh filter passes) drives most "
            f"of the range. A filter datasheet (its bypass setting) or a water-hardness test downstream of a fresh "
            f"filter would narrow it: knowing it to +-{halfwidth:g} around its median would narrow the long-run change "
            f"from {_pct(lr.rate_change['p05'])}..{_pct(lr.rate_change['p95'])} to "
            f"{_pct(lr.rate_change_r_known['low'])}..{_pct(lr.rate_change_r_known['high'])}. The boilers' own k "
            f"(judged lives, wide) sets when they reach their limit, and only boiler data (a descaling record, scale "
            f"inspection) would narrow it.")


def whatif(run_id: str, twin: TwinDefinition, ps: ParamSet, contract: SensorContract, dataset: SyntheticDataset,
           settings: KafulSettings, machine_dir: Path, renew_at: dict[str, float], halfwidth: float) -> WhatIfReport:
    renewing = {m.binding for m in twin.maintenance if m.trigger == "threshold"}
    unknown = sorted(set(renew_at) - renewing)
    if unknown:
        raise ValueError(f"--renew-at names bindings without a threshold rule: {unknown}")
    conns = sorted((c for c in twin.connections if c.from_binding in renew_at), key=lambda c: c.id)
    if not conns:
        raise ValueError(f"no connection leaves {sorted(renew_at)}: the what-if would move nothing it feeds")
    by_id = {b.id: b for b in twin.bindings}
    targets = tuple(sorted({c.to_binding for c in conns}))
    wanted = tuple(sorted({*targets, *renew_at}))
    inputs = {t: tuple(sorted({by_id[t].params["k"], *(p for c in conns if c.to_binding == t
                                                        for p in c.params.values())})) for t in targets}
    base = Job(twin, ps, contract, dataset, settings, dict(sorted(renew_at.items())), wanted, inputs, None)
    jobs = [base] + [Job(**{**base.__dict__, "source": s}) for s in sources(dataset, contract, machine_dir)]
    results = run_in_order(run_job, jobs, settings.estimate.workers)
    new, units = results[0][0], [u for r in results[1:] for u in r]
    def per_issue(b: str, of: str) -> float:
        vals = [(e.change.n_changed if of == "moved" else e.documented.n_already_triggered) / e.change.n_samples
                for u in units for e in u.events if e.binding == b]
        return float(np.median(vals)) if vals else 0.0

    moved = {b: per_issue(b, "moved") for b in wanted}
    failed = {b: per_issue(b, "failed") for b in wanted}
    runs = {c.id: long_run(c, ps, renew_at[c.from_binding], halfwidth) for c in conns}
    return WhatIfReport(
        machine_id=twin.machine_id, run_id=run_id, disclaimer=DISCLAIMER, renew_at=dict(sorted(renew_at.items())),
        horizon=dataset.horizon, time_unit=twin.time_unit,
        from_new=FromNew(events=new.events, conclusion=_new_conclusion(new.events, targets, twin.time_unit)),
        per_unit=PerUnit(issues=tuple(units), moved_share=moved, failed_share=failed,
                         conclusion=_unit_conclusion(moved, failed, targets)),
        long_run=runs, narrowing=narrowing(conns, runs, halfwidth))
