"""The estimate phase: the verified twin + ParamSet + a data source -> forecasts per estimator (M13). No LLM call.

Data: the synthetic dataset of a synthesize run (real-data sources come with streaming, parked). Per source, in
the order `estimate_source` runs them:
  a_open_sources      the source's rows (a FileSource over its CSV; no truth) and its monitored event channels
  b_start_estimators  open-loop over every group; the particle filter and the UKF over the observed groups only
                      (unobserved groups are shared from the open-loop run: identical by construction)
  c_filter            rows fed one at a time; a per-row trace; beliefs at the issue times
  d_forecast          per issue time, the next time each part reaches its limit (every sample kept), censoring
                      counts, percentiles, parameter and health summaries
  e_write             the forecast artifacts and their sidecar CSVs
Sources run in parallel workers when settings.estimate.workers > 1 (helpers/workers); results are written in source
order. A source the data rule out under the model (every particle impossible) fails the phase, naming it.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.forecast import ForecastEntry, ForecastSet, GroupRecord, OperatingProfile, SkippedEstimator
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends import load_backend
from kaful.config import EstimateSettings, EstimatorName, KafulSettings
from kaful.dynamics.step import simulation_step
from kaful.estimation.groups import groups
from kaful.estimation.resampling import DegenerateWeights
from kaful.estimation.summaries import medians
from kaful.phases import PhaseName
from kaful.phases.p8_estimate.a_open_sources import Source, read_rows, sources
from kaful.phases.p8_estimate.b_start_estimators import (
    observed_bindings,
    predict_seed,
    start,
    ukf_skip_reason,
    unobserved_bindings,
)
from kaful.phases.p8_estimate.c_filter import Filtered, issue_rows, run_filter
from kaful.phases.p8_estimate.d_forecast import forecast_issue, merge, merge_beliefs, samples_csv, summarize_issue
from kaful.phases.p8_estimate.e_write import build, group_record, trace_csv
from kaful.phases.p8_estimate.helpers.results import IssueResult, SourceResult
from kaful.phases.p8_estimate.helpers.workers import run_in_order
from kaful.runs import RunContext, load_artifact, now_utc, write_text_atomic

log = logging.getLogger("kaful.estimate")

FLAG_RHO = ("process noise: rho = {rho} (settings.estimate.process_noise_cv) is an estimator assumption, not a "
            "physical parameter; synthetic truth has no process noise")
FLAG_SHARED = ("{bindings} share the channel {channel}: which of them degrades is not identifiable from it (label "
               "switching); judge them jointly (sorted rates, first failure, the channel's reading)")


@dataclass(frozen=True)
class Job:
    base: dict[str, Any]
    twin: TwinDefinition
    ps: ParamSet
    contract: SensorContract
    dataset: SyntheticDataset
    settings: EstimateSettings
    steps_per_shortest_life: int
    backend: str
    machine_dir: Path
    phase_dir: Path
    source: Source


def flags(twin: TwinDefinition, settings: EstimateSettings) -> tuple[str, ...]:
    out = {FLAG_RHO.format(rho=settings.process_noise_cv)}
    for g in groups(twin):
        for ch in g.channels:
            on = sorted({o.binding for o in g.observations if o.channel == ch and o.form == "linear"})
            if len(on) > 1:
                out.add(FLAG_SHARED.format(bindings=", ".join(on), channel=ch))
    return tuple(sorted(out))


def ignored_channels(twin: TwinDefinition, profile: OperatingProfile) -> tuple[str, ...]:
    active = {i.channel for i in twin.external_inputs if i.provenance == "channel" and i.channel}
    rules = {m.trigger_channel for m in twin.maintenance if m.trigger_channel}
    used = {k for s in profile.segments for k in s.values} - active
    return tuple(sorted(used | ({e.channel for e in profile.events} - rules)))


def estimate_source(job: Job) -> SourceResult:
    """Steps a to e for one source, every estimator."""
    out, s, src = SourceResult(), job.settings, job.source
    rows = read_rows(src, job.contract, job.dataset)
    if not rows:
        out.skipped += [SkippedEstimator(source=src.source_id, estimator=m, reason="the source has no rows")
                        for m in s.methods]
        return out
    backend = load_backend(job.backend)
    built = backend.build(job.twin, job.ps, job.contract)
    dt, fixed = simulation_step(job.twin, job.ps, job.steps_per_shortest_life), medians(job.ps)
    ol = start(backend, built, job.ps, job.dataset.profile, "open_loop", dt, s, src.spawn_key, src.monitored)
    issues = issue_rows([r.t for r in rows], s.issue_fractions)
    ol_filtered = run_filter(ol, rows, issues, job.twin, fixed)
    shared = [forecast_issue(backend, built, b, job.dataset.profile, dt, job.dataset.horizon, s,
                             predict_seed(s, src.spawn_key, i), unobserved_bindings(job.twin))
              for i, b in enumerate(ol_filtered.beliefs)]
    for method in s.methods:
        try:
            _one(job, method, backend, built, rows, issues, ol, ol_filtered, shared, dt, out)
        except DegenerateWeights as exc:
            out.failures.append(f"{src.source_id} {method}: {exc}")
    return out


def _one(job: Job, method: EstimatorName, backend: Any, built: Any, rows: Any, issues: Any, ol: Any,
         ol_filtered: Filtered, shared: list[Any], dt: float, out: SourceResult) -> None:
    s, src = job.settings, job.source
    if method == "open_loop":
        filtered, records, merged_ol = ol_filtered, [group_record(f, method, False) for f in ol.filters], None
    else:
        est = start(backend, built, job.ps, job.dataset.profile, method, dt, s, src.spawn_key, src.monitored)
        reason = ukf_skip_reason(est) if method == "ukf" else None
        if reason:
            out.skipped.append(SkippedEstimator(source=src.source_id, estimator=method, reason=reason))
            return
        filtered = run_filter(est, rows, issues, job.twin, medians(job.ps))
        records = [group_record(f, method, False) for f in getattr(est, "filters")]
        records += [group_record(f, method, True) for f in ol.filters if not f.group.observed]
        merged_ol = ol_filtered
    trace = [{**o, **own} for o, own in zip(ol_filtered.trace, filtered.trace)]
    results = _issues(job, built, backend, filtered, shared, dt, merged_ol)
    records.sort(key=lambda r: r.bindings)
    build(job.base, src.source_id, src.spawn_key, method, s, job.dataset.profile, job.dataset.horizon,
          ignored_channels(job.twin, job.dataset.profile), records, results, trace_csv(trace),
          flags(job.twin, s), job.machine_dir, job.phase_dir, out)


def _issues(job: Job, built: Any, backend: Any, filtered: Filtered, shared: list[Any], dt: float,
            open_loop: Filtered | None) -> list[IssueResult]:
    out = []
    fixed = medians(job.ps)
    for i, belief in enumerate(filtered.beliefs):
        own = forecast_issue(backend, built, belief, job.dataset.profile, dt, job.dataset.horizon, job.settings,
                             predict_seed(job.settings, job.source.spawn_key, i), observed_bindings(job.twin))
        pred = merge([own, shared[i]])
        whole = merge_beliefs(belief, open_loop.beliefs[i] if open_loop else None)
        events, params, health = summarize_issue(pred, whole, job.twin, job.ps, fixed)
        out.append(IssueResult(t_issue=belief.t, events=events, params=params, health=health,
                               samples=samples_csv(pred)))
    return out


def recorded_settings(s: EstimateSettings) -> dict[str, Any]:
    """The settings that shape the outputs (not `workers`)."""
    out: dict[str, Any] = {}
    for k, v in sorted(s.model_dump(exclude={"workers"}).items()):
        out[k] = list(v) if isinstance(v, tuple) else v
    return out


class EstimatePhase:
    name: PhaseName = "estimate"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started, clock = now_utc(), time.perf_counter()
        try:
            refs = {p: {r.kind: r for r in ctx.manifest.phase(p).artifacts}
                    for p in ("verify", "parameterize", "understand", "synthesize")}
            if "synthetic_dataset" not in refs["synthesize"]:
                raise FileNotFoundError("no data source: estimate reads the synthetic dataset of a synthesize run "
                                        "(pass --base-run; real-data sources come with streaming, parked)")
            loaded = [self._load(ctx, refs[p], k, m) for p, k, m in (
                ("verify", "twin_definition", TwinDefinition), ("parameterize", "param_set", ParamSet),
                ("understand", "sensor_contract", SensorContract), ("synthesize", "synthetic_dataset", SyntheticDataset))]
        except FileNotFoundError as exc:
            log.error("estimate cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        twin, ps, contract, dataset = loaded
        inputs = tuple(sorted((refs["verify"]["twin_definition"], refs["parameterize"]["param_set"],
                               refs["understand"]["sensor_contract"], refs["synthesize"]["synthetic_dataset"]),
                              key=lambda r: (r.kind, r.path)))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": twin.index_hash,
                "inputs": inputs}
        done = estimate(base, twin, ps, contract, dataset, ctx.settings, ctx.machine_dir, ctx.run_dir / self.name,
                        lambda a, n: ctx.save_artifact(self.name, a, n))
        if done.failures:
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=done.failures)
        ref = ctx.save_artifact(self.name, done.forecast_set)
        seconds = time.perf_counter() - clock
        log.info("estimate finished", extra={"fields": {"seconds": round(seconds, 1)}})
        return PhaseRecord(phase=self.name, status="passed", started_at=started, finished_at=now_utc(),
                           artifacts=(ref,), notes=(*done.forecast_set.flags,
                                                    f"{len(done.forecast_set.forecasts)} forecasts in {seconds:.0f} s"))

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)


@dataclass(frozen=True)
class Estimated:
    forecast_set: ForecastSet
    failures: tuple[str, ...]


def estimate(base: dict[str, Any], twin: TwinDefinition, ps: ParamSet, contract: SensorContract,
             dataset: SyntheticDataset, settings: KafulSettings, machine_dir: Path, phase_dir: Path,
             save: Any) -> Estimated:
    """Every source of the dataset, every estimator; files written in source order."""
    s = settings.estimate
    jobs = [Job(base=base, twin=twin, ps=ps, contract=contract, dataset=dataset, settings=s,
                steps_per_shortest_life=settings.verify.steps_per_shortest_life, backend=settings.build.backend,
                machine_dir=machine_dir, phase_dir=phase_dir, source=src)
            for src in sources(dataset, contract, machine_dir)]
    entries, skipped, failures = [], [], []
    for result in run_in_order(estimate_source, jobs, s.workers):
        for path, text in sorted(result.files.items()):
            write_text_atomic(path, text)
        for forecast, name in result.forecasts:
            ref = save(forecast, name)
            entries.append(ForecastEntry(source=forecast.source, estimator=forecast.estimator, path=ref.path,
                                         sha256=ref.sha256))
        skipped += result.skipped
        failures += result.failures
    entries.sort(key=lambda e: (e.source, e.estimator))
    fs = ForecastSet(**base, time_unit=twin.time_unit, settings=recorded_settings(s), forecasts=tuple(entries),
                     skipped=tuple(sorted(skipped, key=lambda e: (e.source, e.estimator))),
                     flags=flags(twin, s), disclaimer=dataset.disclaimer)
    return Estimated(forecast_set=fs, failures=tuple(failures))


__all__ = ["EstimatePhase", "GroupRecord", "estimate", "estimate_source"]
