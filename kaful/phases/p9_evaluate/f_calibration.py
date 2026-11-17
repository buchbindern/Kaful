"""Step f: forecast calibration on synthetic truth. The only step of evaluate that reads truth.

For every forecast of the lineage's estimate run (each unit, estimator and issue time) and every quantity:
- quantity: each event (the next time a part reaches its limit after the issue time), and for parts sharing one
  channel the label-invariant first failure of the group (approved 2026-11-16: per-part is not identifiable there);
- coverage of the 90% interval, by class (helpers/metrics.classify), per issue fraction and pooled, with Wilson
  intervals on the pairs and, honestly, on the units;
- alpha-lambda, prognostic horizon and RMSE of the median RUL over pairs with a true crossing ahead.
The 85-95% target is judged on observed quantities of the primary estimator only; joined and unobserved parts are the
prior predictive scored against truth drawn from that prior, reported apart (decided 2026-11-17). Nothing is tuned.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.evaluation import (
    CALIBRATION_TARGET_90,
    PAIR_CLASSES,
    AccuracyScore,
    Calibration,
    CoverageCell,
    PairClass,
    PartClass,
    Verdict,
)
from kaful.artifacts.forecast import Forecast, ForecastSet
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import EvaluateSettings
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.parts import part_classes
from kaful.phases.p9_evaluate.helpers.metrics import (
    Horizon,
    PointError,
    alpha_mass,
    classify,
    median_error,
    prognostic_horizon,
    triggered,
    width,
    wilson,
)
from kaful.runs import ArtifactMismatch, load_artifact, sha256_file
from kaful.scoring.samples import toe_samples
from kaful.scoring.groups import shared_groups
from kaful.scoring.samples import read_table
from kaful.scoring.truth import Unit, load_units, true_next_event

SCORED: tuple[PairClass, ...] = ("covered", "missed", "censored_covered", "censored_missed")
COVERED: tuple[PairClass, ...] = ("covered", "censored_covered")
FLAG_RHO = "rho = {rho} (process noise) is an estimator assumption; synthetic truth has none; it was not tuned here"
FLAG_SHARED = "{bindings}: one channel, not identifiable per part; judged on {quantity}"
NOTE_DECIDABLE = ("at n = {n} units, above vs within target is not decidable; intervals are slightly conservative by "
                  "design (M13: 15-30% wider than the grid oracle; rho adds spread absent from noise-free truth)")
OPEN_LOOP = "open_loop"
FLAG_SKIPPED = "{estimator} skipped on {n} sources: {reason}"


@dataclass(frozen=True)
class Record:
    unit: str
    issue: int
    t: float
    truth: float | None
    cls: PairClass
    alpha_mass: float | None
    alpha_ok: bool | None
    error: PointError | None
    agree: bool | None
    width: float | None  # the 90% interval's width (+inf: open past the horizon), for pairs with a crossing ahead


@dataclass
class Collected:
    part: dict[str, PartClass] = field(default_factory=dict)
    binding_of: dict[str, str] = field(default_factory=dict)
    records: dict[tuple[str, str], list[Record]] = field(default_factory=lambda: defaultdict(list))


def first_failure(bindings: tuple[str, ...]) -> str:
    return f"first_failure({', '.join(bindings)})"


def _record(samples: np.ndarray, truth: float | None, t: float, unit: str, i: int, decidable: bool,
            s: EvaluateSettings) -> Record:
    cls = classify(samples, truth, t, s.interval, decidable)
    if not (decidable and truth is not None and truth > t):
        return Record(unit=unit, issue=i, t=t, truth=truth, cls=cls, alpha_mass=None, alpha_ok=None, error=None,
                      agree=triggered(samples, t) if cls == "already_triggered" else None, width=None)
    mass = alpha_mass(samples, truth, t, s.alpha)
    return Record(unit=unit, issue=i, t=t, truth=truth, cls=cls, alpha_mass=mass, alpha_ok=mass >= s.beta,
                  error=median_error(samples, truth, t), agree=None, width=width(samples, s.interval))


def _issue_rows(machine_dir: Path, path: str, sha: str) -> list[dict[str, str]]:
    target = resolve_in(machine_dir, path)
    if sha256_file(target) != sha:
        raise ArtifactMismatch(f"{path}: content does not match the forecast's samples_sha256")
    return read_table(target)


def collect_forecast(f: Forecast, unit: Unit, twin: TwinDefinition, machine_dir: Path, s: EvaluateSettings,
                     c: Collected) -> None:
    by_id = {b.id: b for b in twin.bindings}
    renewing = {m.binding for m in twin.maintenance if m.trigger == "threshold"}
    decidable = unit.truth.horizon >= f.horizon
    for i, issue in enumerate(f.issues):
        rows = _issue_rows(machine_dir, issue.samples, issue.samples_sha256)
        truths: dict[str, float | None] = {}
        toes: dict[str, np.ndarray] = {}
        for e in issue.events:
            bd = by_id[e.binding]
            truths[e.binding] = true_next_event(unit, e.event, bd.damage_state, bd.threshold_param, issue.t_issue,
                                                e.binding in renewing)
            toes[e.binding] = toe_samples(rows, e.event)
            c.binding_of[e.event] = e.binding
            c.records[(f.estimator, e.event)].append(
                _record(toes[e.binding], truths[e.binding], issue.t_issue, f.source, i, decidable, s))
        for _, bs in shared_groups(twin):
            known = [t for t in (truths[b] for b in bs) if t is not None]
            samples = np.min(np.stack([toes[b] for b in bs]), axis=0)
            c.records[(f.estimator, first_failure(bs))].append(
                _record(samples, min(known) if known else None, issue.t_issue, f.source, i, decidable, s))


def _cell(estimator: str, quantity: str, part: PartClass, fraction: float | None, recs: list[Record],
          s: EvaluateSettings) -> CoverageCell:
    classes = {k: sum(r.cls == k for r in recs) for k in PAIR_CLASSES}
    pairs = sum(classes[k] for k in SCORED)
    covered = sum(classes[k] for k in COVERED)
    units = len({r.unit for r in recs if r.cls in SCORED})
    share = covered / pairs if pairs else None
    return CoverageCell(estimator=estimator, quantity=quantity, part_class=part, issue_fraction=fraction,
                        classes=classes, pairs=pairs, covered=covered, units=units, share=share,
                        triggered_agree=sum(bool(r.agree) for r in recs),
                        ci_pairs=wilson(covered, pairs, s.ci_z),
                        ci_units=wilson(share * units, units, s.ci_z) if share is not None else None)


@dataclass(frozen=True)
class Sharpness:
    relative_median: float | None
    censored: int
    ratio_open_loop: float | None


def _sharpness(recs: list[Record], open_loop: dict[tuple[str, int], float]) -> Sharpness:
    """Median width / true RUL; and, per (unit, issue) both have finite, this width / open-loop's."""
    ahead = [r for r in recs if r.width is not None and r.truth is not None]
    rel = sorted(r.width / (r.truth - r.t) for r in ahead if r.width is not None and r.truth is not None)
    med = float(np.quantile(rel, 0.5, method="inverted_cdf")) if rel else None
    ratios = [r.width / open_loop[(r.unit, r.issue)] for r in ahead if r.width is not None and np.isfinite(r.width)
              and np.isfinite(open_loop.get((r.unit, r.issue), np.inf)) and open_loop[(r.unit, r.issue)] > 0]
    return Sharpness(relative_median=med if med is not None and np.isfinite(med) else None,
                     censored=sum(not np.isfinite(r.width) for r in ahead if r.width is not None),
                     ratio_open_loop=float(median(ratios)) if ratios else None)


def _horizons(recs: list[Record]) -> list[Horizon]:
    out = []
    for unit in sorted({r.unit for r in recs}):
        mine = sorted((r for r in recs if r.unit == unit and r.cls != "undecidable"), key=lambda r: r.issue)
        h = prognostic_horizon([(r.t, r.truth, bool(r.alpha_ok)) for r in mine])
        if h is not None:
            out.append(h)
    return out


def _accuracy(estimator: str, quantity: str, part: PartClass, recs: list[Record],
              open_loop: dict[tuple[str, int], float]) -> AccuracyScore:
    al = [r.alpha_ok for r in recs if r.alpha_ok is not None]
    errs = [r.error for r in recs if r.error is not None]
    logs = [abs(e.log_ratio) for e in errs if e.log_ratio is not None]
    ahead = [r for r in recs if r.alpha_ok is not None]
    horizons, sharp = _horizons(recs), _sharpness(recs, open_loop)
    return AccuracyScore(
        estimator=estimator, quantity=quantity, part_class=part, alpha_lambda_pass=sum(al), alpha_lambda_n=len(al),
        alpha_lambda=sum(al) / len(al) if al else None,
        alpha_mass_median=median(r.alpha_mass for r in ahead if r.alpha_mass is not None) if ahead else None,
        horizon_units=len(horizons),
        horizon_median=median(h.value for h in horizons) if horizons else None,
        horizon_share_median=median(h.share for h in horizons) if horizons else None,
        rmse=float(np.sqrt(np.mean([e.error ** 2 for e in errs]))) if errs else None,
        rmse_relative=float(np.sqrt(np.mean([e.relative ** 2 for e in errs]))) if errs else None,
        rmse_n=len(errs), rmse_excluded=len(ahead) - len(errs),
        abs_relative_error_median=float(median(abs(e.relative) for e in errs)) if errs else None,
        abs_log_error_median=float(median(logs)) if logs else None,
        width_relative_median=sharp.relative_median, width_censored=sharp.censored,
        width_ratio_open_loop=sharp.ratio_open_loop if estimator != "open_loop" else None)


def _verdict(cell: CoverageCell) -> Verdict | None:
    if cell.share is None or cell.ci_units is None:
        return None
    lo, hi = CALIBRATION_TARGET_90
    word = "within" if lo <= cell.share <= hi else "above" if cell.share > hi else "below"
    overlap = cell.ci_units.low <= hi and cell.ci_units.high >= lo
    return Verdict(quantity=cell.quantity, coverage=cell.share, units=cell.units, ci_units=cell.ci_units, verdict=word,
                   ci_overlaps_target=overlap)


def _notes(verdicts: list[Verdict]) -> tuple[str, ...]:
    """Decided 2026-11-17: when every verdict's interval overlaps the target, say why above vs within is open."""
    if not verdicts or not all(v.ci_overlaps_target for v in verdicts):
        return ()
    lo, hi = min(v.units for v in verdicts), max(v.units for v in verdicts)
    return (NOTE_DECIDABLE.format(n=f"{lo}" if lo == hi else f"{lo}-{hi}"),)


def _flags(fs: ForecastSet, twin: TwinDefinition) -> tuple[str, ...]:
    flags = {FLAG_RHO.format(rho=fs.settings.get("process_noise_cv"))}
    flags |= {FLAG_SHARED.format(bindings=", ".join(bs), quantity=first_failure(bs)) for _, bs in shared_groups(twin)}
    by: dict[tuple[str, str], int] = defaultdict(int)
    for sk in fs.skipped:
        by[(sk.estimator, sk.reason)] += 1
    flags |= {FLAG_SKIPPED.format(estimator=e, n=n, reason=r) for (e, r), n in by.items()}
    return tuple(sorted(flags))


def calibration(lineage: Lineage, machine_dir: Path, s: EvaluateSettings) -> Calibration:
    """Called only when estimate was reached on a synthetic dataset."""
    fs_ref, ds_ref = lineage.ref("estimate", "forecast_set"), lineage.ref("synthesize", "synthetic_dataset")
    twin_ref = lineage.ref("verify", "twin_definition") or lineage.ref("build", "twin_definition")
    if fs_ref is None or ds_ref is None or twin_ref is None:
        raise ValueError(f"estimate reached but its forecast set, dataset or twin is missing in run {lineage.run_id}")
    fs = load_artifact(machine_dir, fs_ref, ForecastSet)
    dataset = load_artifact(machine_dir, ds_ref, SyntheticDataset)
    twin = load_artifact(machine_dir, twin_ref, TwinDefinition)
    units = load_units(machine_dir, dataset)
    c = Collected(part=part_classes(twin))
    for entry in fs.forecasts:
        ref = ArtifactRef(kind="forecast", run_id=fs.run_id, path=entry.path, sha256=entry.sha256)
        collect_forecast(load_artifact(machine_dir, ref, Forecast), units[entry.source], twin, machine_dir, s, c)
    fractions = [float(x) for x in fs.settings.get("issue_fractions", [])]  # type: ignore[union-attr]
    cells, accuracy = [], []
    for (est, q), recs in sorted(c.records.items()):
        part = "observed" if q.startswith("first_failure(") else c.part[c.binding_of[q]]
        cells.append(_cell(est, q, part, None, recs, s))
        cells += [_cell(est, q, part, fr, [r for r in recs if r.issue == i], s) for i, fr in enumerate(fractions)]
        ol = {(r.unit, r.issue): r.width for r in c.records.get((OPEN_LOOP, q), []) if r.width is not None}
        accuracy.append(_accuracy(est, q, part, recs, ol))
    verdicts = [v for cell in cells if cell.estimator == s.primary_estimator and cell.issue_fraction is None
                and cell.part_class == "observed" and (v := _verdict(cell)) is not None]
    return Calibration(forecast_set=fs_ref, dataset=ds_ref, interval=s.interval, alpha=s.alpha, beta=s.beta,
                       primary_estimator=s.primary_estimator,
                       process_noise_cv=float(fs.settings.get("process_noise_cv", 0.0)),  # type: ignore[arg-type]
                       cells=tuple(cells), accuracy=tuple(accuracy), verdicts=tuple(verdicts),
                       notes=_notes(verdicts), flags=_flags(fs, twin))
