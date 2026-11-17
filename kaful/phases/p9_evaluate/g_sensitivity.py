"""Step g: sensitivity. Which priors dominate RUL uncertainty, what the data did to them, and which data would narrow
each part most.

- prior: per full-model component, each prior's first-order share of var(log RUL) at t = 0 (kaful.sensitivity, the
  computation of M9's report, on the lineage's ParamSet with settings.parameterize's samples and seed);
- posterior (when the lineage has forecasts): per carried parameter, at the primary estimator's last issue time,
  median over units of its 90% width contraction (posterior / prior) and of its share of the remaining log
  time-to-limit spread (correlation ratio over quantile bins);
- recommendations: rules (settings.evaluate.recommendations) on what observes the part and on its dominant prior.
Rho, the process noise, is an estimator assumption, listed apart, never a prior.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.evaluation import ComponentSensitivity, PartClass, PosteriorParam, PriorEffect, Sensitivity
from kaful.artifacts.forecast import Forecast, ForecastSet, ParamSummary
from kaful.artifacts.param_set import ComponentTemplate, ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import EvaluateSettings, ParameterizeSettings
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.parts import part_classes
from kaful.runs import load_artifact
from kaful.scoring.samples import toe_samples
from kaful.scoring.groups import shared_groups
from kaful.scoring.samples import read_table
from kaful.sensitivity import component_sensitivity, own_parameters

LOAD_RATINGS_FLAG = "load ratings needed"
NOTE_NO_FORECASTS = "no forecasts in the lineage: posterior contraction and remaining shares not reported"


def contraction(p: ParamSummary) -> float | None:
    """Posterior 90% width / prior 90% width, on the log scale when the parameter is positive."""
    lo, hi, plo, phi = p.prior["p05"], p.prior["p95"], p.posterior["p05"], p.posterior["p95"]
    if min(lo, plo) > 0:
        lo, hi, plo, phi = (float(np.log(x)) for x in (lo, hi, plo, phi))
    return (phi - plo) / (hi - lo) if hi > lo else None


def correlation_ratio(x: np.ndarray, y: np.ndarray, bins: int) -> float | None:
    """Var(E[y | x-bin]) / Var(y) over quantile bins of x: the first-order share of y's spread x explains."""
    if np.ptp(x) == 0 or np.var(y) == 0:
        return None
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    idx = np.searchsorted(edges, x, side="right")
    mean = float(np.mean(y))
    between = sum(np.sum(idx == b) * (float(np.mean(y[idx == b])) - mean) ** 2 for b in np.unique(idx))
    return float(min(1.0, between / float(np.sum((y - mean) ** 2))))


@dataclass(frozen=True)
class LifeParams:
    k: str
    damage_limit: str
    damage_limit_median: float
    prior_log_width: float


def life_params(ps: ParamSet, twin: TwinDefinition, interval: tuple[float, float]) -> dict[str, LifeParams]:
    """Binding -> what its life D_limit / k is made of, and the log 90% width of its life_nominal prior."""
    out = {}
    for b in twin.bindings:
        k, life = b.params.get("k"), f"{b.component}.life_nominal"
        if k is None or life not in {x.param_id for x in ps.parameters}:
            continue
        lo, hi = (ps.get(life).prior.quantile(q) for q in interval)
        out[b.id] = LifeParams(k=k, damage_limit=b.threshold_param,
                               damage_limit_median=float(ps.get(b.threshold_param).prior.quantile(0.5)),
                               prior_log_width=float(np.log(hi / lo)))
    return out


def _life_contraction(rows: list[dict[str, str]], lp: LifeParams, interval: tuple[float, float]) -> float | None:
    if not rows or lp.k not in rows[0] or lp.prior_log_width <= 0:
        return None
    k = np.array([float(r[lp.k]) for r in rows])
    dl = (np.array([float(r[lp.damage_limit]) for r in rows]) if lp.damage_limit in rows[0]
          else np.full(k.size, lp.damage_limit_median))
    lo, hi = (float(np.quantile(np.log(dl / k), q)) for q in interval)
    return (hi - lo) / lp.prior_log_width


def _posterior_unit(f: Forecast, machine_dir: Path, s: EvaluateSettings, lives: dict[str, LifeParams],
                    contr: dict[str, list[float]], shares: dict[str, list[float]]) -> None:
    last = f.issues[-1]
    for p in last.params:
        c = contraction(p)
        if c is not None:
            contr[p.param_id].append(c)
    rows = read_table(machine_dir / last.samples)
    for binding, lp in lives.items():
        lc = _life_contraction(rows, lp, s.interval)
        if lc is not None:
            contr[f"{binding}|life"].append(lc)
    columns = [p.param_id for p in last.params]
    for e in last.events:
        toe = toe_samples(rows, e.event)
        live = np.isfinite(toe) & (toe > last.t_issue)
        if live.sum() < s.posterior_min_samples:
            continue
        y = np.log(toe[live] - last.t_issue)
        for col in columns:
            r = correlation_ratio(np.array([float(row[col]) for row in rows])[live], y, s.posterior_bins)
            if r is not None:
                shares[f"{e.binding}|{col}"].append(r)


@dataclass(frozen=True)
class Posterior:
    params: dict[str, tuple[PosteriorParam, ...]]  # binding -> its posterior parameters
    life: dict[str, float]  # binding -> median life contraction


def posterior(fs: ForecastSet, machine_dir: Path, s: EvaluateSettings, lives: dict[str, LifeParams]) -> Posterior:
    """Contraction of each part's own parameters and life; shares of whatever explains its remaining spread."""
    contr: dict[str, list[float]] = defaultdict(list)
    shares: dict[str, list[float]] = defaultdict(list)
    for entry in fs.forecasts:
        if entry.estimator == s.primary_estimator:
            ref = ArtifactRef(kind="forecast", run_id=fs.run_id, path=entry.path, sha256=entry.sha256)
            _posterior_unit(load_artifact(machine_dir, ref, Forecast), machine_dir, s, lives, contr, shares)
    out: dict[str, dict[str, PosteriorParam]] = defaultdict(dict)
    for key, v in sorted(shares.items()):
        binding, param = key.split("|")
        own = contr.get(param, []) if param.split(".")[0] == binding else []
        out[binding][param] = PosteriorParam(param=param, contraction=median(own) if own else None,
                                             remaining_share=median(v), units=len(v))
    life = {key.split("|")[0]: median(v) for key, v in sorted(contr.items()) if key.endswith("|life")}
    return Posterior(params={b: tuple(v[k] for k in sorted(v)) for b, v in out.items()}, life=life)


def recommendations(component: str, binding: str, part: PartClass, effects: tuple[PriorEffect, ...],
                    template: ComponentTemplate, life_source: str, twin: TwinDefinition,
                    s: EvaluateSettings) -> tuple[str, ...]:
    r = s.recommendations
    out = []
    if part == "not_identifiable":
        group = next(bs for _, bs in shared_groups(twin) if binding in bs)
        out.append(r["shared_channel"].format(bindings=", ".join(group)))
    elif part == "joined":
        sources = sorted({c.from_binding for c in twin.connections if c.to_binding == binding})
        out.append(r["joined"].format(component=component, sources=", ".join(sources) or "a connection"))
    elif part == "unobserved":
        out.append(r["unobserved"].format(component=component))
    if life_source == "llm_judgement":
        out.append(r["judged_life"].format(component=component))
    if any(f.startswith(LOAD_RATINGS_FLAG) for f in template.flags):
        out.append(r["load_ratings"].format(component=component))
    if effects and effects[0].source == "default":
        out.append(r["default_value"].format(param=effects[0].param))
    return tuple(out)


def _component(t: ComponentTemplate, ps: ParamSet, twin: TwinDefinition, parts: dict[str, PartClass],
               post: Posterior, p: ParameterizeSettings,
               s: EvaluateSettings) -> ComponentSensitivity | None:
    own = own_parameters(t, ps)
    binding = next((b.id for b in twin.bindings if b.component == t.component), None)
    if "k" not in own or "damage_limit" not in own or binding is None:
        return None
    sens = component_sensitivity(t, own, p.sensitivity_samples, p.sensitivity_seed)
    roles = {f"{t.component}.{name}": x.role for name, x in own.items()}
    effects = tuple(PriorEffect(param=e.param, source=e.source, role=roles[e.param], share=e.share,
                                low_median=e.low_median, high_median=e.high_median) for e in sens.effects)
    mine = post.params.get(binding, ())
    k = post.life.get(binding)
    sigma = float(np.sqrt(sens.var_log_rul))
    life = own["life_nominal"].source if "life_nominal" in own else own["k"].source
    return ComponentSensitivity(
        component=t.component, part_class=parts[binding], life_source=life, sigma_log_rul=sigma, effects=effects,
        posterior=mine, life_contraction=k, remaining_spread=sigma * (1.0 if k is None else k),
        recommendations=recommendations(t.component, binding, parts[binding], effects, t, life, twin, s))


def by_source(components: list[ComponentSensitivity]) -> dict[str, float]:
    totals: dict[str, list[float]] = defaultdict(list)
    sources = sorted({e.source for c in components for e in c.effects})
    for c in components:
        total = sum(e.share for e in c.effects)
        for src in sources:
            totals[src].append(sum(e.share for e in c.effects if e.source == src) / total if total else 0.0)
    return {src: float(np.mean(v)) for src, v in sorted(totals.items())}


def sensitivity(lineage: Lineage, machine_dir: Path, p: ParameterizeSettings, s: EvaluateSettings) -> Sensitivity:
    """Called only when parameterize and verify were reached; the posterior part needs estimate."""
    ps_ref = lineage.ref("parameterize", "param_set")
    twin_ref = lineage.ref("verify", "twin_definition") or lineage.ref("build", "twin_definition")
    if ps_ref is None or twin_ref is None:
        raise ValueError(f"parameterize and verify reached but the ParamSet or twin is missing in {lineage.run_id}")
    ps, twin = load_artifact(machine_dir, ps_ref, ParamSet), load_artifact(machine_dir, twin_ref, TwinDefinition)
    fs_ref = lineage.ref("estimate", "forecast_set") if lineage.reached("estimate") else None
    fs = load_artifact(machine_dir, fs_ref, ForecastSet) if fs_ref else None
    post = posterior(fs, machine_dir, s, life_params(ps, twin, s.interval)) if fs else Posterior(params={}, life={})
    parts = part_classes(twin)
    comps = [c for c in (_component(t, ps, twin, parts, post, p, s) for t in ps.templates) if c is not None]
    comps.sort(key=lambda c: (-c.remaining_spread, c.component))
    assumptions = tuple(f for f in fs.flags if "estimator assumption" in f) if fs else ()
    return Sensitivity(param_set=ps_ref, samples=p.sensitivity_samples, seed=p.sensitivity_seed,
                       components=tuple(comps), variance_by_source=by_source(comps),
                       estimator_assumptions=assumptions, notes=() if fs else (NOTE_NO_FORECASTS,))
