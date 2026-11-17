"""The parameterize report (M9): provenance counts and the sensitivity of RUL to every prior. No API calls.

For each full-model component, RUL at t = 0 under a constant operating profile is D_limit / (k * f(profile)),
drawn by Monte Carlo from the priors (seeded). Profiles (settings.parameterize.sensitivity_profiles) scale the
load or speed, or raise the temperature; a profile a component cannot respond to (no load exponent, no nominal
temperature documented or declared) is reported as not evaluable, never silently nominal.
- share: the part of var(log RUL) a prior explains (1 - variance with it frozen at its median / variance);
- tornado: median RUL with the prior at its 16th and 84th percentile (about -1 and +1 sd), others at median.
The assumptions behind each life prior are listed: the interval rule (q, sigma, why) and, for a calibrated
life, the median it would have under each other rule; the duty, throughput and judgement settings used.
Judged (llm_judgement) lives come first, sorted by how much RUL spread they carry. Saved as data:
outputs/param_reports/<run_id>.json and .md.
"""

import math
from statistics import NormalDist

import numpy as np
from pydantic import Field

from kaful.artifacts.base import SOURCE_KINDS, StrictModel
from kaful.artifacts.param_set import ComponentTemplate, ParamSet
from kaful.config import ParameterizeSettings
from kaful.phases.p4_parameterize.c_life_priors import category
from kaful.sensitivity import Effect, component_sensitivity, rul

Z = NormalDist().inv_cdf
FLAG_ALL_JUDGED = "every full-model life on this machine is an llm_judgement: the whole model rests on judged lives"


class ProfileRUL(StrictModel):
    profile: str
    evaluable: bool
    note: str = ""
    median: float | None = None
    p05: float | None = None
    p95: float | None = None


class ComponentReport(StrictModel):
    component: str
    family: str
    life_source: str
    life_ref: str
    sigma_log_rul: float = Field(description="sd of log RUL at nominal: how much spread this component carries.")
    profiles: tuple[ProfileRUL, ...]
    effects: tuple[Effect, ...]
    assumption: str = Field(description="The rule or judgement behind the life prior, with its numbers.")
    alternatives: dict[str, float] = Field(default={}, description="Calibrated life: median under each other rule.")
    flags: tuple[str, ...] = ()


class ParamReport(StrictModel):
    machine_id: str
    run_id: str
    time_unit: str
    by_source: dict[str, int]
    judged_first: tuple[str, ...] = Field(description="Components whose life is llm_judgement, most spread first.")
    components: tuple[ComponentReport, ...]
    settings_used: dict[str, str]
    flags: tuple[str, ...] = ()


def _assumption(component: str, ps: ParamSet, p: ParameterizeSettings) -> tuple[str, dict[str, float]]:
    life = ps.get(f"{component}.life_nominal")
    stmt = next((s for s in ps.life_statements if s.id == life.ref), None)
    if life.source == "llm_judgement":
        return (f"llm_judgement ({life.ref}); log-sd floored at {p.judgement_sigma_floor}", {})
    if stmt is None or stmt.bound != "lower" or stmt.in_time_unit is None:
        return (f"{life.source} ({life.ref})", {})
    cat = category(stmt)
    rule = p.interval_priors[cat]
    alts = {k: round(stmt.in_time_unit * math.exp(-Z(r.q) * r.sigma), 3) for k, r in sorted(p.interval_priors.items())}
    return (f"{cat}: P(T < {stmt.in_time_unit:g}) = {rule.q}, log-sd {rule.sigma} ({rule.why})", alts)


def _component(t: ComponentTemplate, ps: ParamSet, p: ParameterizeSettings) -> ComponentReport | None:
    own = {x.name: x for x in ps.parameters if x.component == t.component}
    if "k" not in own or "damage_limit" not in own:
        return None
    n, seed = p.sensitivity_samples, p.sensitivity_seed
    profiles = []
    for name, profile in p.sensitivity_profiles.items():
        samples, note = rul(t, own, profile, n, seed)
        profiles.append(ProfileRUL(profile=name, evaluable=False, note=note) if samples is None else ProfileRUL(
            profile=name, evaluable=True, median=float(np.median(samples)), p05=float(np.quantile(samples, 0.05)),
            p95=float(np.quantile(samples, 0.95))))
    sens = component_sensitivity(t, own, n, seed)
    var, effects = sens.var_log_rul, sens.effects
    life = own["life_nominal"]
    assumption, alts = _assumption(t.component, ps, p)
    return ComponentReport(component=t.component, family=t.family, life_source=life.source, life_ref=life.ref,
                           sigma_log_rul=round(math.sqrt(var), 4), profiles=tuple(profiles),
                           effects=effects,
                           assumption=assumption, alternatives=alts, flags=t.flags)


def build_report(ps: ParamSet, p: ParameterizeSettings) -> ParamReport:
    comps = [c for c in (_component(t, ps, p) for t in ps.templates) if c is not None]
    judged = sorted((c for c in comps if c.life_source == "llm_judgement"),
                    key=lambda c: (-c.sigma_log_rul, c.component))
    order = [*judged, *sorted((c for c in comps if c.life_source != "llm_judgement"), key=lambda c: c.component)]
    flags = (FLAG_ALL_JUDGED,) if comps and len(judged) == len(comps) else ()
    used = {f"interval_priors.{k}": f"q {r.q}, log-sd {r.sigma}: {r.why}" for k, r in sorted(p.interval_priors.items())}
    used |= {f"duty_defaults.{k}": f"{d.per_year_median:g} per year, log-sd {d.sigma}: {d.why}"
             for k, d in sorted(p.duty_defaults.items())}
    used |= {f"throughput_defaults.{k}": f"{d.median:g} {d.unit}, log-sd {d.sigma}: {d.why}"
             for k, d in sorted(p.throughput_defaults.items())}
    used |= {"judgement_sigma_floor": str(p.judgement_sigma_floor), "rated_life_weibull_beta": str(p.rated_life_weibull_beta)}
    return ParamReport(machine_id=ps.machine_id, run_id=ps.run_id, time_unit=ps.time_unit,
                       by_source={s: ps.counts_by_source()[s] for s in SOURCE_KINDS},
                       judged_first=tuple(c.component for c in judged), components=tuple(order),
                       settings_used=used, flags=flags)


def _n(x: float | None) -> str:
    return "-" if x is None else f"{x:,.4g}"


def report_md(r: ParamReport) -> str:
    lines = [f"# {r.machine_id}: parameters and RUL sensitivity (run {r.run_id})", "",
             f"Time unit: {r.time_unit}. Parameters by source: "
             + ", ".join(f"{k} {v}" for k, v in r.by_source.items() if v) + ".", ""]
    lines += [f"**{f}**" for f in r.flags] + ([""] if r.flags else [])
    lines += ["## Life priors, judged first (most RUL spread first)", "",
              "| component | family | life source | sd log RUL | RUL nominal: median (5–95%) | load +20% | speed +20% | "
              "temperature +10 K | assumption |", "|---|---|---|---|---|---|---|---|---|"]
    for c in r.components:
        cells = {pr.profile: (f"{_n(pr.median)} ({_n(pr.p05)}–{_n(pr.p95)})" if pr.evaluable
                              else f"not evaluable: {pr.note}") for pr in c.profiles}
        lines.append(f"| {c.component} | {c.family} | {c.life_source} | {c.sigma_log_rul} | {cells.get('nominal')} | "
                     f"{cells.get('load +20%')} | {cells.get('speed +20%')} | {cells.get('temperature +10 K')} | "
                     f"{c.assumption} |")
    lines += ["", "## Effect of each prior on RUL (nominal profile)", "",
              "| parameter | source | share of var(log RUL) | median RUL at -1 sd / +1 sd |", "|---|---|---|---|"]
    lines += [f"| {e.param} | {e.source} | {e.share:.2f} | {_n(e.low_median)} / {_n(e.high_median)} |"
              for c in r.components for e in c.effects]
    alts = [c for c in r.components if c.alternatives]
    if alts:
        lines += ["", "## Interval assumption: median life under each rule", "",
                  "| component | " + " | ".join(sorted(alts[0].alternatives)) + " |",
                  "|---|" + "---|" * len(alts[0].alternatives)]
        lines += [f"| {c.component} | " + " | ".join(_n(v) for _, v in sorted(c.alternatives.items())) + " |" for c in alts]
    lines += ["", "## Settings used", ""] + [f"- `{k}`: {v}" for k, v in r.settings_used.items()]
    return "\n".join(lines) + "\n"
