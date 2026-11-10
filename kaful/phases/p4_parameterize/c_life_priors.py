"""Step c: per full-model component, a prior on its nominal life T, and from it k and D_limit (no LLM).

The binding statement is a usable point life if there is one (rated life, MTBF), else the largest usable lower
bound in the machine's time unit (a calendar one is compared at the default duty cycle's median). Priors are
lognormal (settings.parameterize, approved 2026-11-09; shown as assumptions in the sensitivity report):
- a lower bound I of category c (consumable replacement, replacement, service, machine-wide):
  P(T < I) = q_c, log-sd sigma_c, so median = I * exp(-z(q_c) * sigma_c);
- a calendar interval: I in years times the duty cycle (a wide default unless machine.toml declares one;
  flagged), the two log-sds combined;
- rated life L10: the lognormal with the Weibull(beta)'s median and 10% point; MTBF: the exponential's.
For the normalised families (f(nominal) = 1), k = D_limit / T, so k is lognormal with the same log-sd. D is a
fraction of D_limit (D_limit = 1, a definitional default) until a documented capacity gives it a unit
(checkpoint 3). A component with no usable statement gets no prior here: it is flagged for life_judgement.
"""

import math
from dataclasses import dataclass, field
from statistics import NormalDist

from kaful.artifacts.distributions import Delta, Lognormal
from kaful.artifacts.param_set import LifeStatement, Parameter
from kaful.config import DutyDefault, ParameterizeSettings
from kaful.units import UnitError, parse_unit

Z = NormalDist().inv_cdf
FLAG_NO_LIFE = "no documented life: prior from life_judgement (last resort, wide)"
FLAG_RATINGS = "load ratings needed: declare bearing designations/ratings via [[spec]]"
FLAG_CALENDAR = "calendar interval converted through a default duty cycle (no duty cycle declared)"
DAMAGE_FRACTION = "1"
DAMAGE_LIMIT_REF = "templates/protocol: D as a fraction of D_limit (no documented capacity yet)"


@dataclass
class LifePrior:
    component: str
    statement: LifeStatement | None
    category: str  # the interval_priors key, "rated_life", "mtbf", or "" (no statement)
    prior: Lognormal | None
    flags: list[str] = field(default_factory=list)


def category(s: LifeStatement) -> str:
    if s.kind in ("rated_life", "mtbf"):
        return s.kind
    if s.basis == "machine_wide":
        return "machine_wide"
    if s.kind == "replacement_interval":
        return "consumable_replacement" if s.consumable else "replacement"
    return "service"


def _years(s: LifeStatement, p: ParameterizeSettings) -> float:
    unit = s.unit.lower()
    days = p.calendar_days.get(unit, p.calendar_days.get(unit.rstrip("s"), 0.0))
    return s.value * days / p.calendar_days["year"]


def duty(p: ParameterizeSettings, time_unit: str) -> DutyDefault:
    """The default duty cycle for the time unit, by name or by unit ('h' finds 'hour')."""
    if time_unit in p.duty_defaults:
        return p.duty_defaults[time_unit]
    for name, d in sorted(p.duty_defaults.items()):
        try:
            if parse_unit(name) == parse_unit(time_unit):
                return d
        except UnitError:
            continue
    raise KeyError(f"no default duty cycle for time unit {time_unit!r}")


def comparable(s: LifeStatement, p: ParameterizeSettings, time_unit: str) -> float:
    """The statement in time units; a calendar one at the default duty cycle's median."""
    return s.in_time_unit if s.in_time_unit is not None else _years(s, p) * duty(p, time_unit).per_year_median


def _point_prior(s: LifeStatement, value: float, p: ParameterizeSettings) -> Lognormal:
    """Lognormal with the median and 10% point of the rated-life Weibull, or of the MTBF exponential."""
    if s.kind == "mtbf":
        median, q10 = value * math.log(2), value * -math.log(0.9)
    else:  # value is L10: Weibull scale eta with eta * (-ln 0.9) ** (1 / beta) = L10
        eta = value / (-math.log(0.9)) ** (1 / p.rated_life_weibull_beta)
        median, q10 = eta * math.log(2) ** (1 / p.rated_life_weibull_beta), value
    return Lognormal(median=median, sigma_log=math.log(median / q10) / -Z(0.1))


def life_prior(component: str, statements: list[LifeStatement], p: ParameterizeSettings, time_unit: str) -> LifePrior:
    usable = [s for s in statements if s.usable and s.bound != "none"]
    if not usable:
        return LifePrior(component, None, "", None, [FLAG_NO_LIFE])
    points = [s for s in usable if s.bound == "point"]
    s = max(points or usable, key=lambda x: (comparable(x, p, time_unit), x.id))
    cat = category(s)
    if s.bound == "point":
        return LifePrior(component, s, cat, _point_prior(s, comparable(s, p, time_unit), p))
    rule = p.interval_priors[cat]
    if s.in_time_unit is not None:
        return LifePrior(component, s, cat, Lognormal(median=s.in_time_unit * math.exp(-Z(rule.q) * rule.sigma),
                                                      sigma_log=rule.sigma))
    cycle = duty(p, time_unit)
    median = _years(s, p) * cycle.per_year_median * math.exp(-Z(rule.q) * rule.sigma)
    return LifePrior(component, s, cat, Lognormal(median=median, sigma_log=math.hypot(rule.sigma, cycle.sigma)),
                     [FLAG_CALENDAR])


def parameters(lp: LifePrior, time_unit: str) -> list[Parameter]:
    """life_nominal, k and damage_limit for a component with a life prior."""
    if lp.prior is None or lp.statement is None:
        return []
    s, t = lp.statement, lp.prior
    common = {"component": lp.component, "source": "calibrated", "ref": s.id, "confidence": s.confidence}
    return [
        Parameter(name="damage_limit", component=lp.component, role="threshold", unit=DAMAGE_FRACTION,
                  prior=Delta(value=1.0), source="default", ref=DAMAGE_LIMIT_REF, confidence=1.0),
        Parameter(name="k", role="rate", unit=f"{DAMAGE_FRACTION} / {time_unit}",
                  prior=Lognormal(median=1.0 / t.median, sigma_log=t.sigma_log), **common),
        Parameter(name="life_nominal", role="life", unit=time_unit, prior=t, **common),
    ]
