"""Step d: the parameterize gates (hard rule 2). Each failure is a line in the phase record, and the run stops.

- every life statement of a full-model component converts to the time unit, or is a calendar interval;
- under a component's life prior, P(T < I) <= max_probability_below_interval for each of its usable lower
  bounds I (a point life below a documented interval, or two intervals that disagree, fail here);
- a component's median time to D_limit (its final life prior, after [[param]]) is not shorter than any of its
  documented cleaning intervals (M10, Natalie 2026-11-10: residue must not reach the limit between cleans);
- every parameter has a source (the model enforces it) and none rests on an `example` value (rule, M7).
"""

from collections.abc import Sequence

from kaful.artifacts.param_set import LifeStatement, Parameter
from kaful.config import ParameterizeSettings
from kaful.lifetime import comparable
from kaful.phases.p4_parameterize.c_life_priors import LifePrior


def check(statements: Sequence[LifeStatement], priors: Sequence[LifePrior], params: Sequence[Parameter],
          p: ParameterizeSettings, time_unit: str) -> tuple[str, ...]:
    failures = [f"life statement {s.id}: {s.value:g} {s.unit} does not convert to {time_unit}" for s in statements
                if s.in_time_unit is None and s.unit.lower().rstrip("s") not in p.calendar_days
                and s.unit.lower() not in p.calendar_days]
    by_component = {lp.component: lp for lp in priors}
    for s in (s for s in statements if s.usable and s.bound == "lower"):
        lp = by_component.get(s.component)
        if lp is None or lp.prior is None:
            continue
        below = lp.prior.cdf(comparable(s, p, time_unit))
        if below > p.max_probability_below_interval:
            failures.append(f"component {s.component}: P(T < {s.id}) = {below:.2f} under its life prior "
                            f"(from {lp.statement.id if lp.statement else '?'}), above "
                            f"{p.max_probability_below_interval}")
    lives = {x.component: x for x in params if x.name == "life_nominal"}
    for s in (s for s in statements if s.usable and s.kind == "cleaning_interval" and s.component in lives):
        median, interval = lives[s.component].prior.quantile(0.5), comparable(s, p, time_unit)
        if median < interval:
            failures.append(f"component {s.component}: median time to D_limit {median:,.4g} is shorter than its "
                            f"documented cleaning interval {s.id} ({interval:,.4g} {time_unit})")
    failures += [f"parameter {x.param_id}: rests on an example value" for x in params
                 if any(e.value_role == "example" for e in x.evidence)]
    return tuple(failures)
