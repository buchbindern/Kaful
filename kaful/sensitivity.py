"""Sensitivity of RUL to the priors (M9's computation, shared by the parameterize report and the evaluate phase).

For one full-model component, RUL at t = 0 under a constant operating profile is D_limit / (k * f(profile)), drawn by
Monte Carlo from the priors (seeded):
- share: the part of var(log RUL) a prior explains (1 - variance with it frozen at its median / variance);
- tornado: median RUL with the prior at its 16th and 84th percentile (about -1 and +1 sd), others at their median.
A profile the component cannot respond to (no load exponent, no nominal temperature documented or declared) is not
evaluable, never silently nominal.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from kaful.artifacts.base import StrictModel
from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import ComponentTemplate, Parameter, ParamSet
from kaful.templates.families import FAMILIES

NOT_RATE = ("life_nominal",)  # reported, but k carries it into RUL


class Effect(StrictModel):
    param: str
    source: str
    share: float = Field(description="Share of var(log RUL) under the nominal profile.")
    low_median: float = Field(description="Median RUL with the prior at its 16th percentile.")
    high_median: float = Field(description="Median RUL with the prior at its 84th percentile.")


def _draw(x: Parameter, rng: np.random.Generator, n: int, freeze: str | None) -> np.ndarray:
    if x.name == freeze or isinstance(x.prior, Delta):
        return np.full(n, x.prior.quantile(0.5), dtype=np.float64)
    return x.prior.sample(rng, n)


def rul(template: ComponentTemplate, own: Mapping[str, Parameter], profile: Mapping[str, float], n: int,
        seed: int, freeze: str | None = None, at: tuple[str, float] | None = None) -> tuple[np.ndarray | None, str]:
    """RUL samples at t = 0 under `profile`, or (None, why it is not evaluable)."""
    family = FAMILIES[template.family]()
    rng = np.random.default_rng(seed)
    params = {name: _draw(x, rng, n, freeze) for name, x in sorted(own.items()) if name not in NOT_RATE}
    if at is not None:
        params[at[0]] = np.full(n, own[at[0]].prior.quantile(at[1]), dtype=np.float64)
    conditions: dict[str, np.ndarray] = {}
    for key, factor in sorted(profile.items()):
        if key in ("load_factor", "speed_factor"):
            cond = key.removesuffix("_factor")
            if cond not in {s.name for s in family.conditions}:
                return None, f"{template.family} does not read {cond}"
            if template.family == "accumulation" and "load_exponent" not in params:
                return None, "no load exponent: no driving load is documented"
            params.setdefault(f"{cond}_nominal", np.ones(n))
            conditions[cond] = params[f"{cond}_nominal"] * factor
        elif key == "temperature_delta":
            if template.family != "arrhenius":
                return None, f"{template.family} does not read temperature"
            if "temperature_nominal" not in params:
                return None, "no nominal temperature documented or declared"
            conditions["temperature"] = params["temperature_nominal"] + factor
    for slot in family.params:  # an input no profile moves leaves f at 1 whatever its value: fill with 1
        if slot.name not in params:
            params[slot.name] = np.ones(n)
    rate = family.rate(np.zeros(n), conditions, params)
    return params["damage_limit"] / rate, ""


def own_parameters(template: ComponentTemplate, ps: ParamSet) -> dict[str, Parameter]:
    return {x.name: x for x in ps.parameters if x.component == template.component}


@dataclass(frozen=True)
class ComponentSensitivity:
    var_log_rul: float
    effects: tuple[Effect, ...]  # most variance first


def component_sensitivity(template: ComponentTemplate, own: Mapping[str, Parameter], n: int,
                          seed: int) -> ComponentSensitivity:
    """Every uncertain prior's share of var(log RUL) and its tornado medians, under the nominal profile."""
    base, _ = rul(template, own, {}, n, seed)
    assert base is not None
    var = float(np.var(np.log(base)))
    effects = []
    for name, x in sorted(own.items()):
        if name in NOT_RATE or isinstance(x.prior, Delta):
            continue
        frozen, _ = rul(template, own, {}, n, seed, freeze=name)
        lo, _ = rul(template, own, {}, n, seed, freeze=None, at=(name, 0.16))
        hi, _ = rul(template, own, {}, n, seed, freeze=None, at=(name, 0.84))
        assert frozen is not None and lo is not None and hi is not None
        share = 0.0 if var == 0 else min(1.0, max(0.0, 1.0 - float(np.var(np.log(frozen))) / var))
        effects.append(Effect(param=f"{template.component}.{name}", source=x.source, share=round(share, 4),
                              low_median=float(np.median(lo)), high_median=float(np.median(hi))))
    return ComponentSensitivity(var_log_rul=var, effects=tuple(sorted(effects, key=lambda e: (-e.share, e.param))))
