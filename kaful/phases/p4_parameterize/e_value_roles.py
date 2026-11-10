"""Step e: thresholds, nominal inputs and the template f parameters, from value roles, standards and [[spec]].

- Observation limits: a channel that observes a full-model component (scope O: fault-mode link or named) and
  has a documented range becomes `<channel>_limit` (role threshold, source manual with the range's quote):
  a `spec` range's upper bound as a point, a `setting_range` as a uniform over the range (the limit is set
  within it). A `default` range is a nominal operating input; an `example` never sets anything (M7 value
  roles; step f fails a parameter whose evidence is an example). `variant` waits for a declared configuration.
- f parameters per family:
  - arrhenius: activation_energy (literature, wide); temperature_nominal only from [[spec]] (no temperature
    channel range is documented yet): without it the template runs at f = 1, flagged;
  - bearing_fatigue: life_exponent (ISO 281: 3 ball, 10/3 roller; truncated normal), load_nominal and
    speed_nominal from [[spec]] (the load-ratings flag stays until declared);
  - accumulation: load_exponent only when a driving channel exists (none documented yet).
- recovery: a component with a documented or declared clean or descale action (kaful.lifetime) gets
  `recovery_<action>`, the fraction of D the action removes: uniform 0.9-1.0, a flagged default no document
  overrides yet, which data can estimate (M10).
- machine.toml [[spec]] for a component and a template slot sets that parameter (source deployment).
- machine.toml [[param]] (the onboarding review's corrections) then replaces any parameter it names (apply_declared).
"""

import math
from collections.abc import Sequence

from kaful.artifacts.distributions import Delta, Lognormal, TruncatedNormal, Uniform
from kaful.artifacts.param_set import Parameter
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.config import DeclaredParam, MachineConfig, ParameterizeSettings
from kaful.doctext import words
from kaful.lifetime import Action
from kaful.phases.p4_parameterize.a_choose_templates import Choice
from kaful.templates.families import FAMILIES

FLAG_RECOVERY = "maintenance recovery assumed near full (default uniform 0.9-1.0; data can estimate it)"
RECOVERY_REF = "default: a clean or descale removes 90-100% of D (no document states the recovery)"
FLAG_NO_TEMPERATURE = "no nominal temperature documented or declared: Arrhenius runs at f = 1 ([[spec]] temperature_nominal)"


def _limits(component: str, scope: Scope, contract: SensorContract) -> list[Parameter]:
    s = next(d for d in scope.decisions if d.component == component).signals
    names = sorted({c.channel for c in s.observed_by} | {c.channel for c in s.named_by})
    out = []
    for ch in (contract.channel(n) for n in names):
        ev = ch.range_evidence
        if ch.range is None or ev is None or ev.value_role not in ("spec", "setting_range"):
            continue
        low, high = ch.range
        prior = Delta(value=high) if ev.value_role == "spec" else Uniform(low=low, high=high)
        out.append(Parameter(name=f"{ch.name}_limit", component=component, role="threshold", unit=ch.unit or "1",
                             prior=prior, source="manual", ref=ch.ref, confidence=ch.confidence, evidence=(ev,)))
    return out


def _declared(component: str, family: str, machine: MachineConfig, p: ParameterizeSettings) -> list[Parameter]:
    slots = {s.name for s in FAMILIES[family].params} - {"k"}
    return [Parameter(name=d.name, component=component, role="nominal_input" if d.name.endswith("_nominal")
                      else "exponent", unit=d.unit, prior=Lognormal(median=d.value, sigma_log=p.declared_spec_sigma),
                      source="deployment", ref=f"machine.toml [[spec]] {component}.{d.name}", confidence=1.0)
            for d in sorted(machine.spec, key=lambda d: d.name) if d.component == component and d.name in slots]


def _family(c: Choice, component_name: str, declared: set[str], p: ParameterizeSettings) -> tuple[list[Parameter], list[str]]:
    params, flags = [], []
    if c.family == "arrhenius":
        a = p.activation_energy
        params.append(Parameter(name="activation_energy", component=c.component, role="exponent", unit=a.unit,
                                prior=Lognormal(median=a.median, sigma_log=a.sigma), source=a.source, ref=a.ref,
                                confidence=math.exp(-a.sigma)))
        if "temperature_nominal" not in declared:
            flags.append(FLAG_NO_TEMPERATURE)
    if c.family == "bearing_fatigue" and "life_exponent" not in declared:
        kind = "roller" if "roller" in words(component_name) else "ball"
        mean, sd = p.bearing_life_exponents[kind], p.bearing_exponent_sd
        params.append(Parameter(name="life_exponent", component=c.component, role="exponent", unit="1",
                                prior=TruncatedNormal(mean=mean, sd=sd, low=mean - 4 * sd, high=mean + 4 * sd),
                                source="standard", ref=f"ISO 281:2007 basic rating life, p for {kind} bearings",
                                confidence=1.0))
    return params, flags


def _recovery(component: str, actions: Sequence[Action], p: ParameterizeSettings) -> list[Parameter]:
    low, high = p.recovery_default
    return [Parameter(name=f"recovery_{a.action}", component=component, role="recovery", unit="1",
                      prior=Uniform(low=low, high=high), source="default", ref=RECOVERY_REF, confidence=1.0)
            for a in actions if a.component == component and a.action in ("clean", "descale")]


def value_roles(choices: Sequence[Choice], names: dict[str, str], scope: Scope, contract: SensorContract,
                machine: MachineConfig, p: ParameterizeSettings,
                actions: Sequence[Action] = ()) -> tuple[list[Parameter], dict[str, list[str]]]:
    params: list[Parameter] = []
    flags: dict[str, list[str]] = {}
    for c in choices:
        recovery = _recovery(c.component, actions, p)
        params += recovery
        if recovery:
            flags.setdefault(c.component, []).append(FLAG_RECOVERY)
        declared = _declared(c.component, c.family, machine, p)
        own, more = _family(c, names[c.component], {d.name for d in declared}, p)
        params += [*_limits(c.component, scope, contract), *declared, *own]
        if more:
            flags.setdefault(c.component, []).extend(more)
    return params, flags


FLAG_DECLARED = "set in machine.toml [[param]] (onboarding review)"
Z95 = 1.6448536269514722


def _declared_prior(d: DeclaredParam, p: ParameterizeSettings) -> Lognormal:
    if d.value is not None:
        return Lognormal(median=d.value, sigma_log=p.declared_spec_sigma)
    assert d.low is not None and d.high is not None
    return Lognormal(median=math.sqrt(d.low * d.high), sigma_log=math.log(d.high / d.low) / (2 * Z95))


def apply_declared(params: list[Parameter], machine: MachineConfig,
                   p: ParameterizeSettings) -> tuple[list[Parameter], dict[str, list[str]], list[str]]:
    """machine.toml [[param]] replaces the parameter it names (source deployment). A declared life_nominal also
    replaces a k derived from it (k = D_limit / T for a fraction D). Returns (params, flags, gate failures)."""
    by_id = {x.param_id: x for x in params}
    flags: dict[str, list[str]] = {}
    failures: list[str] = []
    for d in sorted(machine.param, key=lambda d: d.param):
        old = by_id.get(d.param)
        if old is None:
            failures.append(f"machine.toml [[param]] {d.param}: no such parameter in this ParamSet")
            continue
        if d.unit != old.unit:
            failures.append(f"machine.toml [[param]] {d.param}: unit {d.unit!r}, the parameter's is {old.unit!r}")
            continue
        prior = _declared_prior(d, p)
        ref = f"machine.toml [[param]] {d.param}: {d.reason}"
        by_id[d.param] = old.model_copy(update={"prior": prior, "source": "deployment", "ref": ref,
                                               "confidence": 1.0, "evidence": (), "downgraded_from": None})
        flags.setdefault(old.component, []).append(FLAG_DECLARED)
        limit, k = by_id.get(f"{old.component}.damage_limit"), by_id.get(f"{old.component}.k")
        if old.name == "life_nominal" and k is not None and limit is not None and isinstance(limit.prior, Delta):
            by_id[k.param_id] = k.model_copy(update={
                "prior": Lognormal(median=limit.prior.value / prior.median, sigma_log=prior.sigma_log),
                "source": "deployment", "ref": ref, "confidence": 1.0})
    return sorted(by_id.values(), key=lambda x: x.param_id), flags, failures
