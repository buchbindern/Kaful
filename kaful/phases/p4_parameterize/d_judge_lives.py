"""Step d: a last-resort life prior from parameterize.life_judgement, for a full-model component with no usable
documented life (approved 2026-11-09; the bearings until load ratings are declared).

Each component's deciding fault mode goes to the prompt `judgement_samples` times. Each sample's 5th-95th
percentile range [low, high] becomes a lognormal (median sqrt(low * high), log-sd ln(high / low) / 3.29); the
samples are pooled (geometric mean of medians; log-sd from the mean sample variance plus the spread of the log
medians) and the log-sd is floored at `judgement_sigma_floor`. Confidence is computed by code: base times
exp(-sd of the log medians). Source llm_judgement, flagged.

A life stated in an amount processed (a water filter's capacity in litres) is converted to the time unit
through the amount per time unit (water per product): machine.toml [[spec]] `<quantity>_per_<time unit>`
(component 'machine') first, else the flagged wide default in settings. Then D is that amount: D_limit is the
judged capacity, k = D_limit / T is the amount per time unit, and T = capacity / throughput (lognormal: medians
divide, log-sds add in quadrature).
"""

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pint

from kaful.artifacts.distributions import Delta, Lognormal
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.param_set import Parameter
from kaful.artifacts.base import SourceKind
from kaful.config import MachineConfig, ParameterizeSettings
from kaful.phases.p4_parameterize.a_choose_templates import Choice
from kaful.phases.p4_parameterize.c_life_priors import DAMAGE_FRACTION, DAMAGE_LIMIT_REF
from kaful.prompts.parameterize.life_judgement.models import LifeJudgement, LifeRange
from kaful.units import UnitError, parse_unit, registry

Z95 = 1.6448536269514722
PROMPT_ID = "parameterize.life_judgement"
FLAG_JUDGED = "life from life_judgement (last resort, wide; no document states it)"
FLAG_THROUGHPUT = "amount per time unit from a wide default (no [[spec]] declared, none extracted)"
FLAG_NO_UNIT = "life_judgement gave no usable unit; no life prior"
Judge = Callable[[list[dict[str, str]]], list[list[LifeJudgement]]]


@dataclass
class Judged:
    component: str
    life: Lognormal | None  # T, in the time unit
    quantity: str | None = None  # 'volume', 'mass' when the life was an amount processed
    capacity: Lognormal | None = None  # D_limit, in quantity_units[quantity]
    throughput: tuple[Lognormal, SourceKind, str] | None = None  # amount per time unit, its source and ref
    confidence: float = 0.0
    basis: str = ""
    flags: list[str] = field(default_factory=list)


def inputs_for(choices: Sequence[Choice], model: MachineModel, machine: MachineConfig) -> list[dict[str, str]]:
    comps, modes = {c.id: c for c in model.components}, {m.id: m for m in model.fault_modes}
    out = []
    for c in choices:
        comp, mode = comps[c.component], modes[c.decided_by] if c.decided_by else None
        out.append({"machine": machine.display_name, "time_unit": machine.confirmed_time_unit(),
                    "component": f"{comp.name}: {comp.description}",
                    "fault_mode": f"{mode.mechanism}: {mode.effect}" if mode else "unknown mechanism"})
    return out


def _quantity(unit: str, p: ParameterizeSettings) -> str | None:
    try:
        dim = parse_unit(unit).dimensionality
    except UnitError:
        return None
    return next((q for q, u in sorted(p.quantity_units.items()) if parse_unit(u).dimensionality == dim), None)


def _in(r: LifeRange, time_unit: str, p: ParameterizeSettings) -> tuple[str, float, float] | None:
    """(kind, low, high): kind 'time' in the time unit, or a quantity in its pooled unit; None if unusable."""
    unit = r.unit.strip().lower()
    if unit.rstrip("s") == time_unit.lower().rstrip("s"):
        return "time", r.low, r.high
    reg = registry()
    for kind, target in (("time", time_unit), *((q, u) for q, u in sorted(p.quantity_units.items()))):
        try:
            to = parse_unit(target)
            low, high = (float(reg.Quantity(v, parse_unit(unit)).to(to).magnitude) for v in (r.low, r.high))
            return kind, low, high
        except (UnitError, pint.errors.DimensionalityError):
            continue
    return None


def _pool(ranges: Sequence[tuple[float, float]], p: ParameterizeSettings) -> tuple[Lognormal, float]:
    logs = [0.5 * (math.log(lo) + math.log(hi)) for lo, hi in ranges]
    sigmas = [math.log(hi / lo) / (2 * Z95) for lo, hi in ranges]
    spread = statistics.pstdev(logs) if len(logs) > 1 else 0.0
    sigma = max(p.judgement_sigma_floor, math.sqrt(statistics.fmean(s * s for s in sigmas) + spread ** 2))
    return Lognormal(median=math.exp(statistics.fmean(logs)), sigma_log=sigma), \
        p.judgement_confidence_base * math.exp(-spread)


def throughput(quantity: str, machine: MachineConfig, p: ParameterizeSettings) -> tuple[Lognormal, SourceKind, str, bool] | None:
    """Amount per time unit: (prior in quantity_units[quantity] per time unit, source, ref, defaulted)."""
    name, unit = f"{quantity}_per_{machine.confirmed_time_unit()}", p.quantity_units[quantity]
    declared = next((d for d in machine.spec if d.component == "machine" and d.name == name), None)
    if declared is not None:
        value = float(registry().Quantity(declared.value, parse_unit(declared.unit)).to(parse_unit(unit)).magnitude)
        return Lognormal(median=value, sigma_log=p.declared_spec_sigma), "deployment", f"machine.toml [[spec]] {name}", False
    d = p.throughput_defaults.get(name)
    if d is None:
        return None
    value = float(registry().Quantity(d.median, parse_unit(d.unit)).to(parse_unit(unit)).magnitude)
    return Lognormal(median=value, sigma_log=d.sigma), "default", f"settings parameterize.throughput_defaults.{name}: {d.why}", True


def to_prior(component: str, outputs: Sequence[LifeJudgement], machine: MachineConfig,
             p: ParameterizeSettings) -> Judged:
    read = [x for x in (_in(o.life, machine.confirmed_time_unit(), p) for o in outputs) if x is not None]
    if not read:
        return Judged(component, None, flags=[FLAG_JUDGED, FLAG_NO_UNIT])
    kind = statistics.mode(sorted(k for k, _, _ in read))  # the majority kind; ties to the first in sort order
    pooled, confidence = _pool([(lo, hi) for k, lo, hi in read if k == kind], p)
    basis = " | ".join(sorted({o.basis for o in outputs}))
    if kind == "time":
        return Judged(component, pooled, confidence=confidence, basis=basis, flags=[FLAG_JUDGED])
    rate = throughput(kind, machine, p)
    if rate is None:
        return Judged(component, None, kind, pooled, confidence=confidence, basis=basis,
                      flags=[FLAG_JUDGED, f"no {kind} per {machine.confirmed_time_unit()} to convert its capacity"])
    q, source, ref, defaulted = rate
    life = Lognormal(median=pooled.median / q.median, sigma_log=math.hypot(pooled.sigma_log, q.sigma_log))
    return Judged(component, life, kind, pooled, (q, source, ref), confidence, basis,
                  [FLAG_JUDGED, *([FLAG_THROUGHPUT] if defaulted else [])])


def parameters(j: Judged, time_unit: str, p: ParameterizeSettings, prompt_ref: str) -> list[Parameter]:
    if j.life is None:
        return []
    judged = {"component": j.component, "source": "llm_judgement", "ref": prompt_ref, "confidence": j.confidence}
    life = Parameter(name="life_nominal", role="life", unit=time_unit, prior=j.life, **judged)
    if j.capacity is None or j.throughput is None or j.quantity is None:
        return [Parameter(name="damage_limit", component=j.component, role="threshold", unit=DAMAGE_FRACTION,
                          prior=Delta(value=1.0), source="default", ref=DAMAGE_LIMIT_REF, confidence=1.0),
                Parameter(name="k", role="rate", unit=f"{DAMAGE_FRACTION} / {time_unit}",
                          prior=Lognormal(median=1.0 / j.life.median, sigma_log=j.life.sigma_log), **judged),
                life]
    unit = p.quantity_units[j.quantity]
    q, source, ref = j.throughput
    return [Parameter(name="damage_limit", role="capacity", unit=unit, prior=j.capacity, **judged),
            Parameter(name="k", component=j.component, role="rate", unit=f"{unit} / {time_unit}", prior=q,
                      source=source, ref=ref, confidence=1.0 if source == "deployment" else j.confidence),
            life]
