"""Step g: coupling-strength priors from machine.toml [[connection]] (M13b).

For each declared connection whose from-component has a full model and feeds at least one full-model component,
each slot of its coupling form (kaful.templates.couplings) gets one parameter on the from-component, shared by all
its targets: `<from_component>.<slot>`, role `coupling`, source `default`, the uniform prior in
settings.parameterize.coupling_priors. No document states a coupling's strength (decided 2026-11-16: wide, flagged).
Confidence is computed by code (rule 15): the share of the slot's admissible range the prior excludes, 0 for a prior
over the whole range.

Flags: the from-component's strength is a default; each target's rate now depends on the from-component; a target
whose life is an llm_judgement was judged without the coupling in its prompt, so whether that life assumed the
from-component maintained or not is uncontrolled (it is read as normal service: the from-component renewed at its
limit, decided 2026-11-16).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from kaful.artifacts.distributions import Uniform
from kaful.artifacts.param_set import Parameter
from kaful.config import MachineConfig, ParameterizeSettings
from kaful.templates.couplings import COUPLINGS

FLAG_STRENGTH = ("coupling {form} to {targets}: {slot} has no documented value; a default prior over its whole "
                 "range (a modelling assumption)")
FLAG_TARGET = "scaling rate depends on {source} through {form}; its strength is a default prior"
FLAG_JUDGED_BASIS = ("life judged without the {source} coupling in the prompt: whether it assumed {source} maintained "
                     "is uncontrolled; read as normal service ({source} renewed at its limit)")


@dataclass
class CouplingPriors:
    params: list[Parameter] = field(default_factory=list)
    flags: dict[str, list[str]] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def flag(self, component: str, text: str) -> None:
        if text not in self.flags.setdefault(component, []):
            self.flags[component].append(text)


def coupling_priors(machine: MachineConfig, components: Sequence[str], judged: Sequence[str],
                    p: ParameterizeSettings) -> CouplingPriors:
    """`components`: full-model components; `judged`: those whose life is an llm_judgement."""
    out, modelled = CouplingPriors(), set(components)
    for d in sorted(machine.connection, key=lambda d: (d.from_component, d.form)):
        targets = [t for t in d.to_components if t in modelled]
        if d.from_component not in modelled or not targets:
            continue  # build lists connections without full models at both ends
        form, defaults = COUPLINGS[d.form], p.coupling_priors.get(d.form, {})
        for slot in sorted(s.name for s in form.params):
            prior = defaults.get(slot)
            lo, hi = form.bounds[slot]
            if prior is None:
                out.failures.append(f"machine.toml [[connection]] {d.from_component} {d.form}: no default prior for "
                                    f"{slot} (settings.parameterize.coupling_priors)")
                continue
            if prior.low < lo or prior.high > hi:
                out.failures.append(f"settings.parameterize.coupling_priors {d.form}.{slot}: [{prior.low:g}, "
                                    f"{prior.high:g}] is outside the form's range [{lo:g}, {hi:g}]")
                continue
            out.params.append(Parameter(
                name=slot, component=d.from_component, role="coupling", unit="1",
                prior=Uniform(low=prior.low, high=prior.high), source="default",
                confidence=1.0 - (prior.high - prior.low) / (hi - lo),
                ref=f"default: settings.parameterize.coupling_priors {d.form}.{slot}: {prior.why}"))
            out.flag(d.from_component, FLAG_STRENGTH.format(form=d.form, targets=", ".join(targets), slot=slot))
        for t in targets:
            out.flag(t, FLAG_TARGET.format(source=d.from_component, form=d.form))
            if t in judged:
                out.flag(t, FLAG_JUDGED_BASIS.format(source=d.from_component))
    out.params.sort(key=lambda x: x.param_id)
    return out
