"""Step b: what feeds each template condition (load, speed, temperature), exact matching only (M10).

- A channel that drives one of the component's counted fault modes (scope D: an operating_input in
  `driven_by`) feeds the condition when its unit has the condition's dimension (accumulation's load takes any
  dimension). The nominal parameter `<condition>_nominal`, if the ParamSet has one, goes with it.
- Else a nominal parameter alone holds the condition at nominal; else the condition is undocumented and the
  template runs with f = 1 for it (flagged).
Component-to-component connections need a quantity one full model produces and another consumes; the
templates produce only damage and health, so there are none until observation models (M13). Every input is
sorted by name (rule 14); fuzzy matching is deferred (approved 2026-11-10).
"""

import pint

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import ExternalInput, TemplateBinding
from kaful.templates.families import FAMILIES
from kaful.templates.protocol import ANY_DIMENSION
from kaful.units import UnitError, parse_unit, registry

FLAG_UNDOCUMENTED = "condition {condition} is not documented or declared: the template holds it at nominal (f = 1)"


def _fits(unit: str | None, dimension: str) -> bool:
    """Whether a channel unit has the template slot's dimension (ANY_DIMENSION: any unit)."""
    if dimension == ANY_DIMENSION:
        return True
    try:
        return unit is not None and parse_unit(unit).dimensionality == registry().get_dimensionality(dimension)
    except (UnitError, pint.errors.PintError):
        return False


def wire(binding: TemplateBinding, ps: ParamSet, scope: Scope,
         contract: SensorContract) -> tuple[list[ExternalInput], dict[str, str], list[str]]:
    """(inputs, binding.conditions, flags) for one binding."""
    decision = next(d for d in scope.decisions if d.component == binding.component)
    driving = sorted(c.channel for c in decision.signals.driven_by)
    have = {x.name: x for x in ps.parameters if x.component == binding.component}
    inputs, conditions, flags = [], {}, []
    for slot in sorted(FAMILIES[binding.template].conditions, key=lambda s: s.name):
        name = f"{binding.component}_{slot.name}"
        nominal = have.get(f"{slot.name}_nominal")
        channel = next((c for c in driving if _fits(contract.channel(c).unit, slot.dimension)), None)
        if channel is not None:
            inputs.append(ExternalInput(name=name, binding=binding.id, condition=slot.name, channel=channel,
                                        unit=contract.channel(channel).unit or "", provenance="channel",
                                        nominal_param=nominal.param_id if nominal else None))
        elif nominal is not None:
            inputs.append(ExternalInput(name=name, binding=binding.id, condition=slot.name, unit=nominal.unit,
                                        nominal_param=nominal.param_id, provenance="nominal_param"))
        else:
            inputs.append(ExternalInput(name=name, binding=binding.id, condition=slot.name,
                                        provenance="undocumented", note="f = 1 for this condition"))
            flags.append(FLAG_UNDOCUMENTED.format(condition=slot.name))
        conditions[slot.name] = name
    return inputs, conditions, flags
