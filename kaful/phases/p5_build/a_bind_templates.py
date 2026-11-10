"""Step a: one binding per full-model component, from the ParamSet's template family (M9) and parameters.

A binding maps each template slot to the component's parameter of that name. k and D_limit (`damage_limit`,
role threshold or capacity) are required; D's unit is D_limit's. Another slot may be missing only when no
input moves the condition that needs it (step b: undocumented, f = 1). Escape-hatch code is never produced:
every full model maps to a family; a family the library lacks is a BuildError for that component.
"""

from kaful.artifacts.param_set import ComponentTemplate, ParamSet
from kaful.artifacts.twin_definition import TemplateBinding
from kaful.templates.families import FAMILIES

REQUIRED = ("k", "damage_limit")


def bind(t: ComponentTemplate, ps: ParamSet) -> TemplateBinding:
    own = {x.name: x for x in ps.parameters if x.component == t.component}
    template = FAMILIES[t.family]
    slots = {s.name for s in template.params} | {"damage_limit"}
    limit = own.get("damage_limit")
    return TemplateBinding(
        id=t.component, component=t.component, fault_mode=t.decided_by, other_modes=t.other_modes,
        template=t.family, template_version=template.version, damage_state=f"{t.component}_damage",
        damage_unit=limit.unit if limit else "1", threshold_param=f"{t.component}.damage_limit",
        params={name: own[name].param_id for name in sorted(slots) if name in own and name != "damage_limit"},
        justification=(f"family {t.family} from fault mode {t.decided_by}" if t.decided_by
                       else f"family {t.family} (fallback)"))


def missing(t: ComponentTemplate, ps: ParamSet) -> list[str]:
    have = {x.name for x in ps.parameters if x.component == t.component}
    return [name for name in REQUIRED if name not in have]
