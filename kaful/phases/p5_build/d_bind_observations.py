"""Step d: observation models from machine.toml [[observation]] (M12): one ObservationBinding per reviewed
(component, channel) whose component has a binding.

Every observation model is reviewed: nothing binds from a scope link alone. The binding records the link that
motivated it (`basis`): a fault mode's observable_by (scope O observed_by or threshold_events, with the fault mode as
ref), scope's named_by (M8), or neither (`declared`, e.g. a retrofitted sensor). Parameters are the ones step f of
parameterize made: `machine.<channel>_y_healthy`, `<component>.<channel>_y_limit`, `machine.<channel>_noise_sd`;
step e checks they exist, are in the channel's unit and suit the form. An observation of a component without a full
model is listed in `skipped`, not bound.
"""

from dataclasses import dataclass

from kaful.artifacts.scope import Scope
from kaful.artifacts.twin_definition import ObservationBasis, ObservationBinding, TemplateBinding
from kaful.config import DeclaredObservation, MachineConfig
from kaful.observation.forms import FORMS

DECLARED_REF = "machine.toml [[observation]]"


@dataclass(frozen=True)
class Bound:
    observations: tuple[ObservationBinding, ...]
    skipped: tuple[str, ...]  # "<component> -> <channel>" with no binding


def _basis(d: DeclaredObservation, scope: Scope) -> tuple[ObservationBasis, str]:
    decision = next((x for x in scope.decisions if x.component == d.component), None)
    if decision is not None:
        s = decision.signals
        for link in (*s.observed_by, *s.threshold_events):
            if link.channel == d.channel and link.fault_modes:
                return "observable_by", link.fault_modes[0]
        if any(n.channel == d.channel for n in s.named_by):
            return "named_by", f"scope named_by {d.channel}"
    return "declared", f"{DECLARED_REF} {d.component} -> {d.channel}"


def _params(d: DeclaredObservation) -> dict[str, str]:
    if d.form != "linear":
        return {}
    return {"y_healthy": f"machine.{d.channel}_y_healthy", "y_limit": f"{d.component}.{d.channel}_y_limit",
            "noise_sd": f"machine.{d.channel}_noise_sd"}


def bind_observations(machine: MachineConfig, bindings: list[TemplateBinding], scope: Scope) -> Bound:
    by_component = {b.component: b for b in bindings}
    out, skipped = [], []
    for d in sorted(machine.observation, key=lambda d: (d.component, d.channel)):
        binding = by_component.get(d.component)
        if binding is None:
            skipped.append(f"{d.component} -> {d.channel}")
            continue
        basis, ref = _basis(d, scope)
        out.append(ObservationBinding(
            id=f"{d.component}_on_{d.channel}", binding=binding.id, channel=d.channel, form=d.form,
            form_version=FORMS[d.form].version, params=_params(d), event_value=d.event_value, basis=basis, ref=ref,
            justification=d.reason))
    return Bound(tuple(sorted(out, key=lambda o: o.id)), tuple(skipped))
