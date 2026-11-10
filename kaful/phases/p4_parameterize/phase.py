"""The parameterize phase: Scope + MachineModel + SensorContract -> ParamSet (M9). In progress: checkpoint 1.

Steps, in the order `choose_families` (and later `build_param_set`) runs them:
  a_choose_templates   one template family per full-model component (templates/families.py), from its fault
                       modes' mechanism words; the mode that decided it, the others not separately modelled
Later steps (checkpoint 2+): life statements -> priors on nominal life and k; value roles -> thresholds,
capacities and nominal inputs; life_judgement as a last resort; the gates.
"""

from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.scope import Scope
from kaful.config import KafulSettings
from kaful.phases.p4_parameterize.a_choose_templates import Choice, choose


def choose_families(model: MachineModel, scope: Scope, settings: KafulSettings) -> list[Choice]:
    return choose(model, scope, settings.parameterize)
