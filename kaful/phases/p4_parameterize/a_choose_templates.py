"""Step a: one template family per full-model component, by code from its fault modes' mechanism words.

The families are tried strongest first (settings.parameterize.families); a component gets the first family
that any of its counted fault modes' mechanism or name words start with, and the mode that decided it is
recorded. Its other modes are listed as not separately modelled (one template per component, approved
2026-11-09; per-mode competing risks are logged for later). A component no word matches gets the fallback
family, flagged. Only Scope.full_model_components are parameterized.
"""

import re
from dataclasses import dataclass

from kaful.artifacts.machine_model import FaultMode, MachineModel
from kaful.artifacts.scope import Scope
from kaful.config import ParameterizeSettings
from kaful.phases.p3_scope.a_collect_signals import counts

WORD_RE = re.compile(r"[a-z]+")
FLAG_FALLBACK = "no mechanism word matched a template family; the fallback family is used"


@dataclass(frozen=True)
class Choice:
    component: str
    family: str
    decided_by: str | None  # the fault mode whose words chose the family; None for the fallback
    word: str | None  # the family word that matched
    other_modes: tuple[str, ...]  # counted modes not separately modelled
    flags: tuple[str, ...] = ()


def _words(mode: FaultMode) -> list[str]:
    return WORD_RE.findall(f"{mode.mechanism} {mode.name}".lower())


def _match(modes: list[FaultMode], settings: ParameterizeSettings) -> tuple[str, FaultMode, str] | None:
    for entry in settings.families:
        for mode in modes:
            hit = next((w for w in _words(mode) if w.startswith(entry.words)), None)
            if hit is not None:
                return entry.family, mode, hit
    return None


def choose(model: MachineModel, scope: Scope, settings: ParameterizeSettings) -> list[Choice]:
    choices = []
    for component in scope.full_model_components(model):
        modes = sorted((m for m in model.fault_modes if m.component == component.id and counts(m)),
                       key=lambda m: m.id)
        hit = _match(modes, settings)
        if hit is None:
            choices.append(Choice(component.id, settings.fallback_family, None, None,
                                  tuple(m.id for m in modes), (FLAG_FALLBACK,)))
            continue
        family, mode, word = hit
        choices.append(Choice(component.id, family, mode.id, word, tuple(m.id for m in modes if m.id != mode.id)))
    return choices
