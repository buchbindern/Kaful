"""Parameterize (M9), checkpoint 1: one template family per full-model component, chosen by code."""

from kaful.config import KafulSettings
from kaful.phases.p4_parameterize.a_choose_templates import FLAG_FALLBACK
from kaful.phases.p4_parameterize.phase import choose_families
from tests.unit.test_kaful_scope import _model, _scope

SETTINGS = KafulSettings()


def test_the_strongest_family_any_mode_names_wins_and_only_full_models_get_one() -> None:
    model = _model()
    choices = {c.component: (c.family, c.decided_by) for c in choose_families(model, _scope()[0], SETTINGS)}
    assert choices == {
        "bearings": ("bearing_fatigue", "bearings_flaking"),  # "flaking" beats the mechanism word "wear"
        "filter_element_fe": ("accumulation", "filter_clogging"),
        "shaft_seal": ("accumulation", "seal_wear"),
    }  # motor, pump and valve are not full models


def test_a_component_no_word_matches_gets_the_fallback_flagged() -> None:
    model = _model()
    modes = tuple(m.model_copy(update={"mechanism": "unknown", "name": "odd"}) if m.id == "seal_wear" else m
                  for m in model.fault_modes)
    [seal] = [c for c in choose_families(model.model_copy(update={"fault_modes": modes}), _scope()[0], SETTINGS)
              if c.component == "shaft_seal"]
    assert (seal.family, seal.decided_by, seal.flags) == ("accumulation", None, (FLAG_FALLBACK,))
