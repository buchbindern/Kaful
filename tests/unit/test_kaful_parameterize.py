"""Parameterize (M9): one template family per full-model component, chosen by code; life statements read back
from their pages; interval and point priors on nominal life; k = D_limit / T; the gates."""

import math
from typing import Any

import pytest

from kaful.artifacts.distributions import Delta
from kaful.config import DeclaredLife, KafulSettings
from kaful.phases.p3_scope.phase import build_scope
from kaful.phases.p4_parameterize.c_life_priors import FLAG_CALENDAR, FLAG_NO_LIFE, FLAG_RATINGS
from kaful.phases.p4_parameterize.a_choose_templates import FLAG_FALLBACK
from kaful.phases.p4_parameterize.phase import choose_families
from tests.unit.test_kaful_scope import _contract, _model, _scope

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


RAW_SCHEDULE = "Maintenance schedule\nYearly 4000 Replace filter element (FE)\nHave the hose inspected"


def _index(raw_schedule: bool) -> Any:
    from tests.unit.test_kaful_scope import _index as scope_index

    index = scope_index()
    if not raw_schedule:
        return index  # the schedule is only in the page's LLM layer: its rows are in_layer
    pages = tuple(p.model_copy(update={"text": RAW_SCHEDULE}) if (p.doc_id, p.page) == ("manual", 2) else p
                  for p in index.pages)
    return index.model_copy(update={"pages": pages})


def _build(machine: Any = None, raw_schedule: bool = True) -> tuple[Any, Any, tuple[str, ...]]:
    from kaful.phases.p4_parameterize.phase import build_param_set
    from tests.harness.kaful_factories import BASE
    from tests.unit.test_kaful_scope import _machine

    machine, index = machine or _machine(), _index(raw_schedule)
    scope = build_scope({**BASE}, index, _model(), _contract(), machine, SETTINGS)[0]
    return build_param_set({**BASE}, index, _model(), scope, machine, SETTINGS)


def test_life_statements_are_read_back_from_their_rows_and_sentences() -> None:
    param_set, _, failures = _build()
    assert failures == ()
    got = {s.id: (s.kind, s.bound, s.value, s.unit, s.in_time_unit, s.consumable) for s in param_set.life_statements}
    assert got == {
        "filter_element_fe_machine_wide_1": ("inspection_interval", "none", 1.0, "yearly", None, False),  # by ditto
        "filter_element_fe_stated_1": ("replacement_interval", "lower", 4000.0, "hour", 4000.0, True),  # Running hours
        "shaft_seal_stated_1": ("replacement_interval", "lower", 2000.0, "hours", 2000.0, False),  # a sentence
    }
    assert all(s.source == "manual" and s.usable for s in param_set.life_statements)


def test_a_statement_not_verbatim_on_its_page_is_downgraded_and_not_used() -> None:
    """Hard rule 4: a schedule row found only in the page's LLM layer is in_layer, not manual."""
    param_set, priors, _ = _build(raw_schedule=False)
    row = next(s for s in param_set.life_statements if s.id == "filter_element_fe_stated_1")
    assert (row.source, row.downgraded_from, row.usable, row.evidence[0].verification) == (
        "llm_judgement", "manual", False, "in_layer")
    assert FLAG_NO_LIFE in next(t for t in param_set.templates if t.component == "filter_element_fe").flags


def test_interval_priors_follow_the_approved_q_and_sigma_and_k_is_d_limit_over_t() -> None:
    from statistics import NormalDist

    param_set, priors, _ = _build()
    p = SETTINGS.parameterize.interval_priors
    life = param_set.get("filter_element_fe.life_nominal").prior
    assert life.cdf(4000.0) == pytest.approx(p["consumable_replacement"].q)  # type: ignore[union-attr]
    assert life.sigma_log == p["consumable_replacement"].sigma  # type: ignore[union-attr]
    seal = param_set.get("shaft_seal.life_nominal").prior
    assert seal.median == pytest.approx(2000 * math.exp(-NormalDist().inv_cdf(p["replacement"].q) * p["replacement"].sigma))  # type: ignore[union-attr]
    k = param_set.get("shaft_seal.k")
    assert (k.source, k.ref, k.prior.median) == ("calibrated", "shaft_seal_stated_1", pytest.approx(1 / seal.median))  # type: ignore[union-attr]
    assert param_set.get("shaft_seal.damage_limit").prior == Delta(value=1.0)


def test_a_life_model_without_ratings_is_flagged_and_gets_no_prior_yet() -> None:
    param_set, _, _ = _build()
    bearings = next(t for t in param_set.templates if t.component == "bearings")
    assert FLAG_RATINGS in bearings.flags and FLAG_NO_LIFE in bearings.flags
    assert not [x for x in param_set.parameters if x.component == "bearings"]


def test_a_point_life_below_a_documented_interval_fails_the_gate() -> None:
    from tests.unit.test_kaful_scope import _machine

    machine = _machine(life=(DeclaredLife(component="shaft_seal", kind="rated_life", value=500, unit="h"),))
    param_set, _, failures = _build(machine)
    assert param_set.get("shaft_seal.life_nominal").ref == "shaft_seal_declared_1"  # a point life binds
    assert len(failures) == 1 and failures[0].startswith("component shaft_seal: P(T < shaft_seal_stated_1) = ")


def test_a_calendar_interval_goes_through_the_default_duty_cycle_flagged() -> None:
    from tests.unit.test_kaful_scope import _machine

    machine = _machine(life=(DeclaredLife(component="shaft_seal", kind="service_interval", value=3, unit="year"),))
    param_set, priors, failures = _build(machine)
    [lp] = [x for x in priors if x.component == "shaft_seal"]
    duty, rule = SETTINGS.parameterize.duty_defaults["hour"], SETTINGS.parameterize.interval_priors["service"]
    assert failures == () and lp.statement is not None and lp.statement.id == "shaft_seal_declared_1"  # 3 y > 2000 h
    assert FLAG_CALENDAR in lp.flags
    assert lp.prior is not None and lp.prior.sigma_log == pytest.approx(math.hypot(rule.sigma, duty.sigma))
    assert lp.prior.median == pytest.approx(3 * duty.per_year_median * math.exp(1.6448536 * rule.sigma), rel=1e-6)


@pytest.mark.parametrize("seed", ["0", "1"])
def test_the_param_set_does_not_depend_on_hash_order(seed: str) -> None:
    import os
    import subprocess
    import sys

    from kaful.runs import dump_json

    code = ("from tests.unit.test_kaful_parameterize import _build; from kaful.runs import dump_json; "
            "print(dump_json(_build()[0]))")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": seed})
    assert out.stdout.strip() == dump_json(_build()[0]).strip()
