"""Parameterize (M9): one template family per full-model component, chosen by code; life statements read back
from their pages; interval and point priors on nominal life; k = D_limit / T; the gates."""

import math
from typing import Any

import pytest

from kaful.artifacts.distributions import Delta
from kaful.config import DeclaredLife, DeclaredSpec, KafulSettings
from kaful.phases.p3_scope.phase import build_scope
from kaful.phases.p4_parameterize.c_life_priors import FLAG_CALENDAR, FLAG_NO_LIFE, FLAG_RATINGS, FLAG_TRANSCRIBED
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


def _built(machine: Any = None, raw_schedule: bool = True, judge: Any = None, contract: Any = None,
           index: Any = None) -> Any:
    from kaful.phases.p4_parameterize.phase import build_param_set
    from tests.harness.kaful_factories import BASE
    from tests.unit.test_kaful_scope import _machine

    machine, index, contract = machine or _machine(), index or _index(raw_schedule), contract or _contract()
    scope = build_scope({**BASE}, index, _model(), contract, machine, SETTINGS)[0]
    return build_param_set({**BASE}, index, _model(), scope, contract, machine, SETTINGS, judge)


def _build(machine: Any = None, raw_schedule: bool = True) -> tuple[Any, Any, tuple[str, ...]]:
    built = _built(machine, raw_schedule)
    return built.param_set, built.priors, built.failures


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


def test_a_statement_from_a_scanned_page_is_used_from_its_transcription_with_a_wider_prior() -> None:
    """Natalie 2026-11-09: a scanned page has no text layer, so its LLM transcription is all there is."""
    plain = _index(raw_schedule=False)
    index = plain.model_copy(update={"pages": tuple(p.model_copy(update={"page_class": "scanned", "text": ""})
                                                    if (p.doc_id, p.page) == ("manual", 2) else p for p in plain.pages)})
    built = _built(index=index)
    param_set, priors = built.param_set, built.priors
    row = next(s for s in param_set.life_statements if s.id == "filter_element_fe_stated_1")
    assert (row.source, row.usable, row.transcribed) == ("llm_judgement", True, True)
    [lp] = [x for x in priors if x.component == "filter_element_fe"]
    rule = SETTINGS.parameterize.interval_priors["consumable_replacement"]
    assert FLAG_TRANSCRIBED in lp.flags and lp.prior is not None
    assert lp.prior.sigma_log == pytest.approx(math.hypot(rule.sigma, SETTINGS.parameterize.transcription_extra_sigma))


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
    assert [x.name for x in param_set.parameters if x.component == "bearings"] == ["life_exponent"]  # no life


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


def _judgement(unit: str, low: float, high: float) -> Any:
    from kaful.prompts.parameterize.life_judgement.models import LifeJudgement

    return LifeJudgement.model_validate({"life": {"unit": unit, "low": low, "high": high}, "basis": "class typical"})


def test_life_judgement_gives_a_wide_flagged_prior_to_a_component_with_no_documented_life() -> None:
    """The bearings have only a life model: they go to life_judgement; samples are pooled and floored."""
    from kaful.phases.p4_parameterize.d_judge_lives import FLAG_JUDGED

    asked: list[dict[str, str]] = []

    def judge(inputs: list[dict[str, str]]) -> list[list[Any]]:
        asked.extend(inputs)
        return [[_judgement("hours", 10000, 40000), _judgement("h", 20000, 80000)] for _ in inputs]

    built = _built(judge=judge)
    assert [i["component"].split(":")[0] for i in asked] == ["Angular contact ball bearings"]
    life = built.param_set.get("bearings.life_nominal")
    assert (life.source, life.prior.median) == ("llm_judgement", pytest.approx(math.sqrt(10000 * 40000 * 20000 * 80000) ** 0.5))  # type: ignore[union-attr]
    assert life.prior.sigma_log == pytest.approx(SETTINGS.parameterize.judgement_sigma_floor)  # type: ignore[union-attr]
    assert life.confidence < SETTINGS.parameterize.judgement_confidence_base  # the samples disagree
    flags = next(t for t in built.param_set.templates if t.component == "bearings").flags
    assert FLAG_JUDGED in flags and FLAG_NO_LIFE not in flags and FLAG_RATINGS in flags
    assert built.param_set.get("bearings.life_exponent").source == "standard"  # ISO 281, ball: p = 3


def test_a_capacity_in_litres_becomes_a_life_in_products_through_water_per_product() -> None:
    """M9 checkpoint 3 (Natalie 2026-11-09): a water filter's capacity is a water amount, the time unit is
    products. T = capacity / water per product; D is litres, D_limit the capacity, k the water per product."""
    from kaful.phases.p4_parameterize.d_judge_lives import FLAG_THROUGHPUT, to_prior
    from tests.unit.test_kaful_scope import _machine

    p = SETTINGS.parameterize
    machine = _machine().model_copy(update={"time_unit": "product"})
    judged = to_prior("water_filter", [_judgement("L", 1000, 10000)], machine, p)
    default = p.throughput_defaults["volume_per_product"]
    assert judged.quantity == "volume" and judged.capacity is not None and judged.life is not None
    assert judged.life.median == pytest.approx(math.sqrt(1000 * 10000) / default.median)
    assert judged.life.sigma_log == pytest.approx(math.hypot(judged.capacity.sigma_log, default.sigma))
    assert FLAG_THROUGHPUT in judged.flags
    declared = machine.model_copy(update={"spec": (DeclaredSpec(component="machine", name="volume_per_product",
                                                                value=150, unit="ml"),)})
    exact = to_prior("water_filter", [_judgement("m3", 1, 10)], declared, p)  # any volume unit
    assert exact.life is not None and exact.life.median == pytest.approx(math.sqrt(1000 * 10000) / 0.150)
    assert exact.throughput is not None and exact.throughput[1:] == ("deployment", "machine.toml [[spec]] volume_per_product")
    assert FLAG_THROUGHPUT not in exact.flags


def test_a_documented_limit_sets_a_threshold_and_an_example_never_does() -> None:
    from kaful.artifacts.base import Evidence

    def leak(role: str) -> Any:
        ev = Evidence(doc_id="manual", page=1, quote="leak", verification="verbatim", value_role=role)
        return _contract().model_copy(update={"channels": tuple(c.model_copy(update={
            "range": (0.0, 5.0), "range_evidence": ev}) for c in _contract().channels)})

    spec = _built(contract=leak("spec")).param_set.get("shaft_seal.leak_flow_limit")
    assert (spec.role, spec.prior, spec.source) == ("threshold", Delta(value=5.0), "manual")
    setting = _built(contract=leak("setting_range")).param_set.get("shaft_seal.leak_flow_limit")
    assert setting.prior.family == "uniform"
    assert not [x for x in _built(contract=leak("example")).param_set.parameters if x.name == "leak_flow_limit"]


def _judge_all(inputs: list[dict[str, str]]) -> list[list[Any]]:
    return [[_judgement("h", 10000, 40000)] for _ in inputs]


def test_the_report_puts_judged_lives_first_and_scales_rul_by_the_template() -> None:
    """RUL at nominal follows the life prior; a bearing at +20% load lives 1.2 ** p shorter (ISO 281); a profile
    a component cannot respond to is not evaluable, with the reason."""
    from kaful.phases.p4_parameterize.tools.report import build_report

    built = _built(judge=_judge_all)
    report = build_report(built.param_set, SETTINGS.parameterize)
    assert report.judged_first == ("bearings",)
    assert [c.component for c in report.components] == ["bearings", "filter_element_fe", "shaft_seal"]
    bearings = report.components[0]
    runs = {p.profile: p for p in bearings.profiles}
    life = built.param_set.get("bearings.life_nominal").prior
    assert runs["nominal"].median == pytest.approx(life.quantile(0.5), rel=0.03)
    assert runs["load +20%"].median == pytest.approx(runs["nominal"].median / 1.2 ** 3, rel=0.02)
    assert not runs["temperature +10 K"].evaluable and "temperature" in runs["temperature +10 K"].note
    k = next(e for e in bearings.effects if e.param == "bearings.k")
    assert k.share > 0.99 and k.low_median > runs["nominal"].median > k.high_median
    filt = next(c for c in report.components if c.component == "filter_element_fe")
    assert filt.assumption.startswith("consumable_replacement: P(T < 4000) = 0.2")
    assert set(filt.alternatives) == set(SETTINGS.parameterize.interval_priors)


def test_a_declared_param_replaces_the_prior_and_its_k_and_bad_declarations_fail() -> None:
    from kaful.config import DeclaredParam
    from kaful.phases.p4_parameterize.e_value_roles import FLAG_DECLARED
    from tests.unit.test_kaful_scope import _machine

    machine = _machine(param=(DeclaredParam(param="bearings.life_nominal", unit="h", low=20000, high=80000,
                                            reason="engineer: L10 from the bearing catalogue"),))
    built = _built(machine=machine, judge=_judge_all)
    life, k = built.param_set.get("bearings.life_nominal"), built.param_set.get("bearings.k")
    assert (life.source, life.prior.median) == ("deployment", pytest.approx(40000))  # type: ignore[union-attr]
    assert (k.source, k.prior.median) == ("deployment", pytest.approx(1 / 40000))  # type: ignore[union-attr]
    assert FLAG_DECLARED in next(t for t in built.param_set.templates if t.component == "bearings").flags
    assert built.failures == ()
    bad = _machine(param=(DeclaredParam(param="bearings.life_nominal", unit="year", value=1, reason="r"),
                          DeclaredParam(param="bearings.nothing", unit="h", value=1, reason="r")))
    assert _built(machine=bad, judge=_judge_all).failures == (
        "machine.toml [[param]] bearings.life_nominal: unit 'year', the parameter's is 'h'",
        "machine.toml [[param]] bearings.nothing: no such parameter in this ParamSet")
