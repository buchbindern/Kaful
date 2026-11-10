"""Deterministic eval checks, golden comparators and version comparison."""

from typing import Any

import pytest

from kaful.evals.checks import classify_quote, resolve, run_check, score_golden
from kaful.evals.models import CheckOutcome, CheckSpec, GoldenSpec

SOURCE = """--- page 1 ---
  inlet_pressure (bar), range 0 to 10
  Rated flow: 6.0 m3/h (permitted range 2.0 to 8.0 m3/h)
"""
OUTPUT: dict[str, Any] = {"channels": [
    {"name": "inlet_pressure", "unit": "bar", "range_max": 10, "page": 1,
     "evidence": "inlet_pressure (bar), range 0 to 10"},
    {"name": "flow_rate", "unit": "M3/H", "range_max": 12.0, "page": 4,
     "evidence": "rated flow: 6.0 m3/h permitted range 2.0 to 8.0 m3/h"},
    {"name": "made_up", "unit": "A", "range_max": None, "page": 1, "evidence": "motor current 0 to 15 A"},
]}


def test_resolve_paths() -> None:
    assert resolve(OUTPUT, "channels[].name") == ["inlet_pressure", "flow_rate", "made_up"]
    assert resolve({"verdict": "supported"}, "verdict") == ["supported"]
    assert resolve(OUTPUT, "channels")[0] is OUTPUT["channels"]


@pytest.mark.parametrize(("quote", "expected"), [
    ("inlet_pressure (bar),   range 0 to 10", "verbatim"),
    ("RATED FLOW: 6.0 m3/h (permitted range 2.0 to 8.0 m3/h)", "near"),
    ("Rated flow: 6.5 m3/h", "unsupported"),
    ("", "unsupported"),
])
def test_classify_quote(quote: str, expected: str) -> None:
    assert classify_quote(quote, SOURCE) == expected


def test_quote_check_uses_judge_only_for_unplaced_quotes() -> None:
    asked: list[str] = []

    def judge(claim: str, source: str) -> str:
        asked.append(claim)
        return "unsupported"

    spec = CheckSpec(name="evidence", kind="quote_in_source", field="channels[].evidence", source="pages", judge=True)
    outcome, scores = run_check(spec, OUTPUT, {"pages": SOURCE}, judge)
    assert asked == ["motor current 0 to 15 A"]
    assert scores == {"evidence.verbatim": pytest.approx(1 / 3), "evidence.supported": pytest.approx(2 / 3)}
    assert outcome.status == "fail" and "judge: unsupported" in outcome.details[-1]


def test_range_count_and_unit_checks() -> None:
    pages = CheckSpec(name="page_in_range", kind="in_range", field="channels[].page", min=1, max=3)
    outcome, scores = run_check(pages, OUTPUT, {}, None)
    assert outcome.status == "fail" and scores == {"page_in_range": pytest.approx(2 / 3)}
    count = CheckSpec(name="channel_count", kind="count_between", field="channels", min=1, max=2)
    assert run_check(count, OUTPUT, {}, None)[1] == {"channel_count": 0.0}
    units = CheckSpec(name="unit_parses", kind="unit_parses", field="channels[].unit")
    # Parsing is all this checks: 'M3/H' parses (as molar**3/henry); units_correct catches it.
    assert run_check(units, OUTPUT, {}, None) == (CheckOutcome(check="unit_parses", status="pass"),
                                                  {"unit_parses": 1.0})
    outcome, scores = run_check(units, {"channels": [{"unit": "bar"}, {"unit": "furlongz"}]}, {}, None)
    assert outcome.status == "fail" and scores == {"unit_parses": 0.5}
    assert outcome.details[0].startswith("cannot parse unit 'furlongz'")
    assert run_check(units, {"channels": [{"unit": None}]}, {}, None)[0].status == "skip"


@pytest.mark.parametrize(("spec", "expected"), [
    (dict(field="channels[].name", compare="set_f1", expected=["inlet_pressure", "flow_rate", "motor_current", "x"]),
     2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4))),
    (dict(field="channels", compare="keyed", key="name", value="unit",
          expected={"inlet_pressure": "bar", "flow_rate": "m3/h", "motor_current": "A"}), 2 / 3),
    (dict(field="channels", compare="keyed", key="name", value="range_max",
          expected={"inlet_pressure": 10.0, "flow_rate": 12}), 1.0),
    (dict(field="channels", compare="keyed", key="name", value="range_max", rel_tol=0.1,
          expected={"flow_rate": 11.0}), 1.0),
    (dict(field="verdict", compare="exact", expected="partial"), 0.0),
    (dict(field="verdict", compare="violation", expected="supported"), 1.0),
])
def test_golden_comparators(spec: dict[str, Any], expected: float) -> None:
    data = {**OUTPUT, "verdict": "supported"}
    assert score_golden(GoldenSpec(metric="m", **spec), data) == pytest.approx(expected)


def test_check_specs_validate_their_arguments() -> None:
    with pytest.raises(ValueError, match="needs `source`"):
        CheckSpec(name="q", kind="quote_in_source", field="x")
    with pytest.raises(ValueError, match="min and/or max"):
        CheckSpec(name="r", kind="in_range", field="x")
    with pytest.raises(ValueError, match="keyed needs"):
        GoldenSpec(metric="m", field="x", compare="keyed", expected=[1])


def test_within_excludes_all_and_listed_in_source() -> None:
    from kaful.evals.checks import score_golden
    from kaful.evals.models import GoldenSpec

    within = GoldenSpec(metric="no_extra", field="channels[].name", compare="within",
                        expected=["inlet_pressure", "flow_rate"])
    assert score_golden(within, OUTPUT) == pytest.approx(2 / 3)  # made_up is extra
    assert score_golden(within, {"channels": []}) == 1.0
    excludes = GoldenSpec(metric="no_sensor", field="channels[].name", compare="excludes_all",
                          expected=["made up", "sensor"])
    assert score_golden(excludes, OUTPUT) == 0.5
    listed = CheckSpec(name="known", kind="listed_in_source", field="channels[].name", source="names")
    outcome, scores = run_check(listed, OUTPUT, {"names": "- inlet_pressure: x\n- flow_rate: y"}, None)
    assert outcome.status == "fail" and outcome.details == ("not listed: 'made_up'",)
    assert scores == {"known": pytest.approx(2 / 3)}
    also = listed.model_copy(update={"also_listed": "added[].id"})
    outcome, scores = run_check(also, {**OUTPUT, "added": [{"id": "made_up"}]},
                                {"names": "- inlet_pressure: x\n- flow_rate: y"}, None)
    assert outcome.status == "pass" and scores == {"known": 1.0}


def test_a_word_hyphenated_at_a_line_break_is_verbatim_as_in_the_pipeline() -> None:
    """eversys p114 (2026-11-08): 'life time cy-\\ncle' quoted as 'life time cycle' is verbatim to the pipeline's
    rule-4 checker (kaful.evidence, matcher-v3); the eval check must agree. A range '10-\\n30' stays apart."""
    from kaful.evals.checks import classify_quote

    source = "The water filter has to be\nchanged after its life time cy-\ncle. Range 10-\n30 um."
    assert classify_quote("The water filter has to be changed after its life time cycle.", source) == "verbatim"
    assert classify_quote("Range 1030 um.", source) == "unsupported"
    assert classify_quote("changed after its life time cy-cle.", source) == "verbatim"  # matcher-v4: both sides


def test_a_judged_range_covers_hidden_values_after_unit_conversion_and_must_be_sharp() -> None:
    """M9 life_judgement: coverage converts units; sharpness (high / low <= max) keeps coverage honest."""
    judged = {"life": {"unit": "h", "low": 1500.0, "high": 6000.0}}
    covers = GoldenSpec(metric="covers", field="life", compare="covers", key="hour", expected=[2000.0, 8000.0])
    assert score_golden(covers, judged) == 0.5
    in_minutes = covers.model_copy(update={"key": "minute", "expected": [120000.0]})
    assert score_golden(in_minutes, judged) == 1.0
    assert score_golden(covers.model_copy(update={"key": "kg"}), judged) == 0.0  # does not convert
    sharp = CheckSpec(name="sharp", kind="ratio_at_most", field="life", max=100.0)
    outcome, scores = run_check(sharp, judged, {}, None)
    assert (outcome.status, outcome.details, scores) == ("pass", ("high/low = 4",), {"sharp": 1.0})
    wide = {"life": {"unit": "h", "low": 10.0, "high": 1e5}}
    outcome, scores = run_check(sharp, wide, {}, None)
    assert (outcome.status, scores) == ("fail", {"sharp": 0.0}) and outcome.details == ("high/low = 1e+04",)
