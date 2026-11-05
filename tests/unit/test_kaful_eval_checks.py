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
