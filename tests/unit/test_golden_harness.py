"""Golden-run harness (M2): tolerance compare and scratch-copy fixes. Offline, fast."""

from pathlib import Path

import pytest

from tests.harness.golden import (
    COMPOSITE, INPUTS_DIR, _repoint_composite, compare, load_expected, write_expected)

RULES = [
    {"files": "*step_d_validation.json", "keys": ".composite_path", "ignore": True},
    {"files": "*step_c_rul.json", "keys": "*.rul_mean", "rel": 0.05},
]


def test_compare_applies_first_matching_rule() -> None:
    exp = {"a/step_c_rul.json": {"e": {"rul_mean": 100.0, "rul_std": 10.0}}}
    act = {"a/step_c_rul.json": {"e": {"rul_mean": 104.0, "rul_std": 10.5}}}
    assert compare(exp, act, RULES) == ["a/step_c_rul.json:.e.rul_std: 10.0 → 10.5"]


def test_compare_ignores_and_reports_structure_changes() -> None:
    exp = {"x/step_d_validation.json": {"composite_path": "/old", "n": [1, 2], "gone": 1}}
    act = {"x/step_d_validation.json": {"composite_path": "/new", "n": [1], "new": True},
           "x/extra.json": {}}
    diffs = compare(exp, act, RULES)
    assert "x/extra.json: not in snapshot" in diffs
    assert "x/step_d_validation.json:.gone: only in snapshot" in diffs
    assert "x/step_d_validation.json:.new: only in run" in diffs
    assert "x/step_d_validation.json:.n: length 2 → 1" in diffs
    assert not any("composite_path" in d for d in diffs)


def test_compare_is_exact_without_rules_and_types_matter() -> None:
    assert compare({"f": {"v": 1.0}}, {"f": {"v": 1.0 + 1e-12}}, []) == ["f:.v: 1.0 → 1.000000000001"]
    assert compare({"f": {"v": True}}, {"f": {"v": 1}}, []) == ["f:.v: 'True' → '1'"]
    assert compare({"f": {"v": None}}, {"f": {"v": 3.0}}, [{"rel": 1.0}]) == ["f:.v: 'None' → '3.0'"]


def test_snapshot_round_trips_byte_stable(tmp_path: Path) -> None:
    data = {"outputs/estimate/p/step_c_rul.json": {"b": 2, "a": [1.5, None]}}
    write_expected(data, tmp_path / "one")
    write_expected(data, tmp_path / "two")
    one = next((tmp_path / "one").rglob("*.gz")).read_bytes()
    assert one == next((tmp_path / "two").rglob("*.gz")).read_bytes()
    assert load_expected(tmp_path / "one") == data


def test_repoint_composite_rewrites_only_the_sys_path_line(tmp_path: Path) -> None:
    path = tmp_path / COMPOSITE
    path.parent.mkdir(parents=True)
    path.write_text("import sys\nsys.path.insert(0, '/Users/someone/x/components')\nfrom a import A\n")
    log = _repoint_composite(tmp_path)
    text = path.read_text()
    assert "/Users/someone" not in text and str(tmp_path / "outputs/codegen/components") in text
    assert text.startswith("import sys\n") and text.endswith("from a import A\n")
    assert "rule 12" in log and "/Users/someone/x/components" in log


def test_repoint_composite_refuses_unexpected_layout(tmp_path: Path) -> None:
    path = tmp_path / COMPOSITE
    path.parent.mkdir(parents=True)
    path.write_text("import sys\n")
    with pytest.raises(RuntimeError, match="expected one sys.path.insert"):
        _repoint_composite(tmp_path)


def test_packed_inputs_are_present() -> None:
    names = {p.name for p in INPUTS_DIR.rglob("*.gz")}
    assert "ceramic_grinders.py.gz" in names
    assert "coffee_machine_events_light_commercial.csv.gz" in names
