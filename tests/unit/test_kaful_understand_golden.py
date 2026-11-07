"""The understand golden label files (eversys, CNC): well-formed; once marked reviewed, complete."""

import re
import tomllib
from pathlib import Path

import pytest

from kaful.artifacts.base import VALUE_ROLES
from kaful.artifacts.sensor_contract import DUAL_ROLES

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "understand" / "golden"
SECTIONS = ("legacy_fields", "candidate_channels", "components", "fault_modes")
VERDICTS = {
    "legacy_fields": {"channel", "not_channel", "unsure"},
    "candidate_channels": {"channel", "not_channel", "unsure"},
    "components": {"top_level", "child", "not_a_component"},
}


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.toml")), ids=lambda p: p.stem)
def test_golden_labels_are_well_formed(path: Path) -> None:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["machine_id"] == path.stem and data["status"] in ("draft", "reviewed")
    for section in SECTIONS:
        rows = data[section]
        assert rows and all("verdict" in row for row in rows), section
        given = [row["verdict"] for row in rows if row["verdict"]]
        allowed = VERDICTS.get(section)
        if allowed is not None:
            assert set(given) <= allowed, (section, sorted(set(given) - allowed))
        if data["status"] == "reviewed":
            assert len(given) == len(rows), f"{section}: unlabelled rows in a reviewed file"
    candidates = data.get("candidate_fault_modes", [])  # optional: fault modes Natalie named (M7)
    given = [row["verdict"] for row in candidates if row["verdict"]]
    assert set(given) <= {"yes", "no"} and all(isinstance(row["page"], int) for row in candidates)
    if data["status"] == "reviewed":
        assert len(given) == len(candidates), "candidate_fault_modes: unlabelled rows in a reviewed file"
    values = data.get("values", [])  # optional: value roles (M7)
    roles = [row["role"] for row in values if row["role"]]
    assert set(roles) <= set(VALUE_ROLES), sorted(set(roles) - set(VALUE_ROLES))
    assert all(row["suggested_role"] in VALUE_ROLES and isinstance(row["page"], int) for row in values)
    if data["status"] == "reviewed":
        assert len(roles) == len(values), "values: rows without a role in a reviewed file"


@pytest.mark.parametrize("path", sorted(GOLDEN.glob("*.toml")), ids=lambda p: p.stem)
def test_golden_labels_follow_the_contract_rules(path: Path) -> None:
    """The rules the SensorContract and MachineModel enforce hold for the labels too (decisions 2026-11-06)."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    rows = data["legacy_fields"] + data["candidate_channels"]
    channels = [r for r in rows if r["verdict"] == "channel"]
    clocks = [r["name"] for r in channels if r["role"] == "usage_clock"]
    assert len(clocks) <= 1, f"at most one usage clock (the machine's time axis), got {clocks}"
    for row in rows:
        second = row["secondary_role"]
        if second:
            assert row["verdict"] == "channel" and second in DUAL_ROLES and row["role"] in DUAL_ROLES, row["name"]
            assert second != row["role"], row["name"]
    for row in data["components"]:
        assert row["verdict"] != "child" or row["parent"], f"{row['id']}: a child needs its parent"
    for row in rows + data["components"]:
        aliases = row.get("aliases", [])
        assert isinstance(aliases, list) and all(isinstance(a, str) and a for a in aliases), row
    for row in data["fault_modes"]:
        assert re.fullmatch(r"(\d+(-\d+)?)?", row["page"]), row  # a page, a range "60-61", or none
        if "-" in row["page"]:
            first, last = (int(p) for p in row["page"].split("-"))
            assert first < last and re.fullmatch(r"\S+-\S+", row.get("page_label", "")), row
        assert not row["page"] or row["doc_id"], f"{row['component']} / {row['what_degrades']}: page without doc_id"


def test_both_scored_machines_have_labels() -> None:
    assert sorted(p.stem for p in GOLDEN.glob("*.toml")) == ["cnc_spindle", "eversys_coffee_machine"]
