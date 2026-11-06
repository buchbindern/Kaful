"""The understand golden label files (eversys, CNC): well-formed; once marked reviewed, complete."""

import tomllib
from pathlib import Path

import pytest

from kaful.artifacts.base import VALUE_ROLES

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
    values = data.get("values", [])  # optional: value roles (M7)
    roles = [row["role"] for row in values if row["role"]]
    assert set(roles) <= set(VALUE_ROLES), sorted(set(roles) - set(VALUE_ROLES))
    assert all(row["suggested_role"] in VALUE_ROLES and isinstance(row["page"], int) for row in values)
    if data["status"] == "reviewed":
        assert len(roles) == len(values), "values: rows without a role in a reviewed file"


def test_both_scored_machines_have_labels() -> None:
    assert sorted(p.stem for p in GOLDEN.glob("*.toml")) == ["cnc_spindle", "eversys_coffee_machine"]
