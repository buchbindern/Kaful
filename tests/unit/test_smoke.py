"""Smoke test: the legacy config resolves machine paths without network access."""

from pathlib import Path

import config

REFERENCE_MACHINE = "eversys_coffee_machine"


def test_get_machine_config_resolves_reference_machine() -> None:
    cfg = config.get_machine_config(REFERENCE_MACHINE)

    assert cfg["machine_id"] == REFERENCE_MACHINE
    assert cfg["machine_dir"] == config.MACHINES_DIR / REFERENCE_MACHINE
    assert cfg["collection_name"] == f"{REFERENCE_MACHINE}_manual"


def test_machine_config_paths_stay_inside_machine_dir() -> None:
    cfg = config.get_machine_config(REFERENCE_MACHINE)
    machine_dir: Path = cfg["machine_dir"]

    paths = [v for v in cfg.values() if isinstance(v, Path)]
    assert paths
    assert all(p == machine_dir or machine_dir in p.parents for p in paths)


def test_reference_machine_queries_file_exists() -> None:
    assert (config.MACHINES_DIR / REFERENCE_MACHINE / "queries.py").is_file()
