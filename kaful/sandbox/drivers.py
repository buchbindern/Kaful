"""Committed drivers: the only code the sandbox runs from outside a machine directory (M11, Natalie 2026-11-13).

A driver is pinned by its path in the repo and its sha256. The parent checks both before it starts the child,
and the child checks the hash again before loading the file, so an edited driver cannot run until its new
hash is written here (a reviewed change; a test fails when the two disagree). A driver is committed code that
loads generated code from the machine directory inside the child.
"""

from pathlib import Path
from typing import NamedTuple

from kaful.runs import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]


class Driver(NamedTuple):
    path: str  # relative to the repo root
    sha256: str


# Empty since M15: the legacy probe (M11), its only driver, was deleted with the legacy twins (tag legacy-final).
DRIVERS: dict[str, Driver] = {}


class DriverNotAllowed(ValueError):
    """The name is not a pinned driver, or the file at its path is missing or has another hash."""


def driver_file(name: str) -> Path:
    """The pinned driver's file, after checking that its content still has the pinned hash."""
    if name not in DRIVERS:
        raise DriverNotAllowed(f"{name!r} is not a pinned sandbox driver ({', '.join(sorted(DRIVERS))})")
    driver = DRIVERS[name]
    path = REPO_ROOT / driver.path
    if not path.is_file():
        raise DriverNotAllowed(f"driver {name}: {driver.path} not found")
    actual = sha256_file(path)
    if actual != driver.sha256:
        raise DriverNotAllowed(f"driver {name}: {driver.path} has sha256 {actual}, pinned {driver.sha256}")
    return path
