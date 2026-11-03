"""A throwaway machine directory for kaful run/CLI tests."""

from pathlib import Path

from kaful.config import KafulSettings

MACHINE_ID = "toy_machine"
MACHINE_TOML = """\
machine_id = "toy_machine"
display_name = "Toy machine"
time_unit = "h"

[[documents]]
path = "manual/toy_manual.txt"
kind = "manual"
"""


def make_machine(machines_dir: Path, machine_id: str = MACHINE_ID) -> Path:
    """Create machines_dir/<machine_id> with machine.toml and one document."""
    machine_dir = machines_dir / machine_id
    (machine_dir / "manual").mkdir(parents=True)
    (machine_dir / "machine.toml").write_text(MACHINE_TOML.replace(MACHINE_ID, machine_id))
    (machine_dir / "manual" / "toy_manual.txt").write_text("Replace the seal every 2000 h.\n")
    return machine_dir


def settings_for(machines_dir: Path) -> KafulSettings:
    return KafulSettings(machines_dir=machines_dir)
