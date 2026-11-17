"""Reading run ledgers (used by c_prompt_quality and e_cost).

Each run keeps its own ledger at outputs/runs/<run_id>/ledger.jsonl; a phase's rows in its producing run are what
that phase called (a reused phase made no call in the evaluated run).
"""

from pathlib import Path

from kaful.llm.ledger import LedgerRow, read_ledger
from kaful.paths import resolve_in

RUNS = "outputs/runs"
LEDGER = "ledger.jsonl"


def ledger_rel(run_id: str) -> str:
    return f"{RUNS}/{run_id}/{LEDGER}"


def run_rows(machine_dir: Path, run_id: str) -> list[LedgerRow]:
    return read_ledger(resolve_in(machine_dir, ledger_rel(run_id)))


def phase_rows(machine_dir: Path, run_id: str | None, phase: str) -> list[LedgerRow]:
    return [r for r in run_rows(machine_dir, run_id) if r.phase == phase] if run_id else []


def all_runs(machine_dir: Path) -> list[str]:
    """Every run of the machine that has a ledger, sorted by run id (time order)."""
    root = resolve_in(machine_dir, RUNS)
    return sorted(p.parent.name for p in root.glob(f"*/{LEDGER}")) if root.is_dir() else []


def count_runs(machine_dir: Path) -> int:
    """Every run directory of the machine, with or without a ledger."""
    root = resolve_in(machine_dir, RUNS)
    return sum(p.is_dir() for p in root.iterdir()) if root.is_dir() else 0
