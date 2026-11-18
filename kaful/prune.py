"""`kaful runs prune <machine>` (M15): which runs can be deleted, and why. A tool; no API calls.

Each estimate run writes ~70-80 MB (mostly sample CSVs), and failed or abandoned runs pile up. A run is kept when:
- it is named with --keep;
- a file in `settings.runs.cited_in` (PLAN.md, the milestone docs) cites its id, in full or as its 6-character suffix;
- it is the newest run in which some phase passed;
- it is marked running and started less than `stale_after_hours` ago (it may be live);
- its manifest cannot be read (nothing is known about it, so nothing is assumed);
- a kept run reads its artifacts or names it as base run (transitively: a kept run's whole lineage stays).
Every other run is deleted: failed (or otherwise unfinished) runs, stale running runs, and superseded runs.

Without --apply nothing is deleted: the plan is printed with each run's reason and size (a dry run).
"""

import argparse
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from kaful.artifacts.manifest import RunManifest
from kaful.config import REPO_ROOT, KafulSettings, RunsSettings
from kaful.phases import PHASE_ORDER
from kaful.runs import MANIFEST_FILE, RUN_ID_PATTERN, now_utc, read_manifest, runs_dir

EXIT_OK = 0
SUFFIX_CHARS = 6
SECONDS_PER_HOUR = 3600.0
BYTES_PER_MB = 1e6


@dataclass(frozen=True)
class RunVerdict:
    run_id: str
    status: str
    keep: bool
    reason: str
    size_bytes: int


def _size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _cited(run_ids: Sequence[str], settings: RunsSettings, root: Path) -> dict[str, str]:
    """Run id -> the first configured file (in setting order, then path order) that cites it."""
    out: dict[str, str] = {}
    for pattern in settings.cited_in:
        for path in sorted(p for p in root.glob(pattern) if p.is_file()):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            for run_id in run_ids:
                suffix = run_id[-SUFFIX_CHARS:]
                if run_id not in out and (run_id in text or re.search(rf"\b{suffix}\b", text)):
                    out[run_id] = rel
    return out


def _reads(manifest: RunManifest) -> set[str]:
    """Runs whose artifacts this run uses, and its base run."""
    used = {ref.run_id for record in manifest.phases for ref in record.artifacts}
    if manifest.base_run is not None:
        used.add(manifest.base_run.run_id)
    return used - {manifest.run_id}


def _latest(manifests: Sequence[RunManifest]) -> dict[str, list[str]]:
    """Run id -> the phases it is the newest pass of."""
    out: dict[str, list[str]] = {}
    for phase in PHASE_ORDER:
        passed = [m for m in manifests if m.phase(phase).status == "passed"]
        if passed:
            newest = max(passed, key=lambda m: (m.created_at, m.run_id))
            out.setdefault(newest.run_id, []).append(phase)
    return out


def _root_reason(m: RunManifest, keep: Sequence[str], cited: dict[str, str], latest: dict[str, list[str]],
                 age_h: float, settings: RunsSettings) -> str | None:
    if m.run_id in keep:
        return "named with --keep"
    if m.run_id in cited:
        return f"cited in {cited[m.run_id]}"
    if m.run_id in latest:
        return f"latest run that passed {', '.join(sorted(latest[m.run_id]))}"
    if m.status == "running" and age_h < settings.stale_after_hours:
        return f"marked running, started {age_h:.1f} h ago: may be live"
    return None


def _drop_reason(m: RunManifest, age_h: float) -> str:
    if m.status == "running":
        return f"stale: still marked running {age_h:.1f} h after it started"
    return "superseded" if m.status == "passed" else m.status


def prune_plan(machine_dir: Path, settings: RunsSettings, now: datetime, root: Path = REPO_ROOT,
               keep: Sequence[str] = ()) -> tuple[RunVerdict, ...]:
    """Every run directory of the machine, kept or not, with the reason (sorted by run id)."""
    base = runs_dir(machine_dir)
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and RUN_ID_PATTERN.match(d.name)) if base.is_dir() else []
    manifests: dict[str, RunManifest] = {}
    reasons: dict[str, str] = {}
    for d in dirs:
        try:
            manifests[d.name] = read_manifest(d)
        except (OSError, ValueError, ValidationError):
            reasons[d.name] = "no readable manifest"
    cited = _cited([d.name for d in dirs], settings, root)
    latest = _latest(list(manifests.values()))
    ages = {r: (now - m.created_at).total_seconds() / SECONDS_PER_HOUR for r, m in manifests.items()}
    for run_id, m in manifests.items():
        reason = _root_reason(m, keep, cited, latest, ages[run_id], settings)
        if reason is not None:
            reasons[run_id] = reason
    frontier = sorted(r for r in reasons if r in manifests)
    while frontier:
        reader = frontier.pop(0)
        for used in sorted(_reads(manifests[reader])):
            if used in manifests and used not in reasons:
                reasons[used] = f"read by {reader}"
                frontier.append(used)
    out = []
    for d in dirs:
        found = manifests.get(d.name)
        kept = d.name in reasons  # every run without a readable manifest is kept, so `found` is set when dropped
        reason = reasons[d.name] if kept or found is None else _drop_reason(found, ages[d.name])
        out.append(RunVerdict(run_id=d.name, status=found.status if found is not None else "unknown", keep=kept,
                              reason=reason, size_bytes=_size(d)))
    return tuple(out)


def apply_plan(machine_dir: Path, plan: Sequence[RunVerdict]) -> int:
    """Delete the plan's dropped runs; returns the bytes freed."""
    freed = 0
    for v in plan:
        if v.keep:
            continue
        run_dir = runs_dir(machine_dir) / v.run_id
        if not RUN_ID_PATTERN.match(v.run_id) or not (run_dir / MANIFEST_FILE).is_file():
            raise ValueError(f"refusing to delete {run_dir}: not a run directory with a manifest")
        shutil.rmtree(run_dir)
        freed += v.size_bytes
    return freed


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("runs", help="run directory tools (offline)").add_subparsers(
        dest="runs_command", required=True)
    prune = sub.add_parser("prune", help="list the runs that can be deleted, and why; --apply deletes them")
    prune.add_argument("machine")
    prune.add_argument("--keep", action="append", default=[], help="a run id to keep (repeatable)")
    prune.add_argument("--apply", action="store_true", help="delete the listed runs (default: a dry run)")


def _table(plan: Sequence[RunVerdict]) -> str:
    rows = [f"  {'keep' if v.keep else 'DELETE':6s} {v.run_id}  {v.status:15s} {v.size_bytes / BYTES_PER_MB:8.1f} MB  "
            f"{v.reason}" for v in plan]
    return "\n".join(rows) + "\n"


def runs_prune(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    plan = prune_plan(machine_dir, settings.runs, now_utc(), REPO_ROOT, tuple(args.keep))
    drop = [v for v in plan if not v.keep]
    size = sum(v.size_bytes for v in drop) / BYTES_PER_MB
    sys.stdout.write(_table(plan))
    if not args.apply:
        sys.stdout.write(f"{args.machine}: {len(drop)} runs to delete ({size:.1f} MB); dry run: nothing deleted "
                         "(--apply deletes them)\n")
        return EXIT_OK
    freed = apply_plan(machine_dir, plan) / BYTES_PER_MB
    sys.stdout.write(f"{args.machine}: deleted {len(drop)} runs, freed {freed:.1f} MB\n")
    return EXIT_OK
