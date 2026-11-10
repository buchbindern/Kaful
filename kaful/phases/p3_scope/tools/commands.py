"""Scope tools (offline; no API calls).

- `kaful scope score <machine> [--run RUN_ID]` scores a run's Scope against the understand golden labels
  (tools/score) and saves it to <machine>/outputs/scope_scores/<run_id>.json.
"""

import argparse
import sys
from pathlib import Path

from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.scope import Scope
from kaful.config import REPO_ROOT, KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p2_understand.tools.commands import GOLDEN_DIR, SCORED
from kaful.phases.p3_scope.tools.score import ScopeScore, score
from kaful.runs import dump_json, list_manifests, load_artifact, write_text_atomic

SCORES_DIR = ("outputs", "scope_scores")
PHASE: PhaseName = "scope"
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    scope = commands.add_parser("scope", help="scope tools (offline)").add_subparsers(
        dest="scope_command", required=True)
    scored = scope.add_parser("score", help="score a run's Scope against the understand golden labels")
    scored.add_argument("machine")
    scored.add_argument("--run", help="run id (default: the latest run whose scope phase passed)")


def _golden(machine: str) -> Path:
    golden = GOLDEN_DIR / f"{machine}.toml"
    if not golden.is_file():
        raise ValueError(f"no golden label file {golden.relative_to(REPO_ROOT)}")
    return golden


def _run(manifests: list[RunManifest], run_id: str | None) -> RunManifest:
    passed = [m for m in manifests if m.phase(PHASE).status in SCORED]
    picked = [m for m in passed if m.run_id == run_id] if run_id else passed[-1:]
    if not picked:
        raise ValueError(f"no run {run_id!r} with a passed scope phase" if run_id
                         else "no run with a passed scope phase")
    return picked[0]


def summary(result: ScopeScore) -> str:
    s, n = result.precision_strict, result.precision_lenient
    return (f"full models: {len(result.full_models)}\n"
            f"recall (given what understand extracted): {result.recall} ({len(result.found)} found, "
            f"missed {list(result.missed)}; criterion 1 unflagged {list(result.unflagged)})\n"
            f"precision strict {s.value}, lenient {n.value}; wrong (strict) {list(s.wrong)}; "
            f"unmatched {list(result.unmatched_full_models)}\n"
            f"not extracted {list(result.not_extracted)}; no fault mode {list(result.no_fault_mode)}\n")


def scope_score(args: argparse.Namespace, settings: KafulSettings) -> int:
    golden, machine_dir = _golden(args.machine), settings.machine_dir(args.machine)
    manifest = _run(list_manifests(machine_dir), args.run)
    [ref] = [r for r in manifest.phase(PHASE).artifacts if r.kind == "scope"]
    result = score(golden, load_artifact(machine_dir, ref, Scope))
    path = machine_dir.joinpath(*SCORES_DIR, f"{manifest.run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(result))
    sys.stdout.write(summary(result) + f"wrote {path.relative_to(machine_dir)}\n")
    return EXIT_OK
