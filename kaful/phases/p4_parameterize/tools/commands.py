"""Parameterize tools (offline; no API calls).

- `kaful parameterize report <machine> [--run RUN_ID]` writes the provenance and sensitivity report of a run's
  ParamSet (tools/report) to <machine>/outputs/param_reports/<run_id>.json and .md.
"""

import argparse
import sys

from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.param_set import ParamSet
from kaful.config import KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p2_understand.tools.commands import SCORED
from kaful.phases.p4_parameterize.tools.report import build_report, report_md
from kaful.runs import dump_json, list_manifests, load_artifact, write_text_atomic

REPORTS_DIR = ("outputs", "param_reports")
PHASE: PhaseName = "parameterize"
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("parameterize", help="parameterize tools (offline)").add_subparsers(
        dest="parameterize_command", required=True)
    report = sub.add_parser("report", help="write the provenance and sensitivity report of a run's ParamSet")
    report.add_argument("machine")
    report.add_argument("--run", help="run id (default: the latest run whose parameterize phase passed)")


def run_with_param_set(manifests: list[RunManifest], run_id: str | None) -> RunManifest:
    passed = [m for m in manifests if m.phase(PHASE).status in SCORED]
    picked = [m for m in passed if m.run_id == run_id] if run_id else passed[-1:]
    if not picked:
        raise ValueError(f"no run {run_id!r} with a passed parameterize phase" if run_id
                         else "no run with a passed parameterize phase")
    return picked[0]


def parameterize_report(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    manifest = run_with_param_set(list_manifests(machine_dir), args.run)
    [ref] = [r for r in manifest.phase(PHASE).artifacts if r.kind == "param_set"]
    report = build_report(load_artifact(machine_dir, ref, ParamSet), settings.parameterize)
    path = machine_dir.joinpath(*REPORTS_DIR, f"{manifest.run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(report))
    write_text_atomic(path.with_suffix(".md"), report_md(report))
    sys.stdout.write(f"wrote {path.relative_to(machine_dir)} and {path.with_suffix('.md').name}\n")
    return EXIT_OK
