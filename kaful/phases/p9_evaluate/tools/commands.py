"""Evaluate tools (offline; no API calls).

- `kaful evaluate report <machine> --run RUN_ID` builds the evaluation report of any run's lineage, through the
  phase's own `evaluate`, and saves it to <machine>/outputs/evaluation/<run_id>.json and .md. It exists for a run that
  cannot be a base run (a required phase failed, e.g. CDV's understand gate): the pipeline's `--only evaluate` needs
  every upstream phase passed. The saved report carries the evaluated run's id; it was not produced by a pipeline run.
"""

import argparse
import sys

from kaful.config import KafulSettings
from kaful.phases.p9_evaluate.helpers.render import report_md
from kaful.phases.p9_evaluate.phase import evaluate
from kaful.runs import RUN_ID_PATTERN, dump_json, read_manifest, runs_dir, write_text_atomic

OUT_DIR = ("outputs", "evaluation")
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("evaluate", help="evaluate tools (offline)").add_subparsers(
        dest="evaluate_command", required=True)
    rep = sub.add_parser("report", help="the evaluation report of any run's lineage (e.g. one that stopped early)")
    rep.add_argument("machine")
    rep.add_argument("--run", required=True, help="the run whose lineage is evaluated")


def evaluate_report(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    if not RUN_ID_PATTERN.match(args.run) or not (runs_dir(machine_dir) / args.run).is_dir():
        raise ValueError(f"run {args.run!r} not found for {args.machine}")
    manifest = read_manifest(runs_dir(machine_dir) / args.run)
    report = evaluate(manifest, args.run, args.run, machine_dir, settings)
    out = machine_dir.joinpath(*OUT_DIR)
    write_text_atomic(out / f"{args.run}.json", dump_json(report))
    write_text_atomic(out / f"{args.run}.md", report_md(report))
    sys.stdout.write(f"{args.machine} {args.run}: complete={report.complete}; "
                     f"{len(report.not_reached)} sections not reached -> {out.name}/{args.run}.md\n")
    return EXIT_OK
