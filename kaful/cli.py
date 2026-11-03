"""Command line: `kaful run <machine> [--from P | --only P ...] [--budget USD]
[--allow-failures] [--base-run RUN_ID]`.

Exit codes: 0 passed, 1 failed or error, 2 usage error (argparse),
3 a selected phase is not implemented yet.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import cast

from kaful.config import KafulSettings, MachineConfigError
from kaful.logs import configure_logging
from kaful.phases import PHASE_ORDER, PhaseName, PhaseSelectionError, select_phases
from kaful.pipeline import RunRequest, execute, start_run
from kaful.runs import BaseRunError

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_NOT_IMPLEMENTED = 3

log = logging.getLogger("kaful.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kaful", description="Kaful digital-twin pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the pipeline for one machine")
    run.add_argument("machine", help="machine id (directory under machines/)")
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--from", dest="from_phase", choices=PHASE_ORDER, help="start at this phase")
    selection.add_argument("--only", nargs="+", choices=PHASE_ORDER, default=[], help="run only these phases")
    run.add_argument("--budget", type=float, default=None, help="API budget in USD for this run")
    run.add_argument("--allow-failures", action="store_true", help="continue past failed checks (recorded)")
    run.add_argument("--base-run", default=None, help="run id whose upstream artifacts are reused")
    return parser


def _run(args: argparse.Namespace, argv: Sequence[str], settings: KafulSettings) -> int:
    if args.budget is not None and args.budget <= 0:
        raise ValueError("--budget must be > 0")
    request = RunRequest(
        machine_id=args.machine,
        argv=tuple(argv),
        from_phase=cast(PhaseName | None, args.from_phase),
        only=tuple(cast(list[PhaseName], args.only)),
        budget_usd=args.budget,
        allow_failures=args.allow_failures,
        base_run=args.base_run,
    )
    selected = select_phases(request.from_phase, request.only)
    ctx, base = start_run(settings, request)
    manifest = execute(ctx, selected, base)
    sys.stdout.write(f"{manifest.status}: {ctx.run_dir}\n")
    if manifest.status == "passed":
        return EXIT_PASSED
    if manifest.status == "not_implemented":
        return EXIT_NOT_IMPLEMENTED
    return EXIT_FAILED


def main(argv: Sequence[str] | None = None, settings: KafulSettings | None = None) -> int:
    """Entry point for the `kaful` command and `python -m kaful`."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    settings = settings if settings is not None else KafulSettings()
    configure_logging(settings.log_level)
    try:
        return _run(args, args_list, settings)
    except (MachineConfigError, PhaseSelectionError, BaseRunError, FileNotFoundError, ValueError) as exc:
        log.error("run not started", extra={"fields": {"error": str(exc)}})
        sys.stderr.write(f"kaful: error: {exc}\n")
        return EXIT_FAILED
