"""Command line.

    kaful run <machine> [--from P | --only P ...] [--budget USD] [--allow-failures]
              [--base-run RUN_ID] [--no-llm-cache] [--strict]
    kaful eval run <prompt_id> [--live --budget USD] [--batch] [--model M] [--effort E]
    kaful eval compare <prompt_id> VERSION[@MODEL] VERSION[@MODEL]
    kaful prompts check | lock <prompt_id>
    kaful ingest review [machine ...] [--per-class N] | classifier-eval [--labels FILE]
    kaful ingest grounding <machine> [--run RUN_ID] [--freeze]

Exit codes: 0 passed, 1 failed or error, 2 usage error (argparse),
3 a selected phase is not implemented yet.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import cast

from kaful.config import KafulSettings, MachineConfigError
from kaful.evals import commands as eval_commands
from kaful.llm.anthropic_api import AnthropicTransport
from kaful.llm.budget import BudgetExceeded
from kaful.llm.embeddings import EmbeddingTransport
from kaful.llm.openai_api import OpenAIEmbeddingTransport
from kaful.llm.transport import Transport
from kaful.llm.gateway import OfflineMissError
from kaful.logs import configure_logging
from kaful.phases import PHASE_ORDER, PhaseName, PhaseSelectionError, select_phases
from kaful.phases.p1_ingest.tools import commands as ingest_commands
from kaful.phases.p2_understand.tools import commands as understand_commands
from kaful.phases.p3_scope.tools import commands as scope_commands
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
    run.add_argument("--no-llm-cache", action="store_true", help="ignore cached LLM responses (still record new ones)")
    run.add_argument("--strict", action="store_true",
                     help="index-hash mismatches and unlisted documents are errors, not warnings")
    eval_commands.add_commands(commands)
    ingest_commands.add_commands(commands)
    understand_commands.add_commands(commands)
    scope_commands.add_commands(commands)
    return parser


Transports = tuple[Transport | None, EmbeddingTransport | None]


def _run(args: argparse.Namespace, argv: Sequence[str], settings: KafulSettings,
         transports: Transports | None = None) -> int:
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
        strict=args.strict,
    )
    selected = select_phases(request.from_phase, request.only)
    if args.no_llm_cache:
        settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"cache_reads": False})})
    llm, embeddings = transports if transports is not None else (
        AnthropicTransport(settings.llm), OpenAIEmbeddingTransport(settings.llm))
    ctx, base = start_run(settings, request, transport=llm, embedding_transport=embeddings)
    manifest = execute(ctx, selected, base)
    sys.stdout.write(f"{manifest.status}: {ctx.run_dir}\n")
    if manifest.status == "passed":
        return EXIT_PASSED
    if manifest.status == "not_implemented":
        return EXIT_NOT_IMPLEMENTED
    return EXIT_FAILED


def main(argv: Sequence[str] | None = None, settings: KafulSettings | None = None,
         transports: Transports | None = None) -> int:
    """Entry point for the `kaful` command and `python -m kaful`. Tests pass `transports`
    (LLM, embeddings) instead of the real API clients."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    settings = settings if settings is not None else KafulSettings()
    configure_logging(settings.log_level)
    handlers = {
        ("eval", "run"): eval_commands.eval_run,
        ("eval", "compare"): eval_commands.eval_compare,
        ("eval", "golden-review"): eval_commands.eval_golden_review,
        ("prompts", "check"): eval_commands.prompts_check,
        ("prompts", "lock"): eval_commands.prompts_lock,
        ("ingest", "review"): ingest_commands.ingest_review,
        ("ingest", "classifier-eval"): ingest_commands.ingest_classifier_eval,
        ("ingest", "grounding"): ingest_commands.ingest_grounding,
        ("understand", "golden-review"): understand_commands.understand_golden_review,
        ("understand", "score"): understand_commands.understand_score,
        ("understand", "review"): understand_commands.understand_review,
        ("understand", "review-apply"): understand_commands.understand_review_apply,
        ("scope", "score"): scope_commands.scope_score,
    }
    sub = str(getattr(args, "eval_command", None) or getattr(args, "prompts_command", None)
              or getattr(args, "ingest_command", None) or getattr(args, "understand_command", None)
              or getattr(args, "scope_command", None))
    try:
        if args.command != "run":
            return handlers[(args.command, sub)](args, settings)
        return _run(args, args_list, settings, transports)
    except (MachineConfigError, PhaseSelectionError, BaseRunError, FileNotFoundError, ValueError,
            OfflineMissError, BudgetExceeded) as exc:
        log.error("command failed", extra={"fields": {"error": str(exc)}})
        sys.stderr.write(f"kaful: error: {exc}\n")
        return EXIT_FAILED
