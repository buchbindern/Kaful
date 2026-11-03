"""`kaful eval run|compare` and `kaful prompts check|lock`.

Paths (repo-relative, committed unless noted):
    evals/recorded/<prompt_id>/v<N>/<prompt_sha256>.json   recorded responses (the eval fixtures)
    evals/results/<prompt_id>/v<N>__<model>.json           eval results
    outputs/evals/<run_id>/ledger.jsonl                     eval-run ledgers (gitignored)

Offline runs (the default) replay recorded responses only; a missing recording is an
error naming the prompt. `--live` sends uncached requests to the API, records them,
and needs an explicit `--budget`.
"""

import argparse
import logging
import sys
from pathlib import Path

from kaful.config import REPO_ROOT, KafulSettings, LLMSettings, ModelChoice
from kaful.evals.compare import compare, render_markdown
from kaful.evals.models import EvalSet, results_path
from kaful.evals.runner import judge_problems, load_result, run_eval, save_result
from kaful.llm.budget import RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.ledger import LEDGER_FILE, Ledger
from kaful.prompts.registry import LockEntry, Prompt, PromptRegistry, lock_problems, read_lock, write_lock
from kaful.runs import code_version, dump_json, new_run_id, now_utc, write_text_atomic

EVALS_ROOT = REPO_ROOT / "evals"
RESULTS_ROOT = EVALS_ROOT / "results"
RECORDED_ROOT = EVALS_ROOT / "recorded"
EVAL_RUNS_ROOT = REPO_ROOT / "outputs" / "evals"

EXIT_OK = 0
EXIT_FAILED = 1

log = logging.getLogger("kaful.evals")


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    evals = commands.add_parser("eval", help="run or compare prompt evals").add_subparsers(dest="eval_command", required=True)
    run = evals.add_parser("run", help="run a prompt's eval set")
    run.add_argument("prompt_id")
    run.add_argument("--live", action="store_true", help="call the API for unrecorded requests (paid)")
    run.add_argument("--model", default=None, help="override the configured model for this prompt")
    run.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--budget", type=float, default=None, help="USD cap for this eval run (required with --live)")
    cmp = evals.add_parser("compare", help="compare two eval results, given as VERSION@MODEL")
    cmp.add_argument("prompt_id")
    cmp.add_argument("a")
    cmp.add_argument("b")
    prompts = commands.add_parser("prompts", help="prompt registry").add_subparsers(dest="prompts_command", required=True)
    prompts.add_parser("check", help="check prompts against versions.lock.json")
    lock = prompts.add_parser("lock", help="lock a prompt's current version (needs a passing eval result)")
    lock.add_argument("prompt_id")


def _with_choice(llm: LLMSettings, prompt: Prompt, model: str | None, effort: str | None) -> LLMSettings:
    current = llm.choice(prompt.id)
    choice = ModelChoice.model_validate({"model": model or current.model, "effort": effort or current.effort})
    return llm.model_copy(update={"prompts": {**llm.prompts, prompt.id: choice}})


def eval_run(args: argparse.Namespace, settings: KafulSettings) -> int:
    registry = PromptRegistry()
    prompt = registry.load(args.prompt_id)
    if args.live and (args.budget is None or args.budget <= 0):
        raise ValueError("--live needs an explicit --budget > 0")
    llm = _with_choice(settings.llm, prompt, args.model, args.effort)
    judge_spec = EvalSet(prompt).spec.judge
    judge = registry.load(judge_spec.prompt) if judge_spec else None
    judge_model = llm.choice(judge.id).model if judge else None
    if judge is not None and judge_model is not None:
        problems = judge_problems(judge, judge_model, read_lock(), RESULTS_ROOT)
        if problems:
            raise ValueError("; ".join(problems) + f" (run `kaful eval run {judge.id}` and lock it first)")
    transport = None
    if args.live:
        from kaful.llm.anthropic_api import AnthropicTransport

        transport = AnthropicTransport(llm)
    run_id = new_run_id(now_utc())
    gateway = Gateway(settings=llm, ledger=Ledger(EVAL_RUNS_ROOT / run_id / LEDGER_FILE),
                      budget=RunBudget(args.budget or settings.default_budget_usd), transport=transport,
                      cache=ResponseStore(RECORDED_ROOT), run_id=run_id)
    choice = llm.choice(prompt.id)
    result = run_eval(prompt, gateway, model=choice.model, effort=choice.effort, judge_prompt=judge,
                      judge_model=judge_model, source="live" if args.live else "offline",
                      git_commit=code_version().git_commit, now=now_utc)
    path = save_result(result, RESULTS_ROOT)
    lines = [f"{prompt.id} v{prompt.version} on {choice.model}: {'PASSED' if result.passed else 'FAILED'}",
             *(f"  {t.metric}: {t.value if t.value is None else round(t.value, 3)} "
               f"{'ok' if t.passed else 'FAIL'} ({t.threshold.model_dump(exclude_none=True)})" for t in result.thresholds),
             f"  billed ${result.billed_cost_usd:.4f}, replayed ${result.unbilled_cost_usd:.4f}",
             f"  saved {path.relative_to(REPO_ROOT)}"]
    sys.stdout.write("\n".join(lines) + "\n")
    return EXIT_OK if result.passed else EXIT_FAILED


def _result_for(prompt_id: str, spec: str, settings: KafulSettings) -> Path:
    version, _, model = spec.partition("@")
    model = model or settings.llm.choice(prompt_id).model
    return results_path(RESULTS_ROOT, prompt_id, int(version.lstrip("v")), model)


def eval_compare(args: argparse.Namespace, settings: KafulSettings) -> int:
    a = load_result(_result_for(args.prompt_id, args.a, settings))
    b = load_result(_result_for(args.prompt_id, args.b, settings))
    comparison = compare(a, b)
    name = f"compare__v{a.version}__{a.model}__vs__v{b.version}__{b.model}.json"
    write_text_atomic(RESULTS_ROOT / args.prompt_id / name, dump_json(comparison))
    sys.stdout.write(render_markdown(comparison))
    return EXIT_OK


def prompts_check(_args: argparse.Namespace, _settings: KafulSettings) -> int:
    problems = lock_problems(PromptRegistry(), read_lock())
    sys.stdout.write("\n".join(problems or ["all prompts match versions.lock.json"]) + "\n")
    return EXIT_FAILED if problems else EXIT_OK


def prompts_lock(args: argparse.Namespace, _settings: KafulSettings) -> int:
    prompt = PromptRegistry().load(args.prompt_id)
    candidates = sorted((RESULTS_ROOT / prompt.id).glob(f"v{prompt.version}__*.json"))
    passing = [p for p in candidates
               if (r := load_result(p)).passed and r.content_sha256 == prompt.content_sha256]
    if not passing:
        raise ValueError(f"{prompt.id} v{prompt.version}: no passing eval result for the current content")
    lock = read_lock()
    entry = LockEntry(version=prompt.version, content_sha256=prompt.content_sha256,
                      eval_results=tuple(sorted(p.relative_to(REPO_ROOT).as_posix() for p in passing)))
    write_lock(lock.model_copy(update={"prompts": {**lock.prompts, prompt.id: entry}}))
    sys.stdout.write(f"locked {prompt.id} v{prompt.version} ({len(passing)} passing result(s))\n")
    return EXIT_OK
