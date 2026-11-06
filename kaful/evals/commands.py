"""`kaful eval run|compare|golden-review` and `kaful prompts check|lock`.

Paths (repo-relative, committed unless noted):
    evals/recorded/<prompt_id>/v<N>/<prompt_sha256>.json   recorded responses (the eval fixtures)
    evals/results/<prompt_id>/v<N>__<model>[__tag-<tag>].json  eval results (tagged: a case subset)
    outputs/eval_review/<prompt_id>__v<N>.html              golden review page (gitignored)
    outputs/evals/<run_id>/ledger.jsonl                     eval-run ledgers (gitignored)

Offline runs (the default) replay recorded responses only; a missing recording is an
error naming the prompt. `--live` sends uncached requests to the API, records them,
and needs an explicit `--budget`.
"""

import argparse
import logging
import sys
from pathlib import Path

from kaful.config import REPO_ROOT, KafulSettings, LLMSettings
from kaful.evals.compare import compare, render_markdown
from kaful.evals.golden_review import write_review
from kaful.evals.models import EvalResult, EvalSet, results_path
from kaful.evals.runner import judge_problems, load_result, run_eval, save_result
from kaful.llm.budget import RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.ledger import LEDGER_FILE, Ledger
from kaful.prompts.registry import (
    LockEntry,
    Prompt,
    PromptRegistry,
    RunScores,
    lock_problems,
    read_lock,
    write_lock,
)
from kaful.runs import code_version, dump_json, new_run_id, now_utc, write_text_atomic

EVALS_ROOT = REPO_ROOT / "evals"
RESULTS_ROOT = EVALS_ROOT / "results"
RECORDED_ROOT = EVALS_ROOT / "recorded"
EVAL_RUNS_ROOT = REPO_ROOT / "outputs" / "evals"
EVAL_BATCHES_ROOT = EVAL_RUNS_ROOT / "batches"
GOLDEN_REVIEW_DIR = REPO_ROOT / "outputs" / "eval_review"

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
    run.add_argument("--batch", action="store_true", help="send uncached requests as Message Batches (50%% price)")
    run.add_argument("--tag", default=None, help="run only the cases with this tag (the result is never lockable)")
    run.add_argument("--routed", action="store_true",
                     help="evaluate the configured routing (first pass, escalation where its rules say)")
    run.add_argument("--sample", type=int, default=0, help="repeat index k: fresh calls (sample k); locks need 2 runs")
    review = evals.add_parser("golden-review", help="write a local page to check golden cases' expected values")
    review.add_argument("prompt_id")
    review.add_argument("--tag", default="golden", help="cases to include (default: golden)")
    cmp = evals.add_parser("compare", help="compare two eval results, given as VERSION@MODEL")
    cmp.add_argument("prompt_id")
    cmp.add_argument("a")
    cmp.add_argument("b")
    prompts = commands.add_parser("prompts", help="prompt registry").add_subparsers(dest="prompts_command", required=True)
    prompts.add_parser("check", help="check prompts against versions.lock.json")
    lock = prompts.add_parser("lock", help="lock a prompt's current version (needs >= 2 passing runs of one setup)")
    lock.add_argument("prompt_id")


def _with_choice(llm: LLMSettings, prompt: Prompt, model: str | None, effort: str | None) -> LLMSettings:
    current = llm.choice(prompt.id)
    choice = current.model_copy(update={"model": model or current.model, "effort": effort or current.effort})
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
                      cache=ResponseStore(RECORDED_ROOT), run_id=run_id, batch_state_dir=EVAL_BATCHES_ROOT)
    choice = llm.choice(prompt.id)
    label = f"routed-{choice.model}+{choice.escalate.model}" if args.routed and choice.escalate else choice.model
    result = run_eval(prompt, gateway, model=label, effort=choice.effort, judge_prompt=judge,
                      judge_model=judge_model, judge_effort=llm.choice(judge.id).effort if judge else None,
                      source="live" if args.live else "offline", batch=args.batch,
                      evalset=EvalSet(prompt, settings, tag=args.tag), git_commit=code_version().git_commit, now=now_utc,
                      sample=args.sample, routed=args.routed)
    path = save_result(result, RESULTS_ROOT)
    lines = [f"{prompt.id} v{prompt.version} on {label} (run {args.sample}): {'PASSED' if result.passed else 'FAILED'}",
             *(f"  {t.metric}: {t.value if t.value is None else round(t.value, 3)} "
               f"{'ok' if t.passed else 'FAIL'} ({t.threshold.model_dump(exclude_none=True)})" for t in result.thresholds),
             f"  billed ${result.billed_cost_usd:.4f}, replayed ${result.unbilled_cost_usd:.4f}",
             f"  saved {path.relative_to(REPO_ROOT)}"]
    sys.stdout.write("\n".join(lines) + "\n")
    return EXIT_OK if result.passed else EXIT_FAILED


def eval_golden_review(args: argparse.Namespace, settings: KafulSettings) -> int:
    evalset = EvalSet(PromptRegistry().load(args.prompt_id), settings)
    path = write_review(GOLDEN_REVIEW_DIR, evalset, args.tag)
    sys.stdout.write(f"wrote {path.relative_to(REPO_ROOT)} (local only: it embeds manual pages)\n")
    return EXIT_OK


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


MIN_LOCK_RUNS = 2  # standing rule (2026-11-06): every eval case runs at least twice before a lock


def prompts_lock(args: argparse.Namespace, _settings: KafulSettings) -> int:
    """Lock the current version on the model (or route) with >= MIN_LOCK_RUNS passing runs,
    recording each run's scores. Any failing run of that model/route blocks the lock."""
    prompt = PromptRegistry().load(args.prompt_id)
    candidates = sorted((RESULTS_ROOT / prompt.id).glob(f"v{prompt.version}__*.json"))
    runs: dict[str, list[tuple[Path, EvalResult]]] = {}
    for p in candidates:
        r = load_result(p)
        if r.content_sha256 == prompt.content_sha256 and r.case_tag is None:
            runs.setdefault(r.model, []).append((p, r))
    ready = sorted(m for m, rs in runs.items()
                   if len({r.sample for _, r in rs}) >= MIN_LOCK_RUNS and all(r.passed for _, r in rs))
    if not ready:
        summary = {m: [(r.sample, r.passed) for _, r in rs] for m, rs in sorted(runs.items())}
        raise ValueError(f"{prompt.id} v{prompt.version}: no model/route with {MIN_LOCK_RUNS}+ runs, all passing, "
                         f"for the current content (runs: {summary}); repeat with --sample 1")
    chosen = sorted(runs[ready[0]], key=lambda pr: pr[1].sample)
    lock = read_lock()
    entry = LockEntry(version=prompt.version, content_sha256=prompt.content_sha256,
                      eval_results=tuple(sorted(p.relative_to(REPO_ROOT).as_posix() for p, _ in chosen)),
                      run_scores=tuple(RunScores(result=p.relative_to(REPO_ROOT).as_posix(), sample=r.sample,
                                                 metrics={m.name: round(m.value, 6) for m in r.metrics})
                                       for p, r in chosen))
    pending = {k: v for k, v in lock.pending.items() if k != prompt.id}
    write_lock(lock.model_copy(update={"prompts": {**lock.prompts, prompt.id: entry}, "pending": pending}))
    sys.stdout.write(f"locked {prompt.id} v{prompt.version} on {ready[0]} ({len(chosen)} passing runs)\n")
    return EXIT_OK
