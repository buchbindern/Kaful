"""Run a prompt's eval set through the gateway and score it.

Every metric is a mean over cases. Always present:
- `parses`: the final reply is JSON;
- `schema_valid`: the final reply validates against the output model;
- `first_attempt_valid`: it validated without the retry.
Checks and golden expectations add their own metrics. A case whose call failed
scores 0 on every metric it would have produced, so failures count against the prompt.
"""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from kaful.evals.checks import JudgeFn, run_check, score_golden
from kaful.evals.models import (
    CaseResult,
    CheckOutcome,
    EvalCase,
    EvalResult,
    EvalSet,
    JudgeRef,
    MetricScore,
    MetricSummary,
    ThresholdOutcome,
    results_path,
)
from kaful.llm.gateway import Call, CallScope, Gateway, LLMOutputError
from kaful.llm.ledger import total_cost
from kaful.prompts.registry import Prompt, PromptLock
from kaful.runs import dump_json, write_text_atomic

EVAL_PHASE = "evals"
JUDGE_VERDICT_FIELD = "verdict"


def make_judge(gateway: Gateway, judge_prompt: Prompt, case_id: str) -> JudgeFn:
    """A judge callable for one case; each verdict is a gateway call (ledgered, cached)."""

    def judge(claim: str, source: str) -> str:
        scope = CallScope(EVAL_PHASE, judge_prompt.id, component=case_id)
        result = gateway.call(Call(judge_prompt, {"claim": claim, "source": source}, scope))
        return str(getattr(result.output, JUDGE_VERDICT_FIELD))

    return judge


def case_call(prompt: Prompt, evalset: EvalSet, case: EvalCase) -> Call:
    return Call(prompt, evalset.inputs(case), CallScope(EVAL_PHASE, prompt.id, component=case.id),
                images=evalset.images(case))


def prefetch_batch(prompt: Prompt, evalset: EvalSet, gateway: Gateway) -> None:
    """Send every case's uncached request as one Message Batch (50% price); scoring then
    reads the responses from the cache. Invalid replies are scored later, not raised here."""
    try:
        gateway.run_many([case_call(prompt, evalset, c) for c in evalset.cases], mode="batch")
    except LLMOutputError:
        pass


def run_case(case: EvalCase, prompt: Prompt, evalset: EvalSet, gateway: Gateway,
             judge_prompt: Prompt | None) -> CaseResult:
    inputs = evalset.inputs(case)
    try:
        result = gateway.call(case_call(prompt, evalset, case))
    except LLMOutputError as exc:
        return _failed_case(case, evalset, exc)
    output = result.output.model_dump(mode="json")
    scores = {"parses": 1.0, "schema_valid": 1.0, "first_attempt_valid": float(result.attempts == 1)}
    judge = make_judge(gateway, judge_prompt, case.id) if judge_prompt is not None else None
    outcomes = []
    for spec in evalset.spec.checks:
        outcome, check_scores = run_check(spec, output, inputs, judge)
        outcomes.append(outcome)
        scores.update(check_scores)
    scores.update({g.metric: score_golden(g, output) for g in case.golden})
    return CaseResult(case_id=case.id, attempts=result.attempts, scores=_sorted_scores(scores),
                      checks=tuple(sorted(outcomes, key=lambda o: o.check)))


def _failed_case(case: EvalCase, evalset: EvalSet, exc: LLMOutputError) -> CaseResult:
    try:
        json.loads(exc.last_text or "")
        parses = 1.0
    except ValueError:
        parses = 0.0
    scores = {"parses": parses, "schema_valid": 0.0, "first_attempt_valid": 0.0}
    for spec in evalset.spec.checks:
        names = [f"{spec.name}.verbatim", f"{spec.name}.supported"] if spec.kind == "quote_in_source" else [spec.name]
        scores.update(dict.fromkeys(names if spec.kind != "unit_parses" else [], 0.0))
    scores.update({g.metric: 0.0 for g in case.golden})
    checks = tuple(CheckOutcome(check=s.name, status="fail", details=("no valid output",))
                   for s in evalset.spec.checks)
    return CaseResult(case_id=case.id, error=str(exc), scores=_sorted_scores(scores), checks=checks)


def _sorted_scores(scores: dict[str, float]) -> tuple[MetricScore, ...]:
    return tuple(MetricScore(name=k, value=v) for k, v in sorted(scores.items()))


def summarize(cases: tuple[CaseResult, ...]) -> tuple[MetricSummary, ...]:
    by_metric: dict[str, list[float]] = {}
    for case in cases:
        for score in case.scores:
            by_metric.setdefault(score.name, []).append(score.value)
    return tuple(MetricSummary(name=k, value=sum(v) / len(v), n_cases=len(v)) for k, v in sorted(by_metric.items()))


def run_eval(
    prompt: Prompt, gateway: Gateway, *, model: str, effort: str | None, judge_prompt: Prompt | None,
    judge_model: str | None, source: Literal["offline", "live"], git_commit: str, now: Callable[[], datetime],
    judge_effort: str | None = None, batch: bool = False, evalset: EvalSet | None = None,
) -> EvalResult:
    """Run every case (first as one batch when `batch`), then score the prompt against its thresholds."""
    evalset = evalset or EvalSet(prompt)
    if evalset.spec.judge is not None and judge_prompt is None:
        raise ValueError(f"{prompt.id}: eval set uses judge {evalset.spec.judge.prompt!r}; none given")
    if batch:
        prefetch_batch(prompt, evalset, gateway)
    cases = tuple(run_case(c, prompt, evalset, gateway, judge_prompt) for c in evalset.cases)
    metrics = summarize(cases)
    values = {m.name: m.value for m in metrics}
    thresholds = tuple(
        ThresholdOutcome(metric=name, threshold=t, value=values.get(name),
                         passed=values.get(name) is not None and t.passes(values[name]))
        for name, t in sorted(prompt.spec.thresholds.items())
    )
    rows = [r for r in gateway.ledger.rows() if r.event == "llm_call"]
    judge = None
    if judge_prompt is not None:
        judge = JudgeRef(prompt_id=judge_prompt.id, version=judge_prompt.version,
                         content_sha256=judge_prompt.content_sha256, model=judge_model or "",
                         effort=judge_effort)
    return EvalResult(
        prompt_id=prompt.id, version=prompt.version, content_sha256=prompt.content_sha256, model=model,
        effort=effort, judge=judge, source=source, cases=cases, metrics=metrics, thresholds=thresholds,
        passed=all(t.passed for t in thresholds), billed_cost_usd=total_cost(rows),
        unbilled_cost_usd=sum(r.unbilled_cost_usd or 0.0 for r in rows),
        input_tokens=sum(r.input_tokens + r.cache_creation_input_tokens + r.cache_read_input_tokens for r in rows),
        output_tokens=sum(r.output_tokens for r in rows), git_commit=git_commit, created_at=now(),
    )


def save_result(result: EvalResult, results_root: Path) -> Path:
    path = results_path(results_root, result.prompt_id, result.version, result.model)
    write_text_atomic(path, dump_json(result))
    return path


def load_result(path: Path) -> EvalResult:
    return EvalResult.model_validate_json(path.read_bytes())


def judge_problems(judge_prompt: Prompt, judge_model: str, lock: PromptLock, results_root: Path) -> list[str]:
    """Why this judge may not be used yet (empty when its validation passed and is locked)."""
    entry = lock.prompts.get(judge_prompt.id)
    if entry is None or (entry.version, entry.content_sha256) != (judge_prompt.version, judge_prompt.content_sha256):
        return [f"judge {judge_prompt.id} v{judge_prompt.version} is not locked"]
    path = results_path(results_root, judge_prompt.id, judge_prompt.version, judge_model)
    if not path.is_file():
        return [f"judge {judge_prompt.id} has no validation result for {judge_model}"]
    result = load_result(path)
    if result.content_sha256 != judge_prompt.content_sha256 or not result.passed:
        return [f"judge {judge_prompt.id} validation on {judge_model} did not pass for this content"]
    return []
