"""`pytest -m evals`: re-score every locked prompt version from its recorded outputs, offline.

For each eval result the lock relies on, the eval set is replayed from evals/recorded/
and must reproduce the committed scores and still pass its thresholds.
"""

from pathlib import Path

import pytest

from kaful.config import REPO_ROOT, LLMSettings, ModelChoice
from kaful.evals.commands import RECORDED_ROOT
from kaful.evals.models import EvalSet
from kaful.evals.runner import choice_for, load_result, run_eval
from kaful.llm.budget import RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.ledger import Ledger
from kaful.prompts.registry import PromptRegistry, read_lock

LOCKED = sorted((pid, path) for pid, entry in read_lock().prompts.items() for path in entry.eval_results)


@pytest.mark.evals
@pytest.mark.parametrize(("prompt_id", "result_path"), LOCKED, ids=[p for _, p in LOCKED])
def test_locked_result_reproduces_offline(tmp_path: Path, prompt_id: str, result_path: str) -> None:
    committed = load_result(REPO_ROOT / result_path)
    registry = PromptRegistry()
    prompt = registry.load(prompt_id)
    missing = [c.id for c in EvalSet(prompt).cases if c.page is not None and not EvalSet(prompt).page_available(c)]
    if missing:
        pytest.skip(f"{prompt_id}: source documents not on disk for {missing} (manuals are not committed)")
    assert committed.content_sha256 == prompt.content_sha256
    judge = registry.load(committed.judge.prompt_id) if committed.judge else None
    choices = {prompt_id: choice_for(committed)}
    if committed.judge and judge:
        assert EvalSet(prompt).spec.judge is not None
        choices[judge.id] = ModelChoice(model=committed.judge.model, effort=committed.judge.effort)  # type: ignore[arg-type]
    gateway = Gateway(settings=LLMSettings(prompts=choices), ledger=Ledger(tmp_path / "ledger.jsonl"),
                      budget=RunBudget(0.01), transport=None, cache=ResponseStore(RECORDED_ROOT))
    replay = run_eval(prompt, gateway, model=committed.model, effort=committed.effort, judge_prompt=judge,
                      judge_model=committed.judge.model if committed.judge else None,
                      judge_effort=committed.judge.effort if committed.judge else None, source="offline",
                      git_commit="replay", now=lambda: committed.created_at, sample=committed.sample,
                      routed=committed.routing is not None)
    assert replay.passed
    assert replay.metrics == committed.metrics and replay.cases == committed.cases
    assert replay.billed_cost_usd == 0
