"""Hard rule 14 for M5: rendered requests, their hashes, and eval results do not depend on
PYTHONHASHSEED. The script replays the committed sample eval offline and prints the request
hashes and the canonical result JSON."""

import os
import subprocess
import sys

from kaful.config import REPO_ROOT

SEEDS = ("0", "1", "12345")
SCRIPT = """
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from kaful.config import KafulSettings
from kaful.evals.commands import RECORDED_ROOT
from kaful.evals.models import EvalSet
from kaful.evals.runner import run_eval
from kaful.llm.budget import RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.hashing import prompt_hash
from kaful.llm.ledger import Ledger
from kaful.prompts.registry import PromptRegistry
from kaful.runs import dump_json

registry = PromptRegistry()
prompt, judge = registry.load("sample.list_sensor_channels"), registry.load("evals.judge_faithful_quote")
llm = KafulSettings().llm
with tempfile.TemporaryDirectory() as tmp:
    gateway = Gateway(settings=llm, ledger=Ledger(Path(tmp) / "l.jsonl"), budget=RunBudget(0.01),
                      transport=None, cache=ResponseStore(RECORDED_ROOT))
    evalset = EvalSet(prompt)
    for case in evalset.cases:
        print(prompt_hash(gateway.request_for(prompt, evalset.inputs(case))))
    result = run_eval(prompt, gateway, model=llm.choice(prompt.id).model, effort=llm.choice(prompt.id).effort,
                      judge_prompt=judge, judge_model=llm.choice(judge.id).model,
                      judge_effort=llm.choice(judge.id).effort, source="offline", git_commit="x",
                      now=lambda: datetime(2026, 11, 3, tzinfo=UTC))
    print(dump_json(result))
"""


def _run(seed: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    env["PYTHONHASHSEED"] = seed
    done = subprocess.run([sys.executable, "-B", "-c", SCRIPT], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_requests_and_eval_results_are_identical_across_hash_seeds() -> None:
    outputs = {seed: _run(seed) for seed in SEEDS}
    assert '"kind": "eval_result"' in outputs["0"] and '"passed": true' in outputs["0"]
    assert len(set(outputs.values())) == 1
