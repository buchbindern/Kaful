"""Every locked prompt's eval recordings are on disk and committed (M10b).

`pytest -m evals` replays each locked eval result from evals/recorded/; a recording that exists only in one
working tree passes there and fails on every fresh clone (M9 left parameterize.life_judgement's untracked).
"""

import subprocess
from pathlib import Path

import pytest

from kaful.config import REPO_ROOT
from kaful.evals.commands import RECORDED_ROOT
from kaful.prompts.registry import read_lock


def _tracked() -> set[Path]:
    try:
        done = subprocess.run(["git", "ls-files", "-z", "--", str(RECORDED_ROOT)], cwd=REPO_ROOT,
                              capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return {REPO_ROOT / p for p in done.stdout.decode().split("\0") if p}


def test_locked_prompts_have_committed_eval_recordings() -> None:
    tracked = _tracked()
    problems = []
    for prompt_id, entry in sorted(read_lock().prompts.items()):
        root = RECORDED_ROOT / prompt_id
        current = root / f"v{entry.version}"
        if not current.is_dir() or not any(p.is_file() for p in current.rglob("*")):
            problems.append(f"{prompt_id} v{entry.version}: no recordings under {current.relative_to(REPO_ROOT)}")
        untracked = sorted(p for p in root.rglob("*") if p.is_file() and p not in tracked)
        problems += [f"{prompt_id}: {p.relative_to(REPO_ROOT)} is not committed" for p in untracked]
    assert not problems, "\n".join(problems)
