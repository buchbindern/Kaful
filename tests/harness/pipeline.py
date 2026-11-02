"""
tests/harness/pipeline.py
-------------------------
Run the legacy pipeline in-process against a scratch machines directory, with
the LLM client and both embedding paths replaced (see fake_anthropic.py and
fake_embedder.py). The real machines/ tree is never written.

Usage:
    with offline_pipeline(machines_dir, client=ReplayAnthropic(store)):
        run_legacy("tiny_pump")
"""

import shutil
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import config
import utils.llm
from tests.harness.fake_embedder import HashingEmbeddingFunction, frozen_progpy_store
from utils import ledger

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_MACHINES_DIR = FIXTURES_DIR / "machines"
LLM_FIXTURES_DIR = FIXTURES_DIR / "llm"


def copy_machine(src: Path, machines_dir: Path, ignore: tuple[str, ...] = ("__pycache__",)) -> Path:
    """Copy one machine directory into a scratch machines dir; returns the copy."""
    dest = machines_dir / src.name
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*ignore))
    return dest


@contextmanager
def offline_pipeline(machines_dir: Path, client: Any) -> Iterator[None]:
    """Point config at `machines_dir` and swap in the fake client and embedders."""
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(config, "MACHINES_DIR", machines_dir))
        stack.enter_context(mock.patch("anthropic.Anthropic", client))
        stack.enter_context(mock.patch.object(utils.llm, "_client", None))
        stack.enter_context(mock.patch("utils.rag.OpenAIEmbeddingFunction",
                                       HashingEmbeddingFunction))
        stack.enter_context(mock.patch("utils.rag.time.sleep", lambda _s: None))
        stack.enter_context(frozen_progpy_store())
        try:
            yield
        finally:
            ledger.end_run()


@contextmanager
def fast_phase7(ensemble: int, mc_samples: int, horizon: int) -> Iterator[None]:
    """Shrink phase 7's module-level sizes (bound at import with `from config import`)."""
    import phases.phase7_estimate.step_b_estimate as step_b
    import phases.phase7_estimate.step_c_rul as step_c
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(step_b, "ENSEMBLE_SIZE", ensemble))
        stack.enter_context(mock.patch.object(step_c, "MC_SAMPLES", mc_samples))
        stack.enter_context(mock.patch.object(step_c, "PREDICTION_HORIZON", horizon))
        yield


def run_legacy(machine_id: str, *args: str) -> None:
    """Call run.main() in-process with the given CLI arguments."""
    import run
    with mock.patch.object(sys, "argv", ["run.py", machine_id, *args]):
        run.main()
