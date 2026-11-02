"""
scripts/record_fixtures.py
--------------------------
PAID: record real Anthropic responses as offline fixtures (M2). Needs approval.

Runs the legacy pipeline on a fixture machine copied into --work-dir, with the
same fake embedders as the offline tests (so recorded prompts match replay),
and saves every response verbatim to tests/fixtures/llm/<machine>/.

Before each call the ledger total plus that call's worst-case cost is checked
against --budget; the run stops cleanly if it would go over. Phases can be
recorded one at a time; the work dir (and its ledger) carries over.

Usage:
    python -B scripts/record_fixtures.py tiny_pump --work-dir <dir> --budget 1.0 --only phase1
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from tests.harness.fake_anthropic import RecordingAnthropic, ledger_spent_usd  # noqa: E402
from tests.harness.fixtures import FixtureStore  # noqa: E402
from tests.harness.pipeline import (  # noqa: E402
    FIXTURE_MACHINES_DIR, LLM_FIXTURES_DIR, copy_machine, offline_pipeline, run_legacy)

# Placeholders keep the real OpenAI key out: no embedding call may leave the process.
OFFLINE_ENV = {"OPENAI_API_KEY": "recording-no-openai", "ANONYMIZED_TELEMETRY": "False"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("machine_id")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--budget", type=float, required=True, help="hard cap in USD for the work dir")
    p.add_argument("--only", required=True, help="phase to record, e.g. phase2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    machines_dir = args.work_dir.resolve() / "machines"
    if not (machines_dir / args.machine_id).exists():
        machines_dir.mkdir(parents=True, exist_ok=True)
        copy_machine(FIXTURE_MACHINES_DIR / args.machine_id, machines_dir)
    ledger_path = machines_dir / args.machine_id / "outputs" / "ledger.jsonl"

    key = dotenv_values(ROOT / ".env").get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("ANTHROPIC_API_KEY missing from .env")
    os.environ.update(OFFLINE_ENV)
    os.environ["ANTHROPIC_API_KEY"] = key
    real = anthropic.Anthropic(api_key=key)

    store = FixtureStore(LLM_FIXTURES_DIR / args.machine_id)
    client = RecordingAnthropic(real, store, args.budget, ledger_path)
    before = ledger_spent_usd(ledger_path)
    try:
        with offline_pipeline(machines_dir, client):
            run_legacy(args.machine_id, "--only", args.only)
    finally:
        after = ledger_spent_usd(ledger_path)
        print(f"\n[record] {args.only}: ${after - before:.4f} this run, "
              f"${after:.4f} total of ${args.budget:.2f} budget")


if __name__ == "__main__":
    main()
