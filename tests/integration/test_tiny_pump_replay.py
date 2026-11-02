"""
tiny_pump replay (M2): the legacy pipeline's phases 1, 2, 4 and 5 run offline
from recorded fixtures, with every recorded response used exactly once.

Phase 3 is skipped: 3a orders topics by set iteration, so the 3b/3c prompts
change with PYTHONHASHSEED and their fixtures cannot replay reliably (legacy
bug, logged under M7). Phase 4 needs only phase 2's outputs; phase 5 needs
phases 2 and 4. Phases 6–7 were not recorded (legacy frozen; see PLAN M2).
"""

import json
from collections import Counter
from pathlib import Path

from tests.harness.fake_anthropic import ReplayAnthropic
from tests.harness.fixtures import FixtureStore
from tests.harness.pipeline import (
    FIXTURE_MACHINES_DIR, LLM_FIXTURES_DIR, copy_machine, offline_pipeline, run_legacy)

MACHINE_ID = "tiny_pump"
REPLAYED_PHASES = ("phase1", "phase2", "phase4", "phase5")
EXPECTED_CALLS = {"phase1/ingest": 3, "phase2/step_b_context": 2, "phase2/step_c_extract": 5,
                  "phase4/step_a_comprehend": 1, "phase4/step_b_physics": 11,
                  "phase5/step_a_assign": 1, "phase5/step_b_triage": 1}


def test_tiny_pump_replays_phases_1_2_4_5_offline(tmp_path: Path) -> None:
    machines_dir = tmp_path / "machines"
    machines_dir.mkdir()
    machine = copy_machine(FIXTURE_MACHINES_DIR / MACHINE_ID, machines_dir)
    store = FixtureStore(LLM_FIXTURES_DIR / MACHINE_ID)
    client = ReplayAnthropic(store)

    with offline_pipeline(machines_dir, client):
        for phase in REPLAYED_PHASES:
            run_legacy(MACHINE_ID, "--only", phase)

    assert Counter(f"{c['phase']}/{c['step']}" for c in client.calls) == EXPECTED_CALLS
    unused = {r["step"] for h, r in store.records.items()
              if r["phase"] != "phase3" and store.used[h] != len(r["responses"])}
    assert not unused
    outputs = machine / "outputs"
    assert len(json.loads((outputs / "schema" / "final_schema.json").read_text())) == 30
    assert len(json.loads((outputs / "triage" / "step_c_full_components.json").read_text())) == 11
    # Replayed LLM calls cost $0. Embedding rows still carry the legacy ledger's
    # estimated OpenAI cost, because utils.rag logs them without knowing they were faked.
    rows = [json.loads(line) for line in (outputs / "ledger.jsonl").read_text().splitlines()]
    assert sum(r["cost_usd"] or 0 for r in rows if r["event"] in ("llm_call", "pdf_extract")) == 0
