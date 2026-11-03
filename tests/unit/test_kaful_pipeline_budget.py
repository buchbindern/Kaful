"""A phase that runs out of budget stops the run cleanly; a rerun replays completed calls at $0."""

from pathlib import Path

import pytest

from kaful import pipeline
from kaful.artifacts.manifest import PhaseRecord
from kaful.llm.gateway import Call, CallScope
from kaful.llm.ledger import read_ledger
from kaful.phases import PhaseName
from kaful.prompts.registry import load_prompt_dir
from kaful.runs import RunContext
from tests.harness.kaful_llm import ScriptedTransport
from tests.harness.kaful_machine import MACHINE_ID, make_machine, settings_for

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "prompts" / "toy" / "echo_items"
DOCS = ("first", "second", "third")


class ThreeCallPhase:
    name: PhaseName = "understand"

    def run(self, ctx: RunContext) -> PhaseRecord:
        assert ctx.gateway is not None
        prompt = load_prompt_dir(TOY)
        for doc in DOCS:
            ctx.gateway.call(Call(prompt, {"doc": doc, "question": "Which?"}, CallScope("understand", "list_items")))
        return PhaseRecord(phase="understand", status="passed")


def _run(tmp_path: Path, budget: float, transport: ScriptedTransport) -> RunContext:
    settings = settings_for(tmp_path)
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(
        update={"token_estimate": "chars", "default_model": "claude-sonnet-5-5"})})
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest", "understand"),
                                  budget_usd=budget)
    ctx, base = pipeline.start_run(settings, request, transport=transport)
    pipeline.execute(ctx, ("ingest", "understand"), base)
    return ctx


def test_budget_stop_then_cached_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "PHASE_IMPLEMENTATIONS", {"understand": ThreeCallPhase()})
    machine_dir = make_machine(tmp_path)
    reply = '{"items": ["x"]}'
    # Each call costs ~$0.0054 (1200 in / 300 out on Sonnet 5.5) and its worst case is ~$0.0066,
    # so a $0.015 budget allows two calls and refuses the third before sending it.
    first = _run(tmp_path, 0.015, transport := ScriptedTransport(lambda _r: reply))
    record = first.manifest.phase("understand")
    assert (first.manifest.status, record.status) == ("budget_exceeded", "budget_exceeded")
    assert len(transport.sent) == 2 and "worst case" in record.failures[0]
    rows = read_ledger(first.run_dir / "ledger.jsonl")
    assert [r.status for r in rows] == ["ok", "ok", "refused_budget"]
    assert record.cost_usd > 0 and record.cost_usd == pytest.approx(sum(r.cost_usd or 0 for r in rows))
    assert first.manifest.prompt_versions == {"toy.echo_items": "1"}

    second = _run(tmp_path, 0.05, transport2 := ScriptedTransport(lambda _r: reply))
    assert second.manifest.phase("understand").status == "passed"
    assert len(transport2.sent) == 1  # only the call that was refused
    rows2 = read_ledger(second.run_dir / "ledger.jsonl")
    assert [r.response_cache for r in rows2] == ["hit", "hit", "miss"]
    assert (machine_dir / "outputs" / "llm_cache" / "toy.echo_items" / "v1").is_dir()
