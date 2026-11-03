"""M5 acceptance, offline: the sample prompt runs through the gateway into the ledger and is
scored by its evals (checks, golden examples, judge), replayed at $0, and compared.

Replies are scripted; nothing touches the network.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from kaful.config import LLMSettings, ModelChoice
from kaful.evals.compare import compare, render_markdown
from kaful.evals.runner import run_eval
from kaful.llm.ledger import Ledger, total_cost
from kaful.prompts.registry import PromptRegistry
from tests.harness.kaful_llm import FIXED_NOW, ScriptedTransport, last_user_text, make_gateway

SAMPLE, JUDGE = "sample.list_sensor_channels", "evals.judge_faithful_quote"
TP200 = [("inlet_pressure", "bar", 10, "inlet_pressure (bar), range 0 to 10"),
         ("discharge_pressure", "bar", 10, "discharge_pressure (bar), range 0 to 10"),
         ("filter_dp", "bar", 2, "filter_dp (bar), differential pressure across F1, range 0 to 2"),
         ("flow_rate", "m3/h", 12, "flow_rate (m3/h), range 0 to 12"),
         ("motor_current", "A", 10, "motor_current (A), range 0 to 10"),
         ("motor_temperature", "degC", 120, "motor_temperature (degC), range 0 to 120"),
         ("vibration_rms", "mm/s", 20, "vibration_rms (mm/s), measured on the pump bearing housing, range 0 to 20")]
TC50 = [("supply_temp", "degC", -10, 30, "supply_temp (degC), chilled water leaving the evaporator, range -10 to 30"),
        ("return_temp", "degC", -10, 40, "return_temp (degC), chilled water returning from the process, range -10 to 40"),
        ("condenser_pressure", "bar", 0, 30, "condenser_pressure (bar), range 0 to 30"),
        ("compressor_current", "A", 0, 60, "compressor_current (A), range 0 to 60")]


def _channels(rows: list[tuple[str, str, float, float, str]], page: int = 1) -> str:
    return json.dumps({"channels": [{"name": n, "unit": u, "range_min": lo, "range_max": hi, "evidence": e, "page": page}
                                    for n, u, lo, hi, e in rows]})


GOOD_TP200 = _channels([(n, u, 0, hi, e) for n, u, hi, e in TP200])
GOOD_TC50 = _channels(TC50)
DUPLICATE_TP200 = _channels([(n, u, 0, hi, e) for n, u, hi, e in TP200[:2] * 2])
FLAWED_TP200 = _channels(
    [(n, u, 0, hi, e) for n, u, hi, e in TP200[:4]]
    + [("motor_current", "A", 0, 15, "motor_current (A), range 0 to 15")]   # fabricated quote and range
    + [("flow_rate_l", "l/min", 0, 200, "flow_rate (m3/h), range 0 to 12")], page=1)


def _responder(flawed: bool) -> Any:
    def respond(request: dict[str, Any]) -> str:
        system = request["system"][0]["text"]
        if system.startswith("You check whether a claim"):
            return json.dumps({"rationale": "The source gives 0 to 10 A.", "verdict": "unsupported"})
        pages = request["messages"][0]["content"][0]["text"]
        if "TC-50" in pages:
            return GOOD_TC50
        if flawed:
            return FLAWED_TP200
        is_retry = "did not validate" in last_user_text(request)
        return GOOD_TP200 if is_retry else DUPLICATE_TP200
    return respond


@pytest.fixture(scope="module")
def prompts() -> Any:
    registry = PromptRegistry()
    return registry.load(SAMPLE), registry.load(JUDGE)


def _settings() -> LLMSettings:
    return LLMSettings(token_estimate="chars", prompts={
        SAMPLE: ModelChoice(model="claude-haiku-5-5", effort="low"),
        JUDGE: ModelChoice(model="claude-sonnet-5-5", effort="low"),
    })


def _eval(tmp: Path, prompts: Any, transport: Any, ledger: str = "first") -> Any:
    sample, judge = prompts
    gateway = make_gateway(tmp, transport, settings=_settings())
    gateway.ledger = Ledger(tmp / ledger / "ledger.jsonl")  # one ledger per eval run, as the CLI does
    result = run_eval(sample, gateway, model="claude-haiku-5-5", effort="low", judge_prompt=judge,
                      judge_model="claude-sonnet-5-5", source="offline", git_commit="test", now=lambda: FIXED_NOW)
    return result, gateway


def test_sample_through_gateway_ledger_and_evals(tmp_path: Path, prompts: Any) -> None:
    result, gateway = _eval(tmp_path, prompts, ScriptedTransport(_responder(flawed=False)))
    assert result.passed, [t for t in result.thresholds if not t.passed]
    tp200 = next(c for c in result.cases if c.case_id == "tp200_manual")
    assert tp200.attempts == 2 and tp200.score("first_attempt_valid") == 0.0
    assert result.metric("channel_names_f1") == 1.0 and result.metric("evidence.verbatim") == 1.0
    assert next(c for c in tp200.checks if c.check == "unit_parses").status == "skip"

    rows = gateway.ledger.rows()
    assert [(r.prompt_id, r.prompt_version, r.component, r.validation) for r in rows] == [
        (SAMPLE, 1, "tc50_quickref", "valid"),
        (SAMPLE, 1, "tp200_manual", "invalid"),
        (SAMPLE, 1, "tp200_manual", "valid"),
    ]  # every quote was found in the source, so the judge was never asked
    assert all(r.model == "claude-haiku-5-5" and r.response_cache == "miss" for r in rows)
    assert result.billed_cost_usd > 0 and result.billed_cost_usd == pytest.approx(total_cost(rows))

    # Offline replay from the cache: same scores, nothing billed.
    replay, replay_gateway = _eval(tmp_path, prompts, None, ledger="replay")
    assert replay.metrics == result.metrics and replay.billed_cost_usd == 0
    assert replay.unbilled_cost_usd == pytest.approx(result.billed_cost_usd)
    assert {r.response_cache for r in replay_gateway.ledger.rows()} == {"hit"}


def test_flawed_outputs_are_caught_and_compared(tmp_path: Path, prompts: Any) -> None:
    good, _ = _eval(tmp_path / "good", prompts, ScriptedTransport(_responder(flawed=False)))
    flawed, gateway = _eval(tmp_path / "flawed", prompts, ScriptedTransport(_responder(flawed=True)))
    assert not flawed.passed
    failing = sorted(t.metric for t in flawed.thresholds if not t.passed)
    assert failing == ["channel_names_f1", "evidence.supported", "pages_correct", "range_max_correct",
                       "range_min_correct", "units_correct"]
    judged = [r for r in gateway.ledger.rows() if r.prompt_id == JUDGE]
    assert len(judged) == 1 and judged[0].model == "claude-sonnet-5-5"  # only the fabricated quote
    tp200 = next(c for c in flawed.cases if c.case_id == "tp200_manual")
    evidence = next(c for c in tp200.checks if c.check == "evidence")
    assert evidence.details == ("not in source (judge: unsupported): 'motor_current (A), range 0 to 15'",)

    cmp = compare(good, flawed)
    deltas = {m.name: m.delta for m in cmp.metrics}
    assert deltas["channel_names_f1"] is not None and deltas["channel_names_f1"] < 0
    assert {(c.case_id, c.metric) for c in cmp.case_changes} >= {("tp200_manual", "units_correct")}
    assert all(c.case_id == "tp200_manual" for c in cmp.case_changes)  # tc50 unchanged
    assert "| channel_names_f1 |" in render_markdown(cmp)
