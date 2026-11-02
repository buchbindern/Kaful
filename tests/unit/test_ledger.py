"""Run ledger (M1): cost arithmetic, attribution, and instrumented call sites. Offline."""

import json
import time
import types
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

import utils.llm as llm
from utils import ledger
from utils.pricing import cost_usd, load_pricing
from utils.prompt_hash import prompt_hash
from utils.rag import ManualRAG


# ── Fakes ─────────────────────────────────────────────────────────────────────

def _usage(inp: int, out: int, cache_write: int = 0, cache_read: int = 0) -> Any:
    return types.SimpleNamespace(
        input_tokens=inp, output_tokens=out,
        cache_creation_input_tokens=cache_write, cache_read_input_tokens=cache_read,
        cache_creation=None,
    )


def _response(text: str, usage: Any) -> Any:
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)], usage=usage)


class FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(outcomes: list[Any]) -> Any:
    return types.SimpleNamespace(messages=FakeMessages(outcomes))


def _rate_limit() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.invalid/v1/messages")
    return anthropic.RateLimitError("429", response=httpx.Response(429, request=request), body=None)


def _in_phase_module(module_name: str, fn_source: str, **env: Any) -> Any:
    """Define a function whose frame has __name__ == module_name (for stack attribution)."""
    namespace: dict[str, Any] = {"__name__": module_name, **env}
    exec(fn_source, namespace)
    return namespace["target"]


@pytest.fixture
def run(tmp_path: Path) -> Path:
    machine_dir = tmp_path / "machines" / "m"
    ledger.start_run({"machine_dir": machine_dir, "machine_id": "m"}, run_id="r1")
    yield machine_dir / "outputs" / "ledger.jsonl"
    ledger.end_run()
    ledger.set_component(None)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


# ── Pricing and prompt hash ───────────────────────────────────────────────────

def test_cost_usd_uses_table_rates() -> None:
    rates = load_pricing()["models"]["claude-sonnet-4-6"]
    expected = (1000 * rates["input"] + 200 * rates["output"]) / 1_000_000
    assert cost_usd("claude-sonnet-4-6", input_tokens=1000, output_tokens=200) == pytest.approx(expected)


def test_cost_usd_applies_cache_multipliers() -> None:
    p = load_pricing()
    rate, m = p["models"]["claude-opus-4-6"]["input"], p["multipliers"]
    got = cost_usd("claude-opus-4-6", cache_write_5m_tokens=1000, cache_read_tokens=1000)
    assert got == pytest.approx(1000 * rate * (m["cache_write_5m"] + m["cache_read"]) / 1_000_000)


def test_cost_usd_unknown_model_is_none() -> None:
    assert cost_usd("no-such-model", input_tokens=10) is None


def test_prompt_hash_ignores_key_order_and_non_output_fields() -> None:
    a = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "temperature": 0.2}
    b = {"temperature": 0.2, "max_tokens": 5, "messages": [{"content": "hi", "role": "user"}],
         "model": "x", "timeout": 30}
    assert prompt_hash(a) == prompt_hash(b)
    assert prompt_hash(a) != prompt_hash({**a, "temperature": 1.0})


# ── call_claude ───────────────────────────────────────────────────────────────

def test_call_claude_records_exact_usage_and_attribution(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client([_response("hello", _usage(1200, 300))])
    monkeypatch.setattr(llm, "_client", client)
    target = _in_phase_module(
        "phases.phase2_schema.step_c_extract",
        "def target():\n    return call_claude('prompt', max_tokens=100)\n",
        call_claude=llm.call_claude,
    )

    assert target() == "hello"

    [row] = _rows(run)
    assert row["run_id"] == "r1" and row["machine_id"] == "m"
    assert (row["event"], row["phase"], row["step"]) == ("llm_call", "phase2", "step_c_extract")
    assert row["call_site"].startswith("phases.phase2_schema.step_c_extract:target:")
    assert (row["input_tokens"], row["output_tokens"], row["tokens_estimated"]) == (1200, 300, False)
    assert row["cost_usd"] == pytest.approx(cost_usd("claude-sonnet-4-6", input_tokens=1200, output_tokens=300))
    assert row["prompt_sha256"] == prompt_hash(client.messages.calls[0])
    assert row["status"] == "ok" and row["cache_hit"] is False


def test_call_claude_logs_each_retry_attempt(run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_client", _client([_rate_limit(), _response("ok", _usage(10, 2))]))

    assert llm.call_claude("p") == "ok"

    rows = _rows(run)
    assert [(r["attempt"], r["status"]) for r in rows] == [(1, "retry"), (2, "ok")]
    assert rows[0]["cost_usd"] == 0.0 and rows[0]["error"] == "RateLimitError"


def test_ledger_write_failure_does_not_break_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")
    ledger.start_run({"machine_dir": blocker, "machine_id": "m"})
    monkeypatch.setattr(llm, "_client", _client([_response("fine", _usage(1, 1))]))
    try:
        assert llm.call_claude("p") == "fine"
    finally:
        ledger.end_run()


def test_no_active_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger.end_run()
    monkeypatch.setattr(llm, "_client", _client([_response("x", _usage(1, 1))]))
    assert llm.call_claude("p") == "x"
    assert not list(tmp_path.rglob("ledger.jsonl"))


# ── Other events ──────────────────────────────────────────────────────────────

def test_embedding_record_is_marked_estimated(run: Path) -> None:
    ledger.record_embedding(model="text-embedding-3-large", texts=["a" * 400],
                            started=time.monotonic(), purpose="query")
    [row] = _rows(run)
    assert row["input_tokens"] == 100 and row["tokens_estimated"] is True
    assert row["cost_usd"] == pytest.approx(cost_usd("text-embedding-3-large", input_tokens=100))


def test_checkpoint_hit_carries_component(run: Path) -> None:
    target = _in_phase_module(
        "phases.phase6_codegen.step_b_generate",
        "def target(path):\n    ledger.set_component('pump')\n    ledger.checkpoint_hit(path)\n",
        ledger=ledger,
    )
    target(Path("/x/pump.py"))
    [row] = _rows(run)
    assert (row["event"], row["step"], row["component"], row["path"]) == (
        "checkpoint_hit", "step_b_generate", "pump", "pump.py")


def test_phase1_module_maps_to_ingest_step(run: Path) -> None:
    target = _in_phase_module("phases.phase1_ingest", "def target():\n    ledger.checkpoint_hit('c')\n",
                              ledger=ledger)
    target()
    [row] = _rows(run)
    assert (row["phase"], row["step"]) == ("phase1", "ingest")


def test_pdf_extraction_logs_rejected_then_ok_attempts(run: Path) -> None:
    rag = ManualRAG.__new__(ManualRAG)  # skip __init__: no Chroma, no API clients
    rag.model = "claude-sonnet-4-6"
    rag.client = _client([
        _response("I notice there is a discrepancy in your request.", _usage(3000, 20)),
        _response("Spec table: " + "x" * 200, _usage(3000, 400)),
    ])

    text = rag._extract_rich_content("JVBERi0=")

    assert text and text.startswith("Spec table")
    rows = _rows(run)
    assert [(r["event"], r["status"]) for r in rows] == [("pdf_extract", "rejected"), ("pdf_extract", "ok")]
    assert rows[0]["cost_usd"] > 0  # rejected attempts are still billed
    assert rows[0]["request_chars"] < 5000  # base64 payload is not counted as text
