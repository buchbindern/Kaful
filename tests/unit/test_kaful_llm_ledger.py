"""One ledger schema: kaful's LedgerRow reads the legacy (v1) rows and validates v2 rows."""

import time
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaful.llm.ledger import Ledger, LedgerRow, cost_by_phase, read_ledger, total_cost
from utils import ledger as legacy_ledger


@pytest.fixture
def legacy_rows(tmp_path: Path) -> list[LedgerRow]:
    machine_dir = tmp_path / "m"
    legacy_ledger.start_run({"machine_dir": machine_dir, "machine_id": "m"}, run_id="r1")
    try:
        usage = types.SimpleNamespace(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=50,
                                      cache_read_input_tokens=10, cache_creation=None)
        request = {"model": "claude-sonnet-4-6", "max_tokens": 300, "temperature": 0.2,
                   "messages": [{"role": "user", "content": "hi"}]}
        legacy_ledger.record_llm_call(event="llm_call", request=request, prompt_sha256="a" * 64,
                                      started=time.monotonic(), status="ok", attempt=1, usage=usage,
                                      response_chars=12)
        legacy_ledger.record_embedding(model="text-embedding-3-large", texts=["abcd"] * 3,
                                       started=time.monotonic(), purpose="query")
        legacy_ledger.checkpoint_hit(machine_dir / "outputs" / "x.json")
    finally:
        legacy_ledger.end_run()
    return read_ledger(machine_dir / "outputs" / "ledger.jsonl")


def test_legacy_v1_rows_parse_with_the_same_meaning(legacy_rows: list[LedgerRow]) -> None:
    call, embedding, hit = legacy_rows
    assert [r.schema_version for r in legacy_rows] == [1, 1, 1]
    assert (call.event, call.input_tokens, call.cache_read_input_tokens, call.cache_hit) == ("llm_call", 1000, 10, True)
    assert call.response_cache is None and call.prompt_id is None
    assert embedding.tokens_estimated and embedding.n_texts == 3
    assert hit.event == "checkpoint_hit" and hit.cost_usd == 0.0
    assert total_cost(legacy_rows) == pytest.approx((call.cost_usd or 0) + (embedding.cost_usd or 0))


def _v2(**fields: object) -> LedgerRow:
    base: dict[str, object] = dict(schema_version=2, ts="2026-11-03T00:00:00Z", run_id="r", machine_id="m",
                                   event="llm_call", phase="understand", step="extract", status="ok",
                                   prompt_id="sample.x", prompt_version=1, response_cache="miss", cost_usd=0.5)
    return LedgerRow.model_validate({**base, **fields})


def test_v2_call_rows_must_name_prompt_and_cache_use() -> None:
    with pytest.raises(ValidationError, match="missing"):
        _v2(prompt_id=None)
    with pytest.raises(ValidationError):
        _v2(response_cache="maybe")


def test_append_round_trips_and_sums_by_phase(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "run" / "ledger.jsonl")
    ledger.append(_v2())
    ledger.append(_v2(phase="scope", cost_usd=0.25))
    ledger.append(_v2(response_cache="hit", cost_usd=0.0, unbilled_cost_usd=0.5))
    rows = ledger.rows()
    assert rows[0] == _v2()
    assert cost_by_phase(rows) == {"scope": 0.25, "understand": 0.5}
    assert '"cache_hit": false' in (tmp_path / "run" / "ledger.jsonl").read_text().splitlines()[0]
