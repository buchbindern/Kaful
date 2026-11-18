"""One ledger schema: kaful's LedgerRow reads the legacy (v1) rows and validates v2 rows."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from kaful.llm.ledger import Ledger, LedgerRow, cost_by_phase, read_ledger, total_cost


# Rows the legacy utils/ledger.py wrote (one LLM call, one embedding, one checkpoint hit), captured before M15
# deleted legacy code (tag legacy-final), so old machine ledgers stay readable.
LEGACY_V1_ROWS = Path(__file__).resolve().parents[1] / "fixtures" / "ledger" / "legacy_v1_rows.jsonl"


@pytest.fixture
def legacy_rows() -> list[LedgerRow]:
    return read_ledger(LEGACY_V1_ROWS)


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


def test_rows_without_a_machine_round_trip(tmp_path: Path) -> None:
    """Eval runs have no machine; None fields are omitted on write and must read back."""
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(_v2(machine_id=None))
    assert ledger.rows()[0].machine_id is None
