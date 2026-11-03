"""Gateway: request shape, response cache, budget, validation retry, ledger, batches. Offline."""

import json
from pathlib import Path
from typing import Any

import pytest

from kaful.config import LLMSettings, ModelChoice
from kaful.llm.batch import BatchPending, custom_id
from kaful.llm.budget import BudgetExceeded
from kaful.llm.gateway import Call, CallScope, LLMOutputError, OfflineMissError
from kaful.llm.hashing import prompt_hash
from kaful.llm.ledger import total_cost
from kaful.llm.pricing import TokenUsage
from kaful.llm.transport import Completion, TransportError
from kaful.prompts.registry import Prompt, load_prompt_dir
from tests.harness.kaful_llm import DEFAULT_USAGE, ScriptedTransport, last_user_text, make_gateway

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "prompts" / "toy" / "echo_items"
GOOD = '{"items": ["pump", "filter"]}'
DUPLICATE = '{"items": ["pump", "pump"]}'
SCOPE = CallScope("understand", "list_items", component="skid", artifact="machine_model")


@pytest.fixture(scope="module")
def toy() -> Prompt:
    return load_prompt_dir(TOY)


def _call(toy: Prompt, doc: str = "A pump and a filter.", sample: int = 0) -> Call:
    return Call(toy, {"doc": doc, "question": "Which items?"}, SCOPE, sample)


def test_request_shape(tmp_path: Path, toy: Prompt) -> None:
    settings = LLMSettings(token_estimate="chars", prompts={"toy.echo_items": ModelChoice(model="claude-sonnet-5-5", effort="low")})
    request = make_gateway(tmp_path, None, settings=settings).request_for(toy, _call(toy).inputs)
    assert request["model"] == "claude-sonnet-5-5" and request["max_tokens"] == 500
    assert request["system"] == [{"type": "text", "text": "You list items. Costs $5 to ask.\n",
                                  "cache_control": {"type": "ephemeral"}}]
    context, task = request["messages"][0]["content"]
    assert context["cache_control"] == {"type": "ephemeral", "ttl": "1h"} and "A pump" in context["text"]
    assert task == {"type": "text", "text": "Question: Which items?\n"}
    assert request["output_config"]["effort"] == "low"
    assert request["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert not {"temperature", "top_p", "top_k"} & request.keys()


def test_request_timeout_is_global_unless_the_prompt_overrides_it(tmp_path: Path, toy: Prompt) -> None:
    assert LLMSettings().request_timeout_s == 120.0
    for prompts, expected in (({}, 120.0), ({"toy.echo_items": ModelChoice(model="claude-haiku-5-5")}, 120.0),
                              ({"toy.echo_items": ModelChoice(model="claude-haiku-5-5", request_timeout_s=300.0)}, 300.0)):
        transport = ScriptedTransport(lambda _r: GOOD)
        settings = LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5", prompts=prompts)
        make_gateway(tmp_path / str(expected) / str(len(prompts)), transport, settings=settings).call(_call(toy))
        assert transport.timeouts == [expected]


def test_miss_then_hit_and_samples(tmp_path: Path, toy: Prompt) -> None:
    transport = ScriptedTransport(lambda _r: GOOD)
    gateway = make_gateway(tmp_path, transport)
    first = gateway.call(_call(toy))
    again = gateway.call(_call(toy))
    other_sample = gateway.call(_call(toy, sample=1))
    assert len(transport.sent) == 2  # sample 0 once, sample 1 once
    assert first.output == again.output and first.cost_usd > 0 and again.cost_usd == 0
    rows = gateway.ledger.rows()
    assert [r.response_cache for r in rows] == ["miss", "hit", "miss"]
    assert rows[1].unbilled_cost_usd == pytest.approx(rows[0].cost_usd)
    assert [r.sample for r in rows] == [0, 0, 1] and rows[0].output_artifact == "machine_model"
    assert gateway.budget.spent_usd == pytest.approx(total_cost(rows))
    assert rows[0].cost_usd == pytest.approx(gateway.pricing.cost("claude-haiku-5-5", DEFAULT_USAGE))
    stored = json.loads(gateway.cache.path(toy.id, 1, first.prompt_sha256).read_text())  # type: ignore[union-attr]
    assert len(stored["responses"]) == 2 and stored["prompt_id"] == toy.id
    assert gateway.prompt_versions == {"toy.echo_items": 1}


def test_cache_reads_can_be_bypassed(tmp_path: Path, toy: Prompt) -> None:
    make_gateway(tmp_path, ScriptedTransport(lambda _r: GOOD)).call(_call(toy))
    transport = ScriptedTransport(lambda _r: GOOD)
    bypass = make_gateway(tmp_path, transport, settings=LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5", cache_reads=False))
    bypass.call(_call(toy))
    assert len(transport.sent) == 1


def test_validation_retry_once_then_cached(tmp_path: Path, toy: Prompt) -> None:
    replies = iter([DUPLICATE, GOOD])
    transport = ScriptedTransport(lambda _r: next(replies))
    gateway = make_gateway(tmp_path, transport)
    result = gateway.call(_call(toy))
    assert result.attempts == 2 and result.output.items == ["pump", "filter"]  # type: ignore[attr-defined]
    retry = transport.sent[1]["messages"]
    assert retry[1] == {"role": "assistant", "content": DUPLICATE}
    assert "items must be unique" in retry[2]["content"]
    rows = gateway.ledger.rows()
    assert [(r.attempt, r.status, r.validation) for r in rows] == [(1, "retry", "invalid"), (2, "ok", "valid")]
    assert result.cost_usd == pytest.approx(total_cost(rows))
    # A rerun replays both attempts from the cache at $0.
    rerun = make_gateway(tmp_path, ScriptedTransport(lambda _r: pytest.fail("network")))
    assert rerun.call(_call(toy)).attempts == 2 and total_cost(rerun.ledger.rows()[2:]) == 0


def test_second_invalid_reply_fails(tmp_path: Path, toy: Prompt) -> None:
    gateway = make_gateway(tmp_path, ScriptedTransport(lambda _r: DUPLICATE))
    with pytest.raises(LLMOutputError, match="after one retry") as exc:
        gateway.call(_call(toy))
    assert exc.value.last_text == DUPLICATE


def test_budget_is_checked_before_sending(tmp_path: Path, toy: Prompt) -> None:
    transport = ScriptedTransport(lambda _r: GOOD)
    gateway = make_gateway(tmp_path, transport, budget_usd=0.0001)
    with pytest.raises(BudgetExceeded, match="worst case"):
        gateway.call(_call(toy))
    assert transport.sent == []
    (row,) = gateway.ledger.rows()
    assert row.status == "refused_budget" and row.cost_usd == 0 and (row.estimated_cost_usd or 0) > 0.0001


def test_worst_case_prices_a_full_cache_write_and_max_tokens(tmp_path: Path, toy: Prompt) -> None:
    gateway = make_gateway(tmp_path, None)
    request = gateway.request_for(toy, _call(toy).inputs)
    tokens = gateway.input_tokens(request)
    expected = gateway.pricing.cost("claude-haiku-5-5", TokenUsage(cache_write_1h_tokens=tokens, output_tokens=500))
    assert gateway.worst_case(toy, request, batch=False) == pytest.approx(expected)


def test_offline_miss_names_the_prompt(tmp_path: Path, toy: Prompt) -> None:
    with pytest.raises(OfflineMissError, match=r"toy\.echo_items v1 \(list_items\)"):
        make_gateway(tmp_path, None).call(_call(toy))


@pytest.mark.parametrize("reply", [
    Completion(text='{"items": ["pu', model="claude-haiku-5-5", stop_reason="max_tokens", usage=DEFAULT_USAGE),
    TransportError("overloaded"),
])
def test_unusable_replies_are_logged_and_not_cached(tmp_path: Path, toy: Prompt, reply: Any) -> None:
    gateway = make_gateway(tmp_path, ScriptedTransport(lambda _r: reply))
    with pytest.raises((LLMOutputError, TransportError)):
        gateway.call(_call(toy))
    assert gateway.ledger.rows()[-1].status == "error"
    assert not list((tmp_path / "llm_cache").rglob("*.json"))


# ── Batches ───────────────────────────────────────────────────────────────────

def _batch_calls(toy: Prompt) -> list[Call]:
    return [_call(toy, doc=f"Document {name}.") for name in ("alpha", "bravo", "charlie", "delta")]


def _by_doc(request: dict[str, Any]) -> str:
    doc = request["messages"][0]["content"][0]["text"]
    if "bravo" in doc and "unique" not in last_user_text(request):
        return DUPLICATE  # invalid first time; the retry round fixes it
    return json.dumps({"items": [doc.split()[-1].strip(".")]})


def test_batch_matches_by_custom_id_and_retries_once(tmp_path: Path, toy: Prompt) -> None:
    gateway0 = make_gateway(tmp_path, None)
    charlie = prompt_hash(gateway0.request_for(toy, _batch_calls(toy)[2].inputs))
    transport = ScriptedTransport(_by_doc, seed=3, batch_errors={custom_id(charlie, 0)})
    gateway = make_gateway(tmp_path, transport)
    results = gateway.run_many(_batch_calls(toy), mode="batch")
    assert [r.output.items for r in results] == [["alpha"], ["bravo"], ["charlie"], ["delta"]]  # type: ignore[attr-defined]
    assert len(transport.batches) == 2  # first round, then bravo's retry and charlie's resend
    assert sorted(c for c, _ in transport.batches["msgbatch_2"]) == sorted(
        [custom_id(charlie, 0), custom_id(results[1].prompt_sha256, 0)])
    rows = [r for r in gateway.ledger.rows() if r.event == "llm_call"]
    assert {r.batch_id for r in rows} == {"msgbatch_1", "msgbatch_2"}
    ok = [r for r in rows if r.status == "ok"]
    assert ok[0].cost_usd == pytest.approx(gateway.pricing.cost("claude-haiku-5-5", DEFAULT_USAGE, batch=True))
    assert gateway.budget.reserved_usd == 0 and not list((tmp_path / "llm_batches").glob("*.json"))
    # Everything is cached now: a rerun sends nothing.
    rerun_transport = ScriptedTransport(_by_doc)
    make_gateway(tmp_path, rerun_transport).run_many(_batch_calls(toy), mode="batch")
    assert rerun_transport.batches == {} and rerun_transport.sent == []


def test_pending_batch_is_resumed_not_resubmitted(tmp_path: Path, toy: Prompt) -> None:
    settings = LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5", batch_max_wait_s=5, batch_poll_s=1)
    transport = ScriptedTransport(_by_doc, polls_before_end=100)
    with pytest.raises(BatchPending, match="msgbatch_1"):
        make_gateway(tmp_path, transport, settings=settings).run_many(_batch_calls(toy)[:1], mode="batch")
    assert len(list((tmp_path / "llm_batches").glob("*.json"))) == 1
    transport.polls_before_end = 0
    results = make_gateway(tmp_path, transport, settings=settings).run_many(_batch_calls(toy)[:1], mode="batch")
    assert results[0].output.items == ["alpha"] and list(transport.batches) == ["msgbatch_1"]  # type: ignore[attr-defined]


def test_batch_reserves_its_worst_case_before_submitting(tmp_path: Path, toy: Prompt) -> None:
    transport = ScriptedTransport(_by_doc)
    gateway = make_gateway(tmp_path, transport, budget_usd=0.0005)
    with pytest.raises(BudgetExceeded, match="batch of 4"):
        gateway.run_many(_batch_calls(toy), mode="batch")
    assert transport.batches == {}
