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


def test_images_go_before_the_task_and_are_budgeted(tmp_path: Path) -> None:
    import struct
    import zlib

    from kaful.llm.budget import estimate_input_tokens
    from kaful.prompts.registry import PromptError

    directory = tmp_path / "toy" / "look"
    directory.mkdir(parents=True)
    for name in ("system.md", "models.py"):
        (directory / name).write_text((TOY / name).read_text())
    (directory / "task.md").write_text("Question: $question\n")
    (directory / "prompt.toml").write_text((TOY / "prompt.toml").read_text()
        .replace('"toy.echo_items"', '"toy.look"').replace('inputs = ["doc", "question"]', 'inputs = ["question"]\nimages = ["page"]')
        .replace('context = { file = "context.md", cache = "1h" }\n', ""))
    prompt = load_prompt_dir(directory)
    raw = b"\x00" + b"\xff\xff\xff" * 3
    ihdr = struct.pack(">IIBBBBB", 300, 250, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
           + struct.pack(">I", len(zlib.compress(raw))) + b"IDAT" + zlib.compress(raw) + b"\0" * 4)
    gateway = make_gateway(tmp_path, None)
    request = gateway.request_for(prompt, {"question": "What?"}, {"page": png})
    image, task = request["messages"][0]["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/png" and task["type"] == "text"
    without = estimate_input_tokens({**request, "messages": [{"role": "user", "content": [task]}]}, gateway.settings)
    assert estimate_input_tokens(request, gateway.settings) - without == 100  # 300 x 250 / 750
    with pytest.raises(PromptError, match="images"):
        gateway.request_for(prompt, {"question": "What?"})
    with pytest.raises(PromptError, match="PNG"):
        gateway.request_for(prompt, {"question": "What?"}, {"page": b"GIF89a"})


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
    # The validation errors are saved with the invalid attempt (seen live 2026-11-07: they were nowhere).
    assert "items must be unique" in (rows[0].error or "") and rows[1].error is None
    assert result.cost_usd == pytest.approx(total_cost(rows))
    # A rerun replays both attempts from the cache at $0.
    rerun = make_gateway(tmp_path, ScriptedTransport(lambda _r: pytest.fail("network")))
    assert rerun.call(_call(toy)).attempts == 2 and total_cost(rerun.ledger.rows()[2:]) == 0


def test_second_invalid_reply_fails(tmp_path: Path, toy: Prompt) -> None:
    gateway = make_gateway(tmp_path, ScriptedTransport(lambda _r: DUPLICATE))
    with pytest.raises(LLMOutputError, match="after one retry") as exc:
        gateway.call(_call(toy))
    assert exc.value.last_text == DUPLICATE
    assert all("items must be unique" in (r.error or "") for r in gateway.ledger.rows())


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


def test_batch_is_sliced_to_fit_the_remaining_budget(tmp_path: Path, toy: Prompt) -> None:
    probe = make_gateway(tmp_path / "probe", None)
    calls = _batch_calls(toy)
    worst = probe.worst_case(toy, probe.request_for(toy, calls[0].inputs), batch=True)
    transport = ScriptedTransport(_by_doc, usage=TokenUsage(input_tokens=100, output_tokens=10))
    gateway = make_gateway(tmp_path, transport, budget_usd=2.5 * worst)
    results = gateway.run_many(calls, mode="batch")
    assert [r.output.items for r in results] == [["alpha"], ["bravo"], ["charlie"], ["delta"]]  # type: ignore[attr-defined]
    first_round = [ids for name, ids in sorted(transport.batches.items())][:2]
    assert [len(ids) for ids in first_round] == [2, 2]  # 4 × worst would not fit; two slices of 2 do
    assert gateway.budget.spent_usd <= gateway.budget.limit_usd and gateway.budget.reserved_usd == 0


def test_a_charged_batch_call_is_ledgered_even_when_a_later_slice_is_refused(tmp_path: Path, toy: Prompt) -> None:
    """Regression (M6 extract_page v3 evals): a slice's results were charged when collected
    but only ledgered when validated, so a refused later slice left paid calls off the ledger."""
    probe = make_gateway(tmp_path / "probe", None)
    calls = _batch_calls(toy)
    worst = probe.worst_case(toy, probe.request_for(toy, calls[0].inputs), batch=True)
    transport = ScriptedTransport(_by_doc, usage=TokenUsage(input_tokens=0, output_tokens=450))
    gateway = make_gateway(tmp_path, transport, budget_usd=1.5 * worst)  # one slice of 1 fits; the next does not
    with pytest.raises(BudgetExceeded):
        gateway.run_many(calls, mode="batch")
    assert len(transport.batches) == 1 and gateway.budget.spent_usd > 0
    rows = [r for r in gateway.ledger.rows() if r.event == "llm_call"]
    assert len(rows) == 1 and rows[0].component == calls[0].scope.component
    assert total_cost(gateway.ledger.rows()) == pytest.approx(gateway.budget.spent_usd)


def test_batch_reserves_its_worst_case_before_submitting(tmp_path: Path, toy: Prompt) -> None:
    probe = make_gateway(tmp_path / "probe", None)
    calls = _batch_calls(toy)
    worst = probe.worst_case(toy, probe.request_for(toy, calls[0].inputs), batch=True)
    transport = ScriptedTransport(_by_doc)
    with pytest.raises(BudgetExceeded, match="batch of 1"):
        make_gateway(tmp_path, transport, budget_usd=0.5 * worst).run_many(calls, mode="batch")
    assert transport.batches == {}


def _store_older(gateway: Any, toy: Prompt, stop_reason: str) -> str:
    """Cache, as version 7 with max_tokens 100, the response to the toy request."""
    from kaful.llm.cache import StoredRecord, StoredResponse

    request = gateway.request_for(toy, _call(toy).inputs)
    older = prompt_hash({**request, "max_tokens": 100})
    record = StoredRecord(prompt_sha256=older, prompt_id=toy.id, prompt_version=7, phase="p", step="s",
                          source="recorded", request=request)
    reply = StoredResponse(text=json.dumps({"items": ["pump"]}), model="claude-haiku-5-5", stop_reason=stop_reason,
                           usage={"input_tokens": 10, "output_tokens": 5}, recorded_at="2026-11-05T00:00:00")
    gateway.cache.put(record, 0, reply)
    return older


def test_an_older_response_that_ended_normally_is_reused_and_recorded(tmp_path: Path, toy: Prompt) -> None:
    from kaful.config import ReuseRule

    settings = LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5",
                           reuse={toy.id: ReuseRule(version=7, max_tokens=100)})
    gateway = make_gateway(tmp_path, None, settings=settings)  # offline: only the cache can answer
    older = _store_older(gateway, toy, "end_turn")
    result = gateway.call(_call(toy))
    assert result.reused_from == (7, older) and result.output.items == ["pump"]  # type: ignore[attr-defined]
    row = next(r for r in gateway.ledger.rows() if r.event == "llm_call")
    assert row.reused_from == f"{toy.id} v7 {older}" and row.cost_usd == 0 and row.response_cache == "hit"


def test_an_older_response_that_hit_its_limit_is_not_reused(tmp_path: Path, toy: Prompt) -> None:
    from kaful.config import ReuseRule

    settings = LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5",
                           reuse={toy.id: ReuseRule(version=7, max_tokens=100)})
    gateway = make_gateway(tmp_path, None, settings=settings)
    _store_older(gateway, toy, "max_tokens")
    with pytest.raises(OfflineMissError):
        gateway.call(_call(toy))
    no_rule = make_gateway(tmp_path / "plain", None)
    _store_older(no_rule, toy, "end_turn")
    with pytest.raises(OfflineMissError):  # without a ReuseRule nothing older is read
        no_rule.call(_call(toy))


class _FlakyStatus:
    """A transport whose batch_status fails transiently `errors` times, then reports ended."""

    def __init__(self, errors: int) -> None:
        self.errors, self.calls = errors, 0

    def batch_status(self, batch_id: str) -> str:
        from kaful.llm.transport import TransientTransportError

        self.calls += 1
        if self.calls <= self.errors:
            raise TransientTransportError("batch status: APIConnectionError: Connection error.")
        return "ended"


def test_batch_polling_survives_transient_network_errors() -> None:
    from kaful.llm.batch import wait_for_batch

    flaky, slept = _FlakyStatus(errors=2), []
    wait_for_batch(flaky, "msgbatch_x", 60.0, 3600.0, slept.append, lambda: 0.0, max_errors=3)  # type: ignore[arg-type]
    assert flaky.calls == 3 and slept == [60.0, 60.0]


def test_batch_polling_stops_cleanly_when_errors_persist() -> None:
    from kaful.llm.batch import BatchPending, wait_for_batch

    with pytest.raises(BatchPending, match="rerun to collect it"):  # the batch stays saved; nothing is paid twice
        wait_for_batch(_FlakyStatus(errors=10), "msgbatch_x", 60.0, 3600.0, lambda _s: None, lambda: 0.0,  # type: ignore[arg-type]
                       max_errors=3)
