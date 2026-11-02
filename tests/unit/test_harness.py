"""Offline harness (M2): fixture store, replay/recording clients, fake embedders. Offline."""

import base64
import io
import json
import types
from pathlib import Path
from typing import Any

import pytest

import utils.llm as llm
from tests.harness.fake_anthropic import (
    BudgetExceededError, RecordingAnthropic, ReplayAnthropic, ledger_spent_usd,
    worst_case_cost_usd)
from tests.harness.fake_embedder import (
    FAKE_EMBEDDER_NAME, HashingEmbeddingFunction, embed_text, frozen_progpy_store)
from tests.harness.fixtures import FixtureStore, MissingFixtureError, redact_request
from tests.harness.pipeline import offline_pipeline
from utils.prompt_hash import prompt_hash

SCOPE = {"phase": "phase2", "step": "step_c_extract", "component": None}


def _request(text: str = "hello", model: str = "claude-sonnet-4-6") -> dict[str, Any]:
    return {"model": model, "max_tokens": 100, "temperature": 1.0,
            "messages": [{"role": "user", "content": text}]}


def _response(text: str) -> dict[str, Any]:
    return {"text": text, "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2}, "recorded_at": "2026-11-02T00:00:00+00:00"}


def _pdf_b64(pages: int) -> str:
    import fitz
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    return base64.b64encode(doc.tobytes()).decode()


# ── FixtureStore ──────────────────────────────────────────────────────────────

def test_append_then_reload_replays_in_order(tmp_path: Path) -> None:
    store = FixtureStore(tmp_path)
    store.append(_request(), SCOPE, _response("first"), source="recorded")
    store.append(_request(), SCOPE, _response("second"), source="recorded")

    reloaded = FixtureStore(tmp_path)
    assert [reloaded.next_response(_request(), SCOPE)["text"] for _ in range(2)] == ["first", "second"]
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and files[0].name.startswith("phase2__step_c_extract__-__")
    assert json.loads(files[0].read_text())["prompt_sha256"] == prompt_hash(_request())


def test_exhausted_fixture_is_an_error(tmp_path: Path) -> None:
    store = FixtureStore(tmp_path)
    store.append(_request(), SCOPE, _response("only"), source="recorded")
    replay = FixtureStore(tmp_path)
    replay.next_response(_request(), SCOPE)
    with pytest.raises(MissingFixtureError, match="exhausted.*call #2"):
        replay.next_response(_request(), SCOPE)


def test_missing_fixture_names_step_and_shows_nearest_diff(tmp_path: Path) -> None:
    FixtureStore(tmp_path).append(_request("old prompt"), SCOPE, _response("x"), source="recorded")
    with pytest.raises(MissingFixtureError) as err:
        FixtureStore(tmp_path).next_response(_request("new prompt"), SCOPE)
    msg = str(err.value)
    assert "phase2/step_c_extract" in msg and prompt_hash(_request("new prompt")) in msg
    assert '-   "content": "old prompt"' in msg and '+   "content": "new prompt"' in msg


def test_redact_replaces_base64_with_digest() -> None:
    data = _pdf_b64(1)
    req = _request()
    req["messages"] = [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
        {"type": "text", "text": "extract"}]}]
    redacted = json.dumps(redact_request(req))
    assert data not in redacted and "data_sha256" in redacted
    assert req["messages"][0]["content"][0]["source"]["data"] == data  # input untouched


# ── Clients ───────────────────────────────────────────────────────────────────

def test_replay_client_serves_call_claude_with_zero_usage(tmp_path: Path) -> None:
    request = {"model": "claude-sonnet-4-6", "max_tokens": 4000, "temperature": 1.0,
               "messages": [{"role": "user", "content": "ping"}]}
    FixtureStore(tmp_path).append(request, SCOPE, _response("pong"), source="recorded")
    client = ReplayAnthropic(FixtureStore(tmp_path))
    with offline_pipeline(tmp_path / "machines", client):
        assert llm.call_claude("ping") == "pong"
    assert client.calls == [{"phase": "unknown", "step": "unknown", "call_site": None,
                             "component": None, "model": "claude-sonnet-4-6"}]


class _RealMessages:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kw: Any) -> Any:
        self.calls += 1
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="  verbatim\n")],
            model="claude-sonnet-4-6", stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=10, output_tokens=5,
                                        cache_creation_input_tokens=0,
                                        cache_read_input_tokens=0, cache_creation=None))


def test_recording_client_saves_text_verbatim(tmp_path: Path) -> None:
    real = types.SimpleNamespace(messages=_RealMessages())
    store = FixtureStore(tmp_path / "fx")
    client = RecordingAnthropic(real, store, budget_usd=1.0, ledger_path=tmp_path / "ledger.jsonl")
    client.messages.create(**_request())
    record = json.loads(next((tmp_path / "fx").glob("*.json")).read_text())
    assert record["source"] == "recorded"
    assert record["responses"][0]["text"] == "  verbatim\n"
    assert record["responses"][0]["usage"]["input_tokens"] == 10


def test_recording_client_stops_before_exceeding_budget(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(json.dumps({"cost_usd": 0.999}) + "\n")
    real = types.SimpleNamespace(messages=_RealMessages())
    client = RecordingAnthropic(real, FixtureStore(tmp_path / "fx"), 1.0, ledger_path)
    with pytest.raises(BudgetExceededError, match="exceeds the \\$1.00 budget"):
        client.messages.create(**_request())
    assert real.messages.calls == 0


def test_worst_case_counts_pdf_pages_and_full_max_tokens() -> None:
    text_only = worst_case_cost_usd(_request("x" * 2500))  # 1000 tok in + 100 out
    assert text_only == pytest.approx((1000 * 3 + 100 * 15) / 1e6)
    req = _request()
    req["messages"] = [{"role": "user", "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                        "data": _pdf_b64(3)}}]}]
    assert worst_case_cost_usd(req) == pytest.approx((9000 * 3 + 100 * 15) / 1e6)


def test_ledger_spent_sums_costs(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    assert ledger_spent_usd(path) == 0.0
    path.write_text('{"cost_usd": 0.25}\n{"cost_usd": null}\n{"cost_usd": 0.5}\n')
    assert ledger_spent_usd(path) == pytest.approx(0.75)


# ── Embedders ─────────────────────────────────────────────────────────────────

def test_hashing_embedder_is_deterministic_and_lexical() -> None:
    a, b, c = embed_text("filter pressure drop"), embed_text("pressure drop filter"), embed_text("motor")
    assert a == b
    assert sum(x * y for x, y in zip(a, b)) == pytest.approx(1.0)
    assert sum(x * y for x, y in zip(a, c)) < 0.9
    ef = HashingEmbeddingFunction(api_key="ignored", model_name="ignored")
    assert ef.name() == FAKE_EMBEDDER_NAME and len(ef(["x"])[0]) == len(a)


def test_hashing_embedder_works_with_chroma(tmp_path: Path) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    col = client.create_collection("tiny", embedding_function=HashingEmbeddingFunction())
    col.add(ids=["f", "p"], documents=["filter cartridge clogging", "pump impeller wear"])
    again = client.get_collection("tiny", embedding_function=HashingEmbeddingFunction())
    assert again.query(query_texts=["impeller wear"], n_results=1)["ids"][0] == ["p"]


def test_frozen_progpy_store_serves_framework_context() -> None:
    from utils.progpy_rag import get_framework_context
    with frozen_progpy_store():
        first, second = get_framework_context(), get_framework_context()
    assert first == second and "PrognosticsModel" in first


def test_frozen_progpy_store_rejects_unknown_query(tmp_path: Path) -> None:
    path = tmp_path / "hits.json"
    path.write_text(json.dumps({"hits": {}}))
    with frozen_progpy_store(path):
        from vector_store import ChromaDBStore  # type: ignore[import-not-found]
        with pytest.raises(LookupError, match="no frozen ProgPy hits"):
            ChromaDBStore().query("anything", top_k=4)
