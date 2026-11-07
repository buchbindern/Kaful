"""Understand building blocks: context selection, merging, quote classes, provenance, sampling."""

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from kaful.artifacts.document_index import (
    Chunk,
    DocumentIndex,
    DocumentRecord,
    PageLayer,
    PageText,
    chunk_id_for,
    index_hash_for,
    text_sha256,
)
from kaful.config import IngestSettings, LLMSettings, UnderstandSettings
from kaful.evidence import verify_quote
from kaful.llm.budget import RunBudget
from kaful.llm.embeddings import Embedder, VectorStore
from kaful.llm.ledger import Ledger
from kaful.phases.p2_understand import d_build_artifacts as build
from kaful.phases.p2_understand.a_select_context import select_context
from kaful.phases.p2_understand.b_merge_samples import consensus, keyed, merge
from kaful.phases.p2_understand.c_verify_quotes import Checker, PageLookup, provenance
from kaful.prompts.understand.sensor_channels.models import ChannelExtraction, ExtractedChannel
from tests.harness.kaful_factories import BASE, FINGERPRINT, SHA, SIGNALS
from tests.harness.kaful_ingest import FakeEmbeddings

SETTINGS = UnderstandSettings()


def _index(texts: list[str], kind: str = "manual", layers: dict[int, str] | None = None) -> DocumentIndex:
    chunks = tuple(sorted((Chunk(chunk_id=chunk_id_for(SHA, p, p, "v1"), doc_id="manual", page_start=p, page_end=p,
                                 text=t, text_sha256=text_sha256(t), chunker_version="v1")
                           for p, t in enumerate(texts, start=1)), key=lambda c: c.chunk_id))
    pages = tuple(PageText(doc_id="manual", page=p, text=t, page_class="text", signals=SIGNALS,
                           layers=(PageLayer(kind="llm", text=(layers or {})[p], producer="x"),)
                           if p in (layers or {}) else ())
                  for p, t in enumerate(texts, start=1))
    return DocumentIndex(**BASE, index_hash=index_hash_for(chunks, FINGERPRINT), fingerprint=FINGERPRINT,
                         chunks=chunks, pages=pages, documents=(DocumentRecord(
                             doc_id="manual", path="manual/pump.pdf", kind=kind, sha256=SHA,  # type: ignore[arg-type]
                             n_pages=len(texts)),))


def test_small_index_goes_in_whole_in_document_order() -> None:
    index = _index(["first page", "second page", "third page"])
    context = select_context(index, None, None, SETTINGS, IngestSettings())
    assert context.selection.mode == "whole" and context.text == "first page\n\nsecond page\n\nthird page"
    assert [index.chunks[[c.chunk_id for c in index.chunks].index(i)].page_start
            for i in context.selection.chunk_ids] == [1, 2, 3]


def test_retrieval_breaks_ties_by_chunk_id_and_lists_by_page(tmp_path: Path) -> None:
    index = _index([f"page {p} " + "x" * 2000 for p in range(1, 9)])
    embedder = Embedder(settings=IngestSettings(embedding_dimensions=8), llm=LLMSettings(),
                        ledger=Ledger(tmp_path / "l.jsonl"), budget=RunBudget(1.0), transport=FakeEmbeddings(),
                        store=VectorStore(tmp_path / "v"))
    settings = UnderstandSettings(context_max_tokens=1600)
    context = select_context(index, np.ones((8, 8), dtype=np.float32), embedder, settings, IngestSettings())
    picked = sorted(c.chunk_id for c in index.chunks)[:3]  # all tied: the three lowest ids fit 1600 tokens
    by_id = {c.chunk_id: c for c in index.chunks}
    assert context.selection.mode == "retrieval" and sorted(context.selection.chunk_ids) == picked
    assert [by_id[i].page_start for i in context.selection.chunk_ids] == sorted(by_id[i].page_start for i in picked)
    assert len(context.selection.queries) == 9


def test_consensus_prefers_majority_then_first_seen() -> None:
    assert consensus(["a", "b", "b"]) == "b" and consensus(["a", "b"]) == "a"
    assert consensus([{"x": 1}, {"x": 1}, {"x": 2}]) == {"x": 1}


def _channel(name: str, unit: str, quote: str = "flow_rate (m3/h), range 0 to 12",
             **changes: Any) -> ExtractedChannel:
    return ExtractedChannel.model_validate({
        "secondary_role": None, "secondary_reason": None, "secondary_evidence": None, **changes,
        "name": name, "printed_name": name, "description": "d", "role": changes.get("role", "observation"),
        "dtype": "float", "unit": unit,
        "range_min": 0, "range_max": 12, "categories": None, "sample_period": None,
        "range_evidence": {"doc_id": "manual", "page": 1, "quote": quote}, "range_role": "spec",
        "evidence": [{"doc_id": "manual", "page": 1, "quote": quote}]})


def test_merge_counts_one_vote_per_sample_and_unions_quotes() -> None:
    samples = [[_channel("flow_rate", "m3/h"), _channel("flow_rate", "m3/h")],
               [_channel("flow_rate", "l/min", "Flow rate is measured at the outlet.")],
               [_channel("flow_rate", "m3/h"), _channel("extra", "bar")]]
    merged = merge(samples, keyed(lambda c: c.name))
    assert [(m.key, m.agreement) for m in merged] == [(("extra",), 1 / 3), (("flow_rate",), 1.0)]
    flow = merged[1]
    assert flow.item.unit == "m3/h" and len(flow.quotes) == 2


def test_quote_classes_follow_rule_4() -> None:
    page = PageText(doc_id="manual", page=1, text="Replace the cartridge every 2000\noperating hours.",
                    page_class="text", signals=SIGNALS,
                    layers=(PageLayer(kind="llm", text="Countdown: 10 s", producer="x"),
                            PageLayer(kind="table_md", text="| Filter | F1 |", producer="pymupdf")))
    def verdict(quote: str) -> str:
        return verify_quote(quote, page, None, 0.85, 0.6)
    assert verdict("Replace the cartridge every 2000 operating hours.") == "verbatim"
    assert verdict("replace the cartridge every 2000 operation hours") == "near"
    assert verdict("| Filter | F1 |") == "near"  # only in the ruled-table markdown (code-derived, no LLM)
    assert verdict("Countdown: 10 s") == "in_layer"
    assert verdict("The cartridge lasts five years.") == "unsupported"


def test_provenance_downgrades_and_maps_document_kinds() -> None:
    index = _index(["Rated flow 6 m3/h."], layers={1: "Countdown: 10 s"})
    pages = PageLookup(index)
    ok = provenance((("manual", 1, "Rated flow 6 m3/h."),), 1.0, pages, SETTINGS)
    assert (ok.source, ok.downgraded_from, ok.confidence, ok.ref) == ("manual", None, 1.0, "manual p.1")
    layer = provenance((("manual", 1, "Countdown: 10 s"),), 2 / 3, pages, SETTINGS)
    assert (layer.source, layer.downgraded_from) == ("llm_judgement", "manual")
    assert layer.confidence == pytest.approx(2 / 3 * 0.5, abs=1e-6)
    wrong_page = provenance((("manual", 9, "Rated flow 6 m3/h."),), 1.0, pages, SETTINGS)
    assert wrong_page.evidence[0].verification == "unsupported"
    standard = provenance((("manual", 1, "Rated flow 6 m3/h."),), 1.0, PageLookup(_index(["Rated flow 6 m3/h."],
                                                                                       kind="standard")), SETTINGS)
    assert standard.source == "standard"
    thesis = PageLookup(_index(["Zone B: 1.8 mm/s."], kind="literature"))
    secondary = provenance((("manual", 1, "Zone B: 1.8 mm/s."),), 1.0, thesis, SETTINGS)
    assert (secondary.source, secondary.downgraded_from) == ("literature", None)  # a thesis is never `standard`
    unread = provenance((("manual", 1, "Zone C: 4.5 mm/s."),), 1.0, thesis, SETTINGS)
    assert (unread.source, unread.downgraded_from) == ("llm_judgement", "literature")


def _sampled(period: dict[str, object] | None, role: str = "observation") -> ExtractedChannel:
    data = _channel("filter_dp", "bar").model_dump()
    period = None if period is None else {"role": "spec", **period}
    return ExtractedChannel.model_validate({**data, "role": role, "sample_period": period})


@pytest.mark.parametrize(("period", "role", "expected"), [
    (None, "observation", "unknown"),
    ({"value": 10, "unit": "seconds", "evidence": {"doc_id": "manual", "page": 1,
                                                   "quote": "Logged once every 10 seconds."}}, "observation", "periodic"),
    ({"value": 5, "unit": "seconds", "evidence": {"doc_id": "manual", "page": 1,
                                                  "quote": "Logged once every 10 seconds."}}, "observation", "unknown"),
    ({"value": 10, "unit": "seconds", "evidence": {"doc_id": "manual", "page": 1,
                                                   "quote": "Logged every minute."}}, "observation", "unknown"),
    ({"value": 10, "unit": "bar", "evidence": {"doc_id": "manual", "page": 1,
                                               "quote": "Logged once every 10 seconds."}}, "observation", "unknown"),
    (None, "event", "event_driven"),
])
def test_a_period_counts_only_when_its_quote_states_it(period: dict[str, object] | None, role: str,
                                                        expected: str) -> None:
    index = _index(["filter_dp (bar), range 0 to 12. Logged once every 10 seconds."])
    log = build.Log()
    fields = build._sampling(_sampled(period, role), Checker(PageLookup(index), SETTINGS), log)
    assert fields["sampling"] == expected
    assert (expected == "periodic") == ("sample_period_s" in fields)
    assert (len(log.notes) > 0) == (period is not None and expected == "unknown")


def test_a_range_from_a_worked_example_is_not_stored() -> None:
    page = "Example: a healthy pump shows flow_rate (m3/h), range 0 to 12 during start-up.\nflow_rate is logged."
    index = _index([page])
    merged = merge([[_channel("flow_rate", "m3/h")]], keyed(lambda c: c.name))
    log = build.Log()
    [channel] = build.channels(merged, (), Checker(PageLookup(index), SETTINGS), log)
    assert channel.range is None and channel.range_evidence is None
    assert "channel flow_rate: range: role spec -> example (quoted from a worked example)" in log.notes
    assert "channel flow_rate: range: an example value, not stored" in log.notes


def test_a_documented_range_keeps_its_role_on_the_evidence() -> None:
    index = _index(["flow_rate (m3/h), range 0 to 12"])
    merged = merge([[_channel("flow_rate", "m3/h")]], keyed(lambda c: c.name))
    [channel] = build.channels(merged, (), Checker(PageLookup(index), SETTINGS), build.Log())
    assert channel.range == (0, 12) and channel.range_evidence is not None
    assert (channel.range_evidence.verification, channel.range_evidence.value_role) == ("verbatim", "spec")


def test_a_stated_secondary_role_is_verified_like_any_quote() -> None:
    line = "The load meter shows the spindle load; it rises as the bearings wear."
    index = _index([f"flow_rate (m3/h), range 0 to 12\n{line}"])
    merged = merge([[_channel("flow_rate", "m3/h", role="operating_input", secondary_role="observation",
                              secondary_reason="rises as the bearings wear",
                              secondary_evidence={"doc_id": "manual", "page": 1, "quote": line})]],
                   keyed(lambda c: c.name))
    [channel] = build.channels(merged, (), Checker(PageLookup(index), SETTINGS), build.Log())
    [second] = channel.secondary_roles
    assert (second.role, second.source, second.evidence[0].verification) == ("observation", "manual", "verbatim")


def test_an_unquoted_secondary_role_is_a_judgement_with_no_confidence() -> None:
    index = _index(["flow_rate (m3/h), range 0 to 12"])
    merged = merge([[_channel("flow_rate", "m3/h", role="operating_input", secondary_role="observation",
                              secondary_reason="load rises with friction")]], keyed(lambda c: c.name))
    [channel] = build.channels(merged, (), Checker(PageLookup(index), SETTINGS), build.Log())
    [second] = channel.secondary_roles
    assert (second.source, second.evidence, second.confidence) == ("llm_judgement", (), 0.0)


def test_the_output_model_allows_a_secondary_role_only_on_inputs_and_observations() -> None:
    def reply(**changes: Any) -> ChannelExtraction:
        return ChannelExtraction(channels=[_channel("c", "h", **changes)])
    with pytest.raises(ValidationError, match="secondary_role is only"):
        reply(role="usage_clock", secondary_role="observation", secondary_reason="r")
    with pytest.raises(ValidationError, match="secondary_reason"):
        reply(role="operating_input", secondary_role="observation")
    assert reply(role="operating_input", secondary_role="observation", secondary_reason="r")


def test_reused_older_responses_are_counted_for_the_run_record() -> None:
    """Sonnet's v1 replies that ended normally serve v2 (only max_tokens rose): the run must say so."""
    from kaful.llm.gateway import LLMResult
    from kaful.phases.p2_understand.b_merge_samples import reuse_notes, sample

    class Gateway:
        def call(self, call: Any) -> LLMResult:
            reused = (1, "abc") if call.sample < 2 else None
            return LLMResult(output=ChannelExtraction(channels=[]), text="{}", prompt_sha256="x", attempts=1,
                             cost_usd=0.0, reused_from=reused)
    prompt = types.SimpleNamespace(id="understand.components", version=2)
    reused: list[tuple[str, int, int]] = []
    outputs = sample(Gateway(), prompt, {}, 3, "extract_components", "machine_model", reused)  # type: ignore[arg-type]
    assert len(outputs) == 3 and reused == [("understand.components", 1, 2), ("understand.components", 1, 2)]
    assert reuse_notes(reused) == ("2 sample(s): understand.components v1 responses reused under v2 (the request "
                                   "differed only in a lower max_tokens; the response ended normally)",)
