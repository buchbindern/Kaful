"""Understand building blocks: context selection, merging, quote classes, provenance, sampling."""

from pathlib import Path

import numpy as np
import pytest

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
from kaful.phases.understand import build
from kaful.phases.understand.context import select_context
from kaful.phases.understand.merge import consensus, keyed, merge
from kaful.phases.understand.verify import PageLookup, provenance
from kaful.prompts.understand.sensor_channels.models import ExtractedChannel
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


def _channel(name: str, unit: str, quote: str = "flow_rate (m3/h), range 0 to 12") -> ExtractedChannel:
    return ExtractedChannel.model_validate({
        "name": name, "printed_name": name, "description": "d", "role": "observation", "dtype": "float", "unit": unit,
        "range_min": 0, "range_max": 12, "categories": None, "sample_period": None,
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
    fields = build._sampling(_sampled(period, role), build.Checker(PageLookup(index), SETTINGS), log)
    assert fields["sampling"] == expected
    assert (expected == "periodic") == ("sample_period_s" in fields)
    assert (len(log.notes) > 0) == (period is not None and expected == "unknown")
