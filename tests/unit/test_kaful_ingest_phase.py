"""Ingest phase end to end on a synthetic machine (offline), plus extraction, chunking,
embeddings and vector search."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kaful import pipeline
from kaful.config import REPO_ROOT, Escalation, KafulSettings, ModelChoice
from kaful.llm.budget import RunBudget
from kaful.llm.embeddings import Embedder, VectorStore
from kaful.llm.gateway import OfflineMissError
from kaful.llm.ledger import Ledger, read_ledger
from kaful.phases.ingest import phase as ingest_phase
from kaful.phases.ingest.chunk import chunk_document, coverage_gaps
from kaful.phases.ingest.extract import render_extraction
from kaful.phases.ingest.index import IndexMismatch, VectorIndex, check_index_hash, load_index, load_vectors
from kaful.phases.ingest.read import ReadDocument, TocEntry
from kaful.prompts.ingest.extract_page.models import PageExtraction
from kaful.prompts.registry import PromptLock
from tests.harness.kaful_ingest import (
    DIMENSIONS,
    EXTRACTION,
    MACHINE_ID,
    FakeEmbeddings,
    extraction_reply,
    ingest_settings,
    locked_extract_page,
    make_pdf_machine,
)
from tests.harness.kaful_llm import ScriptedTransport


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: ScriptedTransport,
         embeddings: FakeEmbeddings, allow_failures: bool = False,
         lock: PromptLock | None = None, settings: KafulSettings | None = None) -> pipeline.RunContext:
    monkeypatch.setattr(ingest_phase, "read_lock", lambda: lock or locked_extract_page())
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest",), budget_usd=1.0,
                                  allow_failures=allow_failures)
    ctx, base = pipeline.start_run(settings or ingest_settings(tmp_path), request, transport=transport,
                                   embedding_transport=embeddings)
    pipeline.execute(ctx, ("ingest",), base)
    return ctx


def test_ingest_builds_a_checked_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    transport, embeddings = ScriptedTransport(extraction_reply), FakeEmbeddings()
    ctx = _run(tmp_path, monkeypatch, transport, embeddings)
    record = ctx.manifest.phase("ingest")
    assert (record.status, record.failures, ctx.manifest.status) == ("passed", (), "passed")
    index = load_index(ctx)
    assert index.index_hash == ctx.manifest.index_hash and index.fingerprint.extraction is not None
    classes = [p.page_class for p in index.pages]
    assert classes == ["text", "table_ruled", "table_unruled", "figure", "figure", "blank"]
    assert len(transport.batches) == 1 and len(transport.sent) == 3  # one batch, the three LLM pages
    llm_pages = [p for p in index.pages if any(layer.kind == "llm" for layer in p.layers)]
    assert [p.page for p in llm_pages] == [3, 4, 5] and "15,0" in llm_pages[0].layers[-1].text
    assert [p.label for p in index.pages[:5]] == ["1", "2", "3", "4", "5"]
    assert "Rated flow" in index.pages[1].text and "[llm" not in index.pages[2].text  # raw text unchanged
    assert coverage_gaps(index.pages, index.chunks) == []
    assert index.vectors is not None
    vectors = load_vectors(ctx.machine_dir, index.vectors)
    assert vectors.shape == (len(index.chunks), DIMENSIONS)
    rows = read_ledger(ctx.run_dir / "ledger.jsonl")
    assert {r.event for r in rows} == {"llm_batch_submit", "llm_call", "embedding"}
    assert record.cost_usd == pytest.approx(sum(r.cost_usd or 0 for r in rows)) and record.cost_usd > 0


def test_rerun_is_free_and_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    first = _run(tmp_path, monkeypatch, ScriptedTransport(extraction_reply), FakeEmbeddings())
    transport, embeddings = ScriptedTransport(extraction_reply), FakeEmbeddings()
    second = _run(tmp_path, monkeypatch, transport, embeddings)
    assert transport.sent == [] and embeddings.calls == [] and second.manifest.phase("ingest").cost_usd == 0
    assert second.manifest.index_hash == first.manifest.index_hash
    a, b = load_index(first), load_index(second)
    assert a.model_copy(update={"run_id": "x", "vectors": None}) == b.model_copy(update={"run_id": "x", "vectors": None})


def _figures_only_on_figure_pages(request: dict[str, Any]) -> str:
    """Haiku sees a table on the unruled-table page and none on the figure pages."""
    tables = "Figure" not in json.dumps(request["messages"])
    return json.dumps({**EXTRACTION, "tables": EXTRACTION["tables"] if tables else []})


def test_pages_with_tables_escalate_to_the_stronger_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    settings = ingest_settings(tmp_path)
    choice = ModelChoice(model="claude-haiku-5-5", escalate=Escalation(model="claude-sonnet-5-5",
                                                                       when=("output_has_tables",)))
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(
        update={"prompts": {"ingest.extract_page": choice}})})
    transport = ScriptedTransport(_figures_only_on_figure_pages)
    ctx = _run(tmp_path, monkeypatch, transport, FakeEmbeddings(), settings=settings)
    assert ctx.manifest.phase("ingest").status == "passed"
    assert [r["model"] for r in transport.sent] == ["claude-haiku-5-5"] * 3 + ["claude-sonnet-5-5"]
    index = load_index(ctx)
    models = {p.page: layer.model for p in index.pages for layer in p.layers if layer.kind == "llm"}
    assert models == {3: "claude-sonnet-5-5", 4: "claude-haiku-5-5", 5: "claude-haiku-5-5"}
    assert index.fingerprint.extraction == (
        "ingest.extract_page v3 claude-haiku-5-5 -> claude-sonnet-5-5 when output_has_tables")
    calls = [r for r in read_ledger(ctx.run_dir / "ledger.jsonl") if r.event == "llm_call"]
    assert sorted((r.component, r.model, r.route) for r in calls) == [
        ("tp_manual:p3", "claude-haiku-5-5", "first pass; escalate to claude-sonnet-5-5 when output_has_tables"),
        ("tp_manual:p3", "claude-sonnet-5-5", "escalated from claude-haiku-5-5: output_has_tables"),
        ("tp_manual:p4", "claude-haiku-5-5", "first pass; escalate to claude-sonnet-5-5 when output_has_tables"),
        ("tp_manual:p5", "claude-haiku-5-5", "first pass; escalate to claude-sonnet-5-5 when output_has_tables")]


def test_a_document_can_send_its_ruled_tables_to_the_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    machine_dir = make_pdf_machine(tmp_path)
    toml = machine_dir / "machine.toml"
    toml.write_text(toml.read_text().replace('kind = "manual"', 'kind = "manual"\nruled_tables = "llm"'))
    transport = ScriptedTransport(extraction_reply)
    ctx = _run(tmp_path, monkeypatch, transport, FakeEmbeddings())
    assert ctx.manifest.phase("ingest").status == "passed" and len(transport.sent) == 4
    index = load_index(ctx)
    ruled = index.pages[1]
    assert (ruled.page_class, ruled.llm_by_document) == ("table_ruled", True)
    assert [layer.kind for layer in ruled.layers] == ["llm", "table_md"]  # the free markdown is kept too
    assert index.fingerprint.llm_ruled_tables == ("tp_manual",) and index.documents[0].ruled_tables == "llm"


def test_a_failed_first_pass_escalates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    settings = ingest_settings(tmp_path)
    choice = ModelChoice(model="claude-haiku-5-5", escalate=Escalation(
        model="claude-sonnet-5-5", when=("first_pass_failed", "output_has_tables")))
    settings = settings.model_copy(update={"llm": settings.llm.model_copy(
        update={"prompts": {"ingest.extract_page": choice}})})

    def reply(request: dict[str, Any]) -> str:  # Haiku fails on the figure pages; Sonnet answers
        if request["model"] == "claude-haiku-5-5" and "Figure" in json.dumps(request["messages"]):
            return "not json"
        return json.dumps({**EXTRACTION, "tables": []})

    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(reply), FakeEmbeddings(), settings=settings)
    assert ctx.manifest.phase("ingest").status == "passed"
    index = load_index(ctx)
    models = {p.page: layer.model for p in index.pages for layer in p.layers if layer.kind == "llm"}
    assert models == {3: "claude-haiku-5-5", 4: "claude-sonnet-5-5", 5: "claude-sonnet-5-5"}
    assert not any(p.extraction_error for p in index.pages)
    routes = {r.route for r in read_ledger(ctx.run_dir / "ledger.jsonl") if r.model == "claude-sonnet-5-5"}
    assert routes == {"escalated from claude-haiku-5-5: first_pass_failed"}
    assert index.fingerprint.extraction is not None and index.fingerprint.extraction.endswith(
        "-> claude-sonnet-5-5 when first_pass_failed or output_has_tables")
    with pytest.raises(ValueError, match="sorted"):
        Escalation(model="m", when=("output_has_tables", "first_pass_failed"))


def test_invalid_extraction_fails_the_phase_but_keeps_raw_text(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(lambda r: "not json"), FakeEmbeddings(), allow_failures=True)
    record = ctx.manifest.phase("ingest")
    assert record.status == "failed" and len(record.failures) == 3
    assert all("extraction failed" in f for f in record.failures)
    index = load_index(ctx)
    assert [p.page for p in index.pages if p.extraction_error] == [3, 4, 5]
    assert coverage_gaps(index.pages, index.chunks) == []


def test_unlocked_prompt_stops_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    transport = ScriptedTransport(extraction_reply)
    ctx = _run(tmp_path, monkeypatch, transport, FakeEmbeddings(), lock=PromptLock())
    record = ctx.manifest.phase("ingest")
    assert record.status == "failed" and "not locked" in record.failures[0] and transport.sent == []


def test_index_hash_mismatch_warns_or_raises() -> None:
    check_index_hash("a" * 64, "a" * 64, strict=True)
    check_index_hash("a" * 64, "b" * 64, strict=False)
    with pytest.raises(IndexMismatch):
        check_index_hash("a" * 64, None, strict=True)


def _doc(tmp_path: Path) -> ReadDocument:
    from kaful.config import IngestSettings
    from kaful.phases.ingest.documents import check_documents
    from kaful.phases.ingest.read import read_document

    machine_dir = make_pdf_machine(tmp_path)
    from kaful.config import load_machine_config
    record = check_documents(machine_dir, load_machine_config(machine_dir)).records[0]
    return read_document(machine_dir, record, IngestSettings())


def test_chunks_follow_sections_skip_blanks_and_respect_the_cap(tmp_path: Path) -> None:
    from kaful.config import IngestSettings

    document = _doc(tmp_path)
    chunks = chunk_document(document, IngestSettings())
    assert [(c.page_start, c.page_end, c.section) for c in chunks] == [
        (1, 1, "Overview"), (2, 2, "Data"), (3, 3, "Settings"), (4, 5, "Figures")]
    assert chunks[0].text.startswith("[tp_manual | Overview]\n--- page 1 (label 1) ---")
    assert "[table_md layer: pymupdf" in chunks[1].text
    tiny = chunk_document(document, IngestSettings(chunk_max_tokens=100, chunk_chars_per_token=0.5))
    assert [(c.page_start, c.page_end) for c in tiny] == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    no_toc = ReadDocument(document.record, document.pages, (TocEntry(1, "All", 1),))
    assert [(c.page_start, c.page_end) for c in chunk_document(no_toc, IngestSettings())] == [(1, 5)]
    assert coverage_gaps(document.pages, chunks[1:]) == ["tp_manual p1: in 0 chunks"]


def test_a_blank_classed_page_with_text_is_chunked(tmp_path: Path) -> None:
    """A part-divider page (a title, little else) is classed blank but keeps its text."""
    from kaful.config import IngestSettings

    document = _doc(tmp_path)
    divider = document.pages[0].model_copy(update={"page_class": "blank", "text": "II. FANUC AC SPINDLE MOTOR"})
    empty = document.pages[1].model_copy(update={"page_class": "blank", "text": " \n", "layers": ()})
    pages = (divider, empty, *document.pages[2:])
    chunks = chunk_document(ReadDocument(document.record, pages, document.toc), IngestSettings())
    assert [(c.page_start, c.page_end) for c in chunks][:2] == [(1, 1), (3, 3)]
    assert "II. FANUC AC SPINDLE MOTOR" in chunks[0].text and chunks[0].chunker_version == "chunker-v2"
    assert coverage_gaps(pages, chunks) == []
    assert coverage_gaps(pages, chunks[1:]) == ["tp_manual p1: in 0 chunks"]


def test_render_extraction_marks_sources() -> None:
    text = render_extraction(PageExtraction.model_validate(EXTRACTION))
    assert "Table (text_layer): Settings\n| Parameter | Value | - |" in text
    assert "| Cake Thickness [mm] | 15,0 | - |" in text and "- Countdown: 10 s" in text


def test_embedder_caches_ledgers_and_refuses_offline_misses(tmp_path: Path) -> None:
    from kaful.config import IngestSettings, LLMSettings

    def embedder(transport: FakeEmbeddings | None, budget: float = 1.0) -> Embedder:
        return Embedder(settings=IngestSettings(embedding_dimensions=4, embedding_batch_size=2), llm=LLMSettings(),
                        ledger=Ledger(tmp_path / "l.jsonl"), budget=RunBudget(budget), transport=transport,
                        store=VectorStore(tmp_path / "cache"))

    fake = FakeEmbeddings()
    first = embedder(fake).embed(["a", "b", "c", "a"], "ingest", "embed")
    assert first.shape == (4, 4) and fake.calls == [2, 1] and np.array_equal(first[0], first[3])
    again = embedder(None).embed(["c", "a"], "ingest", "embed")
    assert np.array_equal(again, first[[2, 0]])
    with pytest.raises(OfflineMissError):
        embedder(None).embed(["new"], "ingest", "embed")
    rows = read_ledger(tmp_path / "l.jsonl")
    assert [(r.response_cache, r.n_texts) for r in rows] == [("miss", 2), ("miss", 1), ("hit", 2)]
    from kaful.llm.budget import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        embedder(fake, budget=1e-12).embed(["x " * 1000], "ingest", "embed")


def test_long_chunks_are_embedded_in_windows_never_cut(tmp_path: Path) -> None:
    from kaful.config import IngestSettings, LLMSettings

    settings = IngestSettings(embedding_dimensions=4, embedding_max_tokens=60, embedding_window_tokens=50,
                              embedding_window_overlap=10)
    embedder = Embedder(settings=settings, llm=LLMSettings(), ledger=Ledger(tmp_path / "l.jsonl"),
                        budget=RunBudget(1.0), transport=FakeEmbeddings(), store=VectorStore(tmp_path / "cache"))
    long = " ".join(f"row{i} {i}" for i in range(80))
    vectors, rows = embedder.embed_chunks(["short text", long, "another"], "ingest", "embed")
    assert rows[0] == 0 and rows[-1] == 2 and rows.count(1) >= 3 and vectors.shape == (len(rows), 4)
    assert embedder.windowing == "cl100k_base 50/10"
    with pytest.raises(ValueError, match="60-token input limit"):  # never cut: too long for one embedding
        embedder.embed([long], "ingest", "embed")


def test_a_chunk_is_found_through_any_of_its_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(extraction_reply), FakeEmbeddings())
    index = load_index(ctx)
    assert index.vectors is not None and index.vectors.row_chunk == () and index.fingerprint.embedding_windows is None
    n = len(index.chunks)
    rows = (0, 1, 1, *range(2, n))  # chunk 1 has two windows
    split = index.model_copy(update={"vectors": index.vectors.model_copy(update={"n_rows": n + 1, "row_chunk": rows})})
    vectors = np.eye(n + 1, dtype=np.float32)
    hits = VectorIndex(split, vectors).search(vectors[2], k=1)  # the second window of chunk 1
    assert hits[0].chunk == index.chunks[1] and hits[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="every vector row"):
        VectorIndex(split, vectors[:n])


def test_ingest_records_windows_only_when_a_chunk_is_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    settings = ingest_settings(tmp_path)
    small = settings.ingest.model_copy(update={"embedding_max_tokens": 120, "embedding_window_tokens": 100,
                                               "embedding_window_overlap": 20})
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(extraction_reply), FakeEmbeddings(),
               settings=settings.model_copy(update={"ingest": small}))
    index = load_index(ctx)
    assert index.vectors is not None and index.vectors.n_rows > len(index.chunks)
    assert index.vectors.chunk_of_rows()[-1] == len(index.chunks) - 1
    assert index.fingerprint.embedding_windows == "cl100k_base 100/20"
    assert load_vectors(ctx.machine_dir, index.vectors).shape[0] == index.vectors.n_rows


def test_vector_search_ranks_by_cosine_with_stable_ties(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_pdf_machine(tmp_path)
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(extraction_reply), FakeEmbeddings())
    index = load_index(ctx)
    assert index.vectors is not None
    vectors = load_vectors(ctx.machine_dir, index.vectors)
    hits = VectorIndex(index, vectors).search(vectors[2], k=2)
    assert hits[0].chunk == index.chunks[2] and hits[0].score == pytest.approx(1.0)
    same = VectorIndex(index, np.ones_like(vectors)).search(vectors[0], k=len(index.chunks))
    assert [h.chunk.chunk_id for h in same] == sorted(c.chunk_id for c in index.chunks)


SCRIPT = """
import sys
from pathlib import Path
import pytest
from kaful import pipeline
from kaful.phases.ingest import phase as ingest_phase
from kaful.runs import dump_json
from kaful.phases.ingest.index import load_index
from tests.harness.kaful_ingest import *
from tests.harness.kaful_llm import ScriptedTransport
tmp = Path(sys.argv[1])
make_pdf_machine(tmp)
ingest_phase.read_lock = locked_extract_page
request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest",), budget_usd=1.0)
ctx, base = pipeline.start_run(ingest_settings(tmp), request, transport=ScriptedTransport(extraction_reply),
                               embedding_transport=FakeEmbeddings())
pipeline.execute(ctx, ("ingest",), base)
index = load_index(ctx).model_copy(update={"run_id": "x"})
sys.stdout.write(dump_json(index.model_copy(update={"vectors": index.vectors.model_copy(update={"path": "x"})})))
"""


def test_index_is_identical_across_hash_seeds(tmp_path: Path) -> None:
    outputs = set()
    for seed in ("0", "1", "4242"):
        env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
        env["PYTHONHASHSEED"] = seed
        out = tmp_path / seed
        done = subprocess.run([sys.executable, "-B", "-c", SCRIPT, str(out)], cwd=REPO_ROOT, env=env,
                              capture_output=True, text=True, check=True)
        outputs.add(done.stdout)
    assert len(outputs) == 1 and '"kind": "document_index"' in outputs.pop()


def test_new_fingerprint_fields_change_the_index_hash_only_when_set() -> None:
    import hashlib
    import json as _json

    from kaful.artifacts.document_index import IndexFingerprint, index_hash_for

    base = IndexFingerprint(classifier_version="c", chunker_version="k", extraction=None, embedding_model="e",
                            embedding_dimensions=8)
    before_the_field = {k: v for k, v in base.model_dump(mode="json").items()
                        if k not in {"llm_ruled_tables", "embedding_windows"}}
    blob = _json.dumps({"chunks": [], "fingerprint": before_the_field}, sort_keys=True, separators=(",", ":"))
    assert index_hash_for((), base) == hashlib.sha256(blob.encode()).hexdigest()  # old indexes keep their hash
    assert index_hash_for((), base.model_copy(update={"llm_ruled_tables": ("fanuc",)})) != index_hash_for((), base)
    assert index_hash_for((), base.model_copy(update={"embedding_windows": "cl100k_base 8000/200"})) != (
        index_hash_for((), base))


def test_a_reused_older_response_is_recorded_on_the_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kaful.llm.gateway import Gateway, LLMResult

    real_call = Gateway.call

    def older(self: Gateway, call: Any) -> LLMResult:  # page 4's response comes from an older version
        result = real_call(self, call)
        if call.scope.component == "tp_manual:p4":
            return LLMResult(result.output, result.text, result.prompt_sha256, 1, 0.0, (2, "b" * 64))
        return result

    monkeypatch.setattr(Gateway, "call", older)
    monkeypatch.setattr(Gateway, "run_many", lambda self, calls, mode=None: [self.call(c) for c in calls])
    make_pdf_machine(tmp_path)
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(extraction_reply), FakeEmbeddings())
    record = ctx.manifest.phase("ingest")
    page4 = load_index(ctx).pages[3]
    layer = next(x for x in page4.layers if x.kind == "llm")
    assert layer.prompt_sha256 == "b" * 64 and layer.producer.startswith("ingest.extract_page v2 (reused under v")
    assert record.notes == (f"1 page(s): {layer.producer}",)
