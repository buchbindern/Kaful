"""M6 grounding: quote set, matching, legacy-index reader, report and the old-only review table."""

import json
import sqlite3
import tomllib
from pathlib import Path

from kaful.artifacts.document_index import PageLayer, PageText
from kaful.config import IngestSettings
from kaful.evals.grounding import (
    CHROMA_FILE,
    Matcher,
    Passage,
    build_report,
    collect_quotes,
    old_only_review,
    read_old_index,
    tokens,
)
from tests.harness.kaful_factories import SIGNALS

SETTINGS = IngestSettings()


def _page(n: int, text: str, layer: str = "", page_class: str = "text") -> PageText:
    layers = (PageLayer(kind="llm", text=layer, producer="ingest.extract_page v1"),) if layer else ()
    return PageText(doc_id="manual", page=n, text=text, page_class=page_class, signals=SIGNALS,  # type: ignore[arg-type]
                    layers=layers)


def test_tokens_fold_case_dashes_and_line_break_hyphens() -> None:
    assert tokens("Cake Thick-\nness [mm] 14,0 – OK") == ["cake", "thickness", "mm", "14.0", "ok"]


def test_a_line_break_hyphen_joins_words_but_not_numbers() -> None:
    assert tokens("Thick-\nness") == ["thickness"]
    assert tokens("compression: 10-\n30μm") == tokens("compression: 10-30μm")  # a range stays a range
    matcher = Matcher([Passage("p38", "Suggested axial compression: 10-\n30μm.")], SETTINGS)
    assert matcher.match("Suggested axial compression: 10-30μm.") == ("verbatim", "p38")


def test_decimal_commas_equal_points_but_thousands_separators_do_not() -> None:
    assert tokens("Relax Time 2,0 s; 0,8 s; 1,25 bar") == tokens("Relax Time 2.0 s; 0.8 s; 1.25 bar")
    assert tokens("1,000 ticks") == ["1,000", "ticks"] and tokens("1,000") != tokens("1.000")
    assert tokens("12,345,678") == ["12,345,678"]  # every group of three: thousands separators
    matcher = Matcher([Passage("p69", "Relax Time; 2,0; [s]; Second Tamping; 2,0; [mm]")], SETTINGS)
    assert matcher.match("Relax Time: 2.0 [s]") == ("verbatim", "p69")
    thousands = Matcher([Passage("p72", "Adjustment range is 0 to 1,000 ticks.")], SETTINGS)
    assert thousands.match("range is 0 to 1.000 ticks")[0] != "verbatim"


def test_matcher_classes() -> None:
    matcher = Matcher([Passage("p1", "Replace the inlet cartridge filter every 2000 operating hours.")], SETTINGS)
    assert matcher.match("inlet cartridge filter every 2000") == ("verbatim", "p1")
    assert matcher.match("replace the inlet cartridge filter every 2000 operation hours") == ("near", "p1")
    assert matcher.match("the pump seal lasts five years") == ("unsupported", None)
    assert matcher.match("filter every 2,000")[0] == "unsupported"  # number formats are kept apart


def test_collect_quotes_dedupes_and_skips_snapshots(tmp_path: Path) -> None:
    for rel, quote in [("outputs/schema/a.json", "Water quantity in ticks"), ("outputs/twin/b.json", "water QUANTITY in ticks"),
                       ("outputs/schema_SNAPSHOT/c.json", "never collected here"), ("outputs/twin/d.json", "too short")]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fields": [{"evidence": quote}]}))
    quotes = collect_quotes(tmp_path, "toy", min_tokens=3)
    assert [q.text for q in quotes.quotes] == ["Water quantity in ticks"]
    assert quotes.quotes[0].sources == ("outputs/schema/a.json#.fields[0].evidence", "outputs/twin/b.json#.fields[0].evidence")


def _chroma(machine_dir: Path, rows: list[tuple[int, str, str, int, int]]) -> None:
    path = machine_dir / CHROMA_FILE
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER)")
        for row_id, text, pdf, start, end in rows:
            db.executemany("INSERT INTO embedding_metadata VALUES (?, ?, ?, ?)", [
                (row_id, "chroma:document", text, None), (row_id, "pdf_file", pdf, None),
                (row_id, "start_page", None, start), (row_id, "end_page", None, end)])


def test_old_index_reader_and_report(tmp_path: Path) -> None:
    _chroma(tmp_path, [(1, "Cake thickness 14.0 mm. Foam texture 50.", "m.pdf", 1, 2)])
    (tmp_path / "outputs" / "schema").mkdir(parents=True)
    (tmp_path / "outputs" / "schema" / "step_a_chunks.json").write_text(json.dumps([{"chunk_id": "c", "text": "saved context"}]))
    (tmp_path / "outputs" / "schema" / "q.json").write_text(json.dumps([
        {"evidence": "Cake thickness 14.0 mm"}, {"evidence": "Foam texture 50"}, {"evidence": "Rated flow 6 m3/h"}]))
    old = read_old_index(tmp_path, {"m.pdf": "manual"}, total_pages=4)
    assert old.pages == {("manual", 1), ("manual", 2)} and old.partial and len(old.passages) == 2
    pages = [_page(1, "Cake Thickness [mm]\n14,0", layer="| Cake Thickness [mm] | 14,0 |"),
             _page(2, "Settings", layer="- Foam Texture: 50"), _page(3, "Rated flow 6 m3/h"), _page(4, "", page_class="blank")]
    quotes = collect_quotes(tmp_path, "toy", min_tokens=3)
    report = build_report(quotes, "a" * 64, old, pages, SETTINGS, run_id="r1", index_hash=None,
                          chunk_pages={("manual", 1), ("manual", 2), ("manual", 3)})
    by_text = {q.text: next(r for r in report.quotes if r.quote_id == q.quote_id) for q in quotes.quotes}
    assert (by_text["Foam texture 50"].old, by_text["Foam texture 50"].new_any, by_text["Foam texture 50"].raw_only) == (
        "verbatim", "verbatim", "unsupported")
    assert by_text["Cake thickness 14.0 mm"].new_any == "unsupported"  # the manual prints 14,0
    assert report.old_only == (next(q.quote_id for q in quotes.quotes if q.text.startswith("Cake")),)
    assert report.new_coverage is not None and report.new_coverage.share == 1.0 and report.old_page_coverage == 0.5
    review = tomllib.loads(old_only_review(report, quotes, pages))
    assert [(q["quote"], q["candidate_page"], q["present"]) for q in review["quotes"]] == [
        ("Cake thickness 14.0 mm", "manual:p1", "")]
