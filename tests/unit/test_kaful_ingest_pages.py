"""Ingest building blocks: document validation, page signals, labels, classifier, tables, render."""

import struct
from pathlib import Path

import fitz  # type: ignore[import-untyped]
import pytest

from kaful.artifacts.document_index import PageSignals
from kaful.config import ClassifierSettings, IngestSettings, MachineConfig, MachineConfigError
from kaful.phases.p1_ingest.c_classify_pages import classify_page
from kaful.phases.p1_ingest.a_check_documents import DocumentError, check_documents, doc_id_for
from kaful.phases.p1_ingest.helpers.page_signals import decode_label, page_signals, printed_labels, raw_text
from kaful.phases.p1_ingest.helpers.render_pages import page_png
from kaful.phases.p1_ingest.d_add_ruled_tables import table_layer
from tests.harness.kaful_pdf import PAGE_KINDS, write_manual

SETTINGS = IngestSettings()
QUIET = PageSignals(chars=900, lines=20, image_frac=0.0, drawings=0, curves=0, short_line_frac=0.1,
                    numeric_line_frac=0.0, garble_ratio=0.0, ruled_tables=0)


@pytest.fixture(scope="module")
def manual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_manual(tmp_path_factory.mktemp("pdf") / "manual" / "TP-200 Manual.pdf")


def _config(*paths: str) -> MachineConfig:
    return MachineConfig.model_validate({"machine_id": "toy", "display_name": "Toy", "time_unit": "hour",
                                         "documents": [{"path": p, "kind": "manual"} for p in sorted(paths)]})


# ── Documents ────────────────────────────────────────────────────────────────

def test_documents_are_validated_against_disk(tmp_path: Path, manual: Path) -> None:
    machine = tmp_path / "toy"
    write_manual(machine / "manual" / "a.pdf")
    (machine / "manual" / "notes.txt").write_text("not a pdf")
    check = check_documents(machine, _config("manual/a.pdf"))
    assert [(r.doc_id, r.n_pages, r.kind) for r in check.records] == [("a", len(PAGE_KINDS), "manual")]
    assert check.warnings == ()
    with pytest.raises(DocumentError, match="not found"):
        check_documents(machine, _config("manual/a.pdf", "manual/missing.pdf"))
    with pytest.raises(DocumentError, match="not a PDF"):
        check_documents(machine, _config("manual/notes.txt"))


def test_duplicate_content_and_unlisted_pdfs(tmp_path: Path) -> None:
    machine = tmp_path / "toy"
    write_manual(machine / "manual" / "a.pdf")
    (machine / "manual" / "b.pdf").write_bytes((machine / "manual" / "a.pdf").read_bytes())
    with pytest.raises(DocumentError, match="share a sha256"):
        check_documents(machine, _config("manual/a.pdf", "manual/b.pdf"))
    check = check_documents(machine, _config("manual/a.pdf"))
    assert check.warnings == ("manual/b.pdf is under manual/ but not listed in machine.toml",)
    with pytest.raises(DocumentError, match="not listed"):
        check_documents(machine, _config("manual/a.pdf"), strict=True)


def test_doc_ids_come_from_file_names() -> None:
    assert doc_id_for("manual/TP-200 Manual.pdf") == "tp_200_manual"
    with pytest.raises(DocumentError):
        doc_id_for("manual/10816.pdf")


def test_provisional_time_unit_cannot_be_used() -> None:
    config = _config().model_copy(update={"time_unit_provisional": True})
    with pytest.raises(MachineConfigError, match="provisional"):
        config.confirmed_time_unit()
    assert _config().confirmed_time_unit() == "hour"


# ── Pages, labels, classifier ────────────────────────────────────────────────

def test_synthetic_pages_get_their_intended_class(manual: Path) -> None:
    expected = {"figure_image": "figure", "figure_curves": "figure"}
    with fitz.open(manual) as doc:
        got = [classify_page(page_signals(p, raw_text(p), SETTINGS), SETTINGS.classifier) for p in doc]
    assert got == [expected.get(kind, kind) for kind in PAGE_KINDS]


def test_printed_labels_follow_one_offset(manual: Path) -> None:
    with fitz.open(manual) as doc:
        labels = printed_labels(doc, SETTINGS.model_copy(update={"label_min_pages": 3}))
    assert [lab.label for lab in labels] == ["1", "2", "3", "4", "5", None]
    assert {lab.source for lab in labels[:5]} == {"printed"}


def test_pdf_page_labels_win_and_hex_prefixes_decode(tmp_path: Path) -> None:
    path = write_manual(tmp_path / "m.pdf", kinds=("text", "text"))
    with fitz.open(path) as doc:
        doc.set_page_labels([{"startpage": 0, "prefix": "s-", "style": "D", "firstpagenum": 1}])
        labels = printed_labels(doc, SETTINGS)
    assert [(lab.label, lab.source) for lab in labels] == [("s-1", "pdf"), ("s-2", "pdf")]
    assert decode_label("<FEFF0073002D>1") == "s-1"


def test_no_labels_without_agreement(tmp_path: Path) -> None:
    path = write_manual(tmp_path / "m.pdf", kinds=("text", "text"), footers=False)
    with fitz.open(path) as doc:
        assert {lab.source for lab in printed_labels(doc, SETTINGS)} == {"none"}


@pytest.mark.parametrize(("update", "expected"), [
    ({}, "text"),
    ({"chars": 10, "lines": 1}, "blank"),
    ({"chars": 10, "lines": 1, "image_frac": 0.9}, "scanned"),
    ({"garble_ratio": 0.05}, "garbled"),
    ({"short_line_frac": 0.8, "numeric_line_frac": 0.3}, "table_unruled"),
    ({"short_line_frac": 0.8, "numeric_line_frac": 0.3, "ruled_tables": 1}, "table_ruled"),
    ({"image_frac": 0.1}, "figure"),
    ({"curves": 150}, "figure"),
    ({"curves": 149}, "text"),
    ({"image_frac": 0.1, "ruled_tables": 2}, "figure"),
])
def test_classifier_rules_in_order(update: dict[str, float], expected: str) -> None:
    assert classify_page(QUIET.model_copy(update=update), ClassifierSettings()) == expected


def test_margin_line_numbers_are_not_content_lines(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    for i in range(20):
        page.insert_text((40, 100 + 20 * i), str(300 + i), fontsize=8)
        page.insert_text((90, 100 + 20 * i), "definition text of an element in the information model", fontsize=9)
    signals = page_signals(page, raw_text(page), SETTINGS)
    assert (signals.lines, signals.numeric_line_frac) == (20, 0.0)


# ── Tables and render ────────────────────────────────────────────────────────

def test_ruled_table_markdown_keeps_rows(manual: Path) -> None:
    with fitz.open(manual) as doc:
        layer = table_layer(doc[1], SETTINGS.classifier)
        assert table_layer(doc[2], SETTINGS.classifier) is None
    assert layer is not None and layer.kind == "table_md" and layer.producer.startswith("pymupdf ")
    assert "|Rated flow|6,0|m3/h|" in layer.text


def test_render_is_deterministic_and_within_the_long_edge(manual: Path) -> None:
    settings = SETTINGS.model_copy(update={"render_long_edge_px": 800})
    with fitz.open(manual) as doc:
        first = page_png(doc[0], settings)
    with fitz.open(manual) as doc:
        assert page_png(doc[0], settings) == first
    width, height = struct.unpack(">II", first[16:24])
    assert first[:8] == b"\x89PNG\r\n\x1a\n" and max(width, height) <= 800
