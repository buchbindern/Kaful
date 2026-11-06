"""Raw page text, classifier signals and printed page labels, all from PyMuPDF.

Raw text is `page.get_text("text")`, unmodified: evidence quotes are checked against
it (hard rule 4). Nothing here calls a model.

Used by steps b_read_pages (raw text, signals, printed labels) and d_add_ruled_tables
(ruled_tables), and by tools/commands (classifier-eval preview).
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import LabelSource, PageSignals
from kaful.config import ClassifierSettings, IngestSettings

# Characters that mean the text layer is broken: replacement char, private-use area, controls.
GARBLE_RE = re.compile(r"[�-\x00-\x08\x0b\x0c\x0e-\x1f]")
CID_RE = re.compile(r"\(cid:\d+\)")
CID_WEIGHT = 5
NUMERIC_LINE_RE = re.compile(r"^[-–—+±.,:;%/()\[\]\d\s]+$")
SHORT_LINE_WORDS = 2
AXIS_TOLERANCE = 0.5  # points; a line closer than this to horizontal/vertical is a table rule
# Printed page numbers in a header/footer line: "12", "- 12 -", "| 12", "Page 12 of 160".
PAGE_NUMBER_RE = re.compile(r"^(?:page\s+)?(\d{1,4})(?:\s+of\s+\d{1,4})?$", re.IGNORECASE)
CHAPTER_PAGE_RE = re.compile(r"^[A-Z]?\d{1,2}[-–.]\d{1,3}$")
LABEL_TRIM = " \t|-–—•·"
# Some PDFs store /PageLabels prefixes as UTF-16 hex, e.g. '<FEFF0073002D>1' for 's-1'.
HEX_LABEL_RE = re.compile(r"<(FEFF(?:[0-9A-Fa-f]{4})*)>")
# Line numbers printed in the side margins (e.g. MTConnect) are not content lines.
MARGIN_TOLERANCE = 2.0  # points
INTEGER_RE = re.compile(r"^\d{1,4}$")


def raw_text(page: fitz.Page) -> str:
    return str(page.get_text("text"))


def _image_frac(page: fitz.Page) -> float:
    area = page.rect.width * page.rect.height
    if area <= 0:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        box = fitz.Rect(info["bbox"]) & page.rect
        covered += max(box.width, 0.0) * max(box.height, 0.0)
    return float(min(covered / area, 1.0))


def _curves(drawings: Sequence[dict[str, object]]) -> int:
    """Bezier items and diagonal lines: diagrams and plots have them, table rules do not."""
    count = 0
    for drawing in drawings:
        for item in drawing.get("items", ()):  # type: ignore[attr-defined]
            if item[0] == "c":
                count += 1
            elif item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) > AXIS_TOLERANCE and abs(p1.y - p2.y) > AXIS_TOLERANCE:
                    count += 1
    return count


def _content_lines(page: fitz.Page, band_frac: float) -> list[str]:
    """Text lines outside the header/footer bands, without margin line numbers: integer-only
    lines that sit wholly left (or right) of every other line on the page."""
    height = page.rect.height
    boxed = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            x0, y0, x1, y1 = line["bbox"]
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text and y1 > height * band_frac and y0 < height * (1 - band_frac):
                boxed.append((x0, x1, text))
    words = [(x0, x1) for x0, x1, text in boxed if not INTEGER_RE.match(text)]
    if not words:
        return [text for _, _, text in boxed]
    left, right = min(x0 for x0, _ in words), max(x1 for _, x1 in words)
    return [text for x0, x1, text in boxed
            if not (INTEGER_RE.match(text) and (x1 <= left + MARGIN_TOLERANCE or x0 >= right - MARGIN_TOLERANCE))]


def _line_stats(lines: list[str]) -> tuple[int, float, float]:
    if not lines:
        return 0, 0.0, 0.0
    numeric = sum(1 for line in lines if NUMERIC_LINE_RE.match(line))
    short = sum(1 for line in lines if len(line.split()) <= SHORT_LINE_WORDS or NUMERIC_LINE_RE.match(line))
    return len(lines), short / len(lines), numeric / len(lines)


def _garble_ratio(text: str) -> float:
    stripped = "".join(text.split())
    if not stripped:
        return 0.0
    bad = len(GARBLE_RE.findall(stripped)) + CID_WEIGHT * len(CID_RE.findall(stripped))
    return min(bad / len(stripped), 1.0)


def ruled_tables(page: fitz.Page, settings: ClassifierSettings) -> list[fitz.table.Table]:
    """Tables found by `find_tables` that pass the quality check, in reading order."""
    found = []
    for table in page.find_tables().tables:
        cells = [cell for row in table.extract() for cell in row]
        if table.row_count < settings.table_min_rows or table.col_count < settings.table_min_cols or not cells:
            continue
        empty = sum(1 for cell in cells if cell is None or not str(cell).strip())
        if empty / len(cells) <= settings.table_max_empty_frac:
            found.append(table)
    return sorted(found, key=lambda t: (round(t.bbox[1], 1), round(t.bbox[0], 1)))


def page_signals(page: fitz.Page, text: str, settings: IngestSettings) -> PageSignals:
    drawings = page.get_drawings()
    lines, short, numeric = _line_stats(_content_lines(page, settings.label_band_frac))
    return PageSignals(
        chars=len(text.strip()), lines=lines, image_frac=round(_image_frac(page), 4), drawings=len(drawings),
        curves=_curves(drawings), short_line_frac=round(short, 4), numeric_line_frac=round(numeric, 4),
        garble_ratio=round(_garble_ratio(text), 4), ruled_tables=len(ruled_tables(page, settings.classifier)),
    )


# ── Printed page labels ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class PageLabel:
    label: str | None
    source: LabelSource


def decode_label(label: str) -> str:
    """Decode UTF-16 hex runs that some PDFs leave in /PageLabels prefixes."""
    return HEX_LABEL_RE.sub(lambda m: bytes.fromhex(m.group(1)).decode("utf-16"), label)


def _band_lines(page: fitz.Page, band_frac: float) -> list[str]:
    """Text lines in the header and footer bands, top to bottom."""
    height = page.rect.height
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            y0, y1 = line["bbox"][1], line["bbox"][3]
            if y1 <= height * band_frac or y0 >= height * (1 - band_frac):
                text = "".join(span["text"] for span in line["spans"]).strip(LABEL_TRIM)
                if text:
                    lines.append((round(y0, 1), round(line["bbox"][0], 1), text))
    return [text for _, _, text in sorted(lines)]


def _candidates(lines: list[str]) -> tuple[list[int], list[str]]:
    numbers = [int(m.group(1)) for line in lines if (m := PAGE_NUMBER_RE.match(line))]
    chapter = [line for line in lines if CHAPTER_PAGE_RE.match(line)]
    return numbers, chapter


def printed_labels(doc: fitz.Document, settings: IngestSettings) -> list[PageLabel]:
    """One label per page: the PDF's /PageLabels if present, else printed numbers in the
    header/footer bands that agree with one offset from the physical page, else none."""
    pdf_labels = [decode_label(str(page.get_label() or "")) for page in doc]
    if any(pdf_labels):
        return [PageLabel(label or None, "pdf" if label else "none") for label in pdf_labels]
    per_page = [_candidates(_band_lines(page, settings.label_band_frac)) for page in doc]
    offsets = Counter(n - (i + 1) for i, (numbers, _) in enumerate(per_page) for n in sorted(set(numbers)))
    numbered = _numbered_labels(per_page, offsets, doc.page_count, settings)
    if numbered is not None:
        return numbered
    with_chapter = sum(1 for _, chapter in per_page if chapter)
    if with_chapter >= settings.label_min_pages and with_chapter / doc.page_count >= settings.label_min_share:
        return [PageLabel(chapter[0], "printed") if chapter else PageLabel(None, "none") for _, chapter in per_page]
    return [PageLabel(None, "none")] * doc.page_count


def _numbered_labels(per_page: list[tuple[list[int], list[str]]], offsets: Counter[int], n_pages: int,
                     settings: IngestSettings) -> list[PageLabel] | None:
    if not offsets:
        return None
    best = max(offsets.values())
    offset = min(o for o, count in offsets.items() if count == best)  # ties: the smallest offset
    if best < settings.label_min_pages or best / n_pages < settings.label_min_share:
        return None
    return [PageLabel(str(i + 1 + offset), "printed") if (i + 1 + offset) in numbers else PageLabel(None, "none")
            for i, (numbers, _) in enumerate(per_page)]
