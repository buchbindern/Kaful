"""Step b: read every page of one document (offline, no model calls).

Per page: the raw PyMuPDF text layer (kept unchanged), the printed page label, and the
signals the classifier decides from; per document: the table of contents. Classifying
(step c) and ruled-table markdown (step d) come next.

Helpers: helpers/page_signals (raw text, signals, printed labels).
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import DocumentRecord, LabelSource, PageSignals
from kaful.config import IngestSettings
from kaful.paths import resolve_in
from kaful.phases.p1_ingest.helpers.documents import TocEntry
from kaful.phases.p1_ingest.helpers.page_signals import (
    page_signals,
    printed_labels,
    raw_text,
)


@dataclass(frozen=True)
class RawPage:
    page: int
    label: str | None
    label_source: LabelSource
    text: str
    signals: PageSignals


@dataclass(frozen=True)
class RawDocument:
    record: DocumentRecord
    pages: tuple[RawPage, ...]
    toc: tuple[TocEntry, ...]


def read_pages(machine_dir: Path, record: DocumentRecord, settings: IngestSettings) -> RawDocument:
    with fitz.open(resolve_in(machine_dir, record.path)) as doc:
        labels = printed_labels(doc, settings)
        pages = []
        for index, page in enumerate(doc):
            text = raw_text(page)
            pages.append(RawPage(page=index + 1, label=labels[index].label, label_source=labels[index].source,
                                 text=text, signals=page_signals(page, text, settings)))
        toc = tuple(TocEntry(level=int(level), title=str(title).strip(), page=int(page))
                    for level, title, page, *_ in doc.get_toc(simple=True) if 1 <= int(page) <= doc.page_count)
    return RawDocument(record, tuple(pages), toc)
