"""Read one document into classified pages (raw text, label, signals, ruled-table layer).

This is the offline part of ingest: no model calls. LLM layers are added later, by
`extract`, for the pages whose class needs them, and for the table_ruled pages of a
document that sets `ruled_tables = "llm"` in machine.toml.
"""

from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import DocumentRecord, PageText
from kaful.config import IngestSettings
from kaful.paths import resolve_in
from kaful.phases.ingest.classify import classify_page
from kaful.phases.ingest.pages import page_signals, printed_labels, raw_text
from kaful.phases.ingest.tables import table_layer


@dataclass(frozen=True)
class TocEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class ReadDocument:
    record: DocumentRecord
    pages: tuple[PageText, ...]
    toc: tuple[TocEntry, ...]


def read_document(machine_dir: Path, record: DocumentRecord, settings: IngestSettings) -> ReadDocument:
    with fitz.open(resolve_in(machine_dir, record.path)) as doc:
        labels = printed_labels(doc, settings)
        pages = []
        for index, page in enumerate(doc):
            text = raw_text(page)
            signals = page_signals(page, text, settings)
            layer = table_layer(page, settings.classifier) if signals.ruled_tables else None
            page_class = classify_page(signals, settings.classifier)
            pages.append(PageText(
                doc_id=record.doc_id, page=index + 1, label=labels[index].label, label_source=labels[index].source,
                text=text, page_class=page_class, signals=signals, layers=(layer,) if layer is not None else (),
                llm_by_document=page_class == "table_ruled" and record.ruled_tables == "llm",
            ))
        toc = tuple(TocEntry(level=int(level), title=str(title).strip(), page=int(page))
                    for level, title, page, *_ in doc.get_toc(simple=True) if 1 <= int(page) <= doc.page_count)
    return ReadDocument(record, tuple(pages), toc)
