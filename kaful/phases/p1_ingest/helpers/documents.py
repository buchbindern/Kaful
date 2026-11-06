"""The document being ingested, as it passes between steps.

Used by steps c_classify_pages (builds it), d_add_ruled_tables, e_extract_hard_pages and
f_chunk_pages (read and return it), and by tools/read_document.
"""

from dataclasses import dataclass

from kaful.artifacts.document_index import DocumentRecord, PageText


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
