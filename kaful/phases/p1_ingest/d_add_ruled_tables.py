"""Step d: ruled tables as markdown, from PyMuPDF's `find_tables` (free and deterministic).

The raw text layer keeps a ruled table's cells but loses which row and column each
belongs to; the markdown layer restores that without an LLM call. It is added to every
page with a good ruled table, whatever its class.

Helpers: helpers/page_signals (ruled_tables).
"""

from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import PageLayer
from kaful.config import ClassifierSettings
from kaful.paths import resolve_in
from kaful.phases.p1_ingest.helpers.documents import ReadDocument
from kaful.phases.p1_ingest.helpers.page_signals import ruled_tables

TABLE_PRODUCER = f"pymupdf {fitz.VersionBind} find_tables"
TABLE_SEPARATOR = "\n\n"


def table_layer(page: fitz.Page, settings: ClassifierSettings) -> PageLayer | None:
    """The page's good ruled tables as one markdown layer, or None if it has none."""
    tables = [t.to_markdown(clean=False).strip() for t in ruled_tables(page, settings)]
    tables = [t for t in tables if t]
    if not tables:
        return None
    return PageLayer(kind="table_md", text=TABLE_SEPARATOR.join(tables), producer=TABLE_PRODUCER)


def add_ruled_tables(machine_dir: Path, document: ReadDocument, settings: ClassifierSettings) -> ReadDocument:
    """The document with a `table_md` layer on every page whose signals found a ruled table."""
    wanted = [i for i, page in enumerate(document.pages) if page.signals.ruled_tables]
    if not wanted:
        return document
    pages = list(document.pages)
    with fitz.open(resolve_in(machine_dir, document.record.path)) as doc:
        for i in wanted:
            layer = table_layer(doc[pages[i].page - 1], settings)
            if layer is not None:
                pages[i] = pages[i].model_copy(update={"layers": (layer,)})
    return ReadDocument(document.record, tuple(pages), document.toc)
