"""Ruled tables as markdown, from PyMuPDF's `find_tables` (free and deterministic).

The raw text layer keeps a ruled table's cells but loses which row and column each
belongs to; the markdown layer restores that without an LLM call.
"""

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import PageLayer
from kaful.config import ClassifierSettings
from kaful.phases.ingest.pages import ruled_tables

TABLE_PRODUCER = f"pymupdf {fitz.VersionBind} find_tables"
TABLE_SEPARATOR = "\n\n"


def table_layer(page: fitz.Page, settings: ClassifierSettings) -> PageLayer | None:
    """The page's good ruled tables as one markdown layer, or None if it has none."""
    tables = [t.to_markdown(clean=False).strip() for t in ruled_tables(page, settings)]
    tables = [t for t in tables if t]
    if not tables:
        return None
    return PageLayer(kind="table_md", text=TABLE_SEPARATOR.join(tables), producer=TABLE_PRODUCER)
