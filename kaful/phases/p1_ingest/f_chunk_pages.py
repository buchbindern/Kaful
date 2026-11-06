"""Chunker: whole pages, packed within one TOC section up to a token cap.

Every page with any text (raw or a layer) is chunked, whatever its class: a part-divider
page classed `blank` still carries its title. Only empty pages are left out. A chunk
never crosses a document, a TOC section start (levels <= `toc_max_level`) or an empty
page, and never splits a page. A page larger than the cap is a chunk by itself.
Chunk text is a header line, then per page a `--- page N (label L) ---` marker, its raw
text, and each non-empty extra layer marked with what produced it. Bump CHUNKER_VERSION
whenever the packing or the text format changes: it is part of every chunk id.
"""

from collections.abc import Sequence

from kaful.artifacts.document_index import Chunk, PageText, chunk_id_for, text_sha256
from kaful.config import IngestSettings
from kaful.phases.p1_ingest.b_read_pages import ReadDocument, TocEntry

CHUNKER_VERSION = "chunker-v2"  # v2: pages classed blank are chunked when they have text


def has_text(page: PageText) -> bool:
    return bool(page.text.strip()) or any(layer.text.strip() for layer in page.layers)


def page_block(page: PageText) -> str:
    marker = f"--- page {page.page}" + (f" (label {page.label})" if page.label else "") + " ---"
    parts = [marker, page.text.strip()]
    parts += [f"[{layer.kind} layer: {layer.producer}]\n{layer.text.strip()}" for layer in page.layers if layer.text.strip()]
    return "\n".join(part for part in parts if part)


def _sections(toc: Sequence[TocEntry], max_level: int) -> dict[int, str]:
    """Section title per start page (the last entry wins when several start on one page)."""
    starts: dict[int, str] = {}
    for entry in toc:
        if entry.level <= max_level:
            starts[entry.page] = entry.title
    return starts


def _groups(document: ReadDocument, settings: IngestSettings) -> list[list[PageText]]:
    starts = _sections(document.toc, settings.toc_max_level)
    cap_chars = settings.chunk_max_tokens * settings.chunk_chars_per_token
    groups: list[list[PageText]] = []
    current: list[PageText] = []
    size = 0
    for page in document.pages:
        if not has_text(page):
            current, size = ([], 0) if current else (current, size)
            continue
        block = len(page_block(page))
        if current and (page.page in starts or size + block > cap_chars):
            current, size = [], 0
        if not current:
            groups.append(current)
        current.append(page)
        size += block
    return groups


def chunk_document(document: ReadDocument, settings: IngestSettings) -> list[Chunk]:
    starts = _sections(document.toc, settings.toc_max_level)
    section_at: dict[int, str] = {}
    title: str | None = None
    for page in document.pages:
        title = starts.get(page.page, title)
        if title is not None:
            section_at[page.page] = title
    chunks = []
    for group in _groups(document, settings):
        first, last = group[0], group[-1]
        section = section_at.get(first.page)
        header = f"[{document.record.doc_id}" + (f" | {section}" if section else "") + "]"
        text = "\n".join([header, *(page_block(p) for p in group)])
        chunks.append(Chunk(
            chunk_id=chunk_id_for(document.record.sha256, first.page, last.page, CHUNKER_VERSION),
            doc_id=document.record.doc_id, page_start=first.page, page_end=last.page, label_start=first.label,
            label_end=last.label, section=section, text=text, text_sha256=text_sha256(text),
            chunker_version=CHUNKER_VERSION,
        ))
    return chunks


def coverage_gaps(pages: Sequence[PageText], chunks: Sequence[Chunk]) -> list[str]:
    """Pages with text in no chunk, or in more than one (empty when coverage is complete)."""
    counts: dict[tuple[str, int], int] = {}
    for chunk in chunks:
        for number in range(chunk.page_start, chunk.page_end + 1):
            counts[(chunk.doc_id, number)] = counts.get((chunk.doc_id, number), 0) + 1
    gaps = []
    for page in pages:
        n = counts.get((page.doc_id, page.page), 0)
        if has_text(page) and n != 1:
            gaps.append(f"{page.doc_id} p{page.page}: in {n} chunks")
    return gaps
