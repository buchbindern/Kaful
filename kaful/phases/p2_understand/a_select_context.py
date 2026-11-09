"""Step a: select the documentation context each understand prompt sees.

A prompt's context is drawn from the documents of the kinds it may use (components and
channels: the machine documents, `machine_kinds`; fault modes: every document) and asked with
its own topic's queries. If those documents fit `whole_max_tokens` (counted with `count`, the API's token count, when
given; else estimated), every chunk is used. Otherwise each configured query (topics and
queries in declared order) retrieves its top-k chunks from every document, so one large manual
cannot crowd out a small catalogue; chunks are added in turns across documents (every
document's best chunk before any document's second; then by score, ties by chunk id) until
`context_max_tokens` is full. Either way the prompt lists chunks in document order (doc_id,
page_start, chunk_id), never in retrieval or set order (hard rule 14; this is the legacy 3a bug).

The machine documents are read in full (M7c, approved 2026-11-09; `select_parts`): when they do not fit
`whole_max_tokens`, they are split into parts of at most `part_max_tokens` (estimated) in document order.
A part holds whole TOC sections (a run of one document's chunks with one `section`; a document without a
TOC is one run); a section larger than a part splits between chunks, and a chunk is never cut. When they
fit, the one part is exactly the whole context. Reference documents stay retrieved (`select_context`).
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import groupby

from kaful.artifacts.document_index import Chunk, DocumentIndex
from kaful.artifacts.understand_report import ContextMode, ContextSelection
from kaful.config import IngestSettings, UnderstandSettings
from kaful.llm.embeddings import Embedder, Vectors
from kaful.retrieval.search import VectorIndex

PHASE = "understand"
CHUNK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class Context:
    selection: ContextSelection
    text: str


def estimated_tokens(text: str, settings: UnderstandSettings) -> int:
    """Context size in tokens, from characters (calibrated against count_tokens; see the setting)."""
    return int(len(text) / settings.context_chars_per_token) + 1


def document_order(chunks: Sequence[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda c: (c.doc_id, c.page_start, c.chunk_id))


def render(chunks: Sequence[Chunk]) -> str:
    return CHUNK_SEPARATOR.join(c.text.strip() for c in document_order(chunks))


def queries(settings: UnderstandSettings, topics: Sequence[str] | None = None) -> tuple[str, ...]:
    """Every configured query (of `topics`, default all) once, topics in sorted order, queries in declared order."""
    seen: list[str] = []
    for topic in sorted(t for t in settings.queries if topics is None or t in topics):
        seen += [q for q in settings.queries[topic] if q not in seen]
    return tuple(seen)


def _retrieve(index: DocumentIndex, vectors: Vectors, embedder: Embedder, asked: tuple[str, ...],
              settings: UnderstandSettings, ingest: IngestSettings, docs: Sequence[str]) -> list[Chunk]:
    """Each query's top-k chunks from every document (so one large manual cannot crowd out a small
    catalogue), then added in turns, every document's best chunk before any document's second."""
    search = VectorIndex(index, vectors)
    best: dict[str, tuple[float, Chunk]] = {}
    for qvec in embedder.embed(list(asked), phase=PHASE, step="retrieve_context"):
        for doc_id in docs:
            for hit in search.search(qvec, settings.retrieval_top_k, doc_id):
                if hit.chunk.chunk_id not in best or hit.score > best[hit.chunk.chunk_id][0]:
                    best[hit.chunk.chunk_id] = (hit.score, hit.chunk)
    ranked = sorted(best.values(), key=lambda sc: (sc[1].doc_id, -round(sc[0], 6), sc[1].chunk_id))
    turn: dict[str, int] = {}
    order = []
    for score, chunk in ranked:
        turn[chunk.doc_id] = turn.get(chunk.doc_id, -1) + 1
        order.append((turn[chunk.doc_id], -round(score, 6), chunk.chunk_id, chunk))
    picked: list[Chunk] = []
    used = 0
    for *_, chunk in sorted(order, key=lambda t: t[:3]):
        cost = estimated_tokens(chunk.text, settings)
        if used + cost > settings.context_max_tokens:
            continue
        picked.append(chunk)
        used += cost
    return picked


def whole_fits(whole: str, settings: UnderstandSettings, count: Callable[[str], int] | None) -> bool:
    limit = settings.context_max_tokens if settings.whole_max_tokens is None else settings.whole_max_tokens
    return (count(whole) if count is not None else estimated_tokens(whole, settings)) <= limit


def _context(chunks: Sequence[Chunk], mode: ContextMode, docs: tuple[str, ...], settings: UnderstandSettings,
             asked: tuple[str, ...] = ()) -> Context:
    text = render(chunks)
    selection = ContextSelection(mode=mode, chunk_ids=tuple(c.chunk_id for c in document_order(chunks)),
                                 estimated_tokens=estimated_tokens(text, settings), queries=asked, documents=docs,
                                 context_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return Context(selection, text)


def _documents(index: DocumentIndex, kinds: Sequence[str] | None) -> tuple[str, ...]:
    docs = tuple(sorted(d.doc_id for d in index.documents if kinds is None or d.kind in kinds))
    if not docs:
        raise RuntimeError(f"no document of kind {sorted(kinds or ())} in the index")
    return docs


def _cost(chunks: Sequence[Chunk], settings: UnderstandSettings) -> int:
    """Estimated tokens of chunks rendered together (each chunk's estimate rounds up, so never below)."""
    return sum(estimated_tokens(c.text.strip() + CHUNK_SEPARATOR, settings) for c in chunks)


def split_parts(chunks: Sequence[Chunk], settings: UnderstandSettings) -> list[list[Chunk]]:
    """Chunks in document order, packed into parts of at most `part_max_tokens`: whole TOC sections where they
    fit, a larger section split between chunks; a chunk larger than a part is a part of its own."""
    limit = settings.part_max_tokens
    units: list[list[Chunk]] = []
    for _, run in groupby(document_order(chunks), key=lambda c: (c.doc_id, c.section)):
        section = list(run)
        if _cost(section, settings) <= limit:
            units.append(section)
            continue
        current: list[Chunk] = []
        for chunk in section:
            if current and _cost([*current, chunk], settings) > limit:
                units.append(current)
                current = []
            current.append(chunk)
        units.append(current)
    parts: list[list[Chunk]] = []
    for unit in units:
        if parts and _cost([*parts[-1], *unit], settings) <= limit:
            parts[-1].extend(unit)
        else:
            parts.append(list(unit))
    return parts


def select_parts(index: DocumentIndex, settings: UnderstandSettings, count: Callable[[str], int] | None = None,
                 kinds: Sequence[str] | None = None) -> list[Context]:
    """The documents of `kinds` in full: one whole context if it fits `whole_max_tokens`, else parts."""
    docs = _documents(index, kinds)
    own = [c for c in index.chunks if c.doc_id in docs]
    if whole_fits(render(own), settings, count):
        return [_context(own, "whole", docs, settings)]
    return [_context(part, "part", tuple(sorted({c.doc_id for c in part})), settings)
            for part in split_parts(own, settings)]


def select_context(index: DocumentIndex, vectors: Vectors | None, embedder: Embedder | None,
                   settings: UnderstandSettings, ingest: IngestSettings,
                   count: Callable[[str], int] | None = None, kinds: Sequence[str] | None = None,
                   topics: Sequence[str] | None = None) -> Context:
    """The context of the documents of `kinds` (default all), asked with the queries of `topics` (default all)."""
    docs = _documents(index, kinds)
    own = [c for c in index.chunks if c.doc_id in docs]
    if whole_fits(render(own), settings, count):
        return _context(own, "whole", docs, settings)
    if vectors is None or embedder is None:
        raise RuntimeError("the index is larger than the context budget; retrieval needs vectors and an embedder")
    asked = queries(settings, topics)
    return _context(_retrieve(index, vectors, embedder, asked, settings, ingest, docs), "retrieval", docs, settings,
                    asked)
