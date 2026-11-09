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
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kaful.artifacts.document_index import Chunk, DocumentIndex
from kaful.artifacts.understand_report import ContextSelection
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


def select_context(index: DocumentIndex, vectors: Vectors | None, embedder: Embedder | None,
                   settings: UnderstandSettings, ingest: IngestSettings,
                   count: Callable[[str], int] | None = None, kinds: Sequence[str] | None = None,
                   topics: Sequence[str] | None = None) -> Context:
    """The context of the documents of `kinds` (default all), asked with the queries of `topics` (default all)."""
    docs = tuple(sorted(d.doc_id for d in index.documents if kinds is None or d.kind in kinds))
    if not docs:
        raise RuntimeError(f"no document of kind {sorted(kinds or ())} in the index")
    own = [c for c in index.chunks if c.doc_id in docs]
    whole = render(own)
    asked: tuple[str, ...] = ()
    if whole_fits(whole, settings, count):
        chunks, mode = document_order(own), "whole"
    else:
        if vectors is None or embedder is None:
            raise RuntimeError("the index is larger than the context budget; retrieval needs vectors and an embedder")
        asked = queries(settings, topics)
        chunks, mode = document_order(_retrieve(index, vectors, embedder, asked, settings, ingest, docs)), "retrieval"
    text = render(chunks)
    selection = ContextSelection(mode=mode, chunk_ids=tuple(c.chunk_id for c in chunks),
                                 estimated_tokens=estimated_tokens(text, settings), queries=asked, documents=docs,
                                 context_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return Context(selection, text)
