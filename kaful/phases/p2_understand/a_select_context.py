"""Step a: select the documentation context the understand prompts see.

If the whole index fits `context_max_tokens`, every chunk is used. Otherwise each
configured query (topics and queries in declared order) retrieves its top-k chunks;
chunks are ranked by their best score (ties by chunk id) and added until the budget is
full. Either way the prompt lists chunks in document order (doc_id, page_start,
chunk_id), never in retrieval or set order (hard rule 14; this is the legacy 3a bug).
"""

import hashlib
from collections.abc import Sequence
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


def estimated_tokens(text: str, ingest: IngestSettings) -> int:
    return int(len(text) / ingest.chunk_chars_per_token) + 1


def document_order(chunks: Sequence[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=lambda c: (c.doc_id, c.page_start, c.chunk_id))


def render(chunks: Sequence[Chunk]) -> str:
    return CHUNK_SEPARATOR.join(c.text.strip() for c in document_order(chunks))


def queries(settings: UnderstandSettings) -> tuple[str, ...]:
    """Every configured query once, topics in sorted order, queries in declared order."""
    seen: list[str] = []
    for topic in sorted(settings.queries):
        seen += [q for q in settings.queries[topic] if q not in seen]
    return tuple(seen)


def _retrieve(index: DocumentIndex, vectors: Vectors, embedder: Embedder, asked: tuple[str, ...],
              settings: UnderstandSettings, ingest: IngestSettings) -> list[Chunk]:
    search = VectorIndex(index, vectors)
    best: dict[str, tuple[float, Chunk]] = {}
    for qvec in embedder.embed(list(asked), phase=PHASE, step="retrieve_context"):
        for hit in search.search(qvec, settings.retrieval_top_k):
            if hit.chunk.chunk_id not in best or hit.score > best[hit.chunk.chunk_id][0]:
                best[hit.chunk.chunk_id] = (hit.score, hit.chunk)
    ranked = sorted(best.values(), key=lambda sc: (-round(sc[0], 6), sc[1].chunk_id))
    picked: list[Chunk] = []
    used = 0
    for _, chunk in ranked:
        cost = estimated_tokens(chunk.text, ingest)
        if used + cost > settings.context_max_tokens:
            continue
        picked.append(chunk)
        used += cost
    return picked


def select_context(index: DocumentIndex, vectors: Vectors | None, embedder: Embedder | None,
                   settings: UnderstandSettings, ingest: IngestSettings) -> Context:
    whole = render(index.chunks)
    asked: tuple[str, ...] = ()
    if estimated_tokens(whole, ingest) <= settings.context_max_tokens:
        chunks, mode = document_order(index.chunks), "whole"
    else:
        if vectors is None or embedder is None:
            raise RuntimeError("the index is larger than the context budget; retrieval needs vectors and an embedder")
        asked = queries(settings)
        chunks, mode = document_order(_retrieve(index, vectors, embedder, asked, settings, ingest)), "retrieval"
    text = render(chunks)
    selection = ContextSelection(mode=mode, chunk_ids=tuple(c.chunk_id for c in chunks),
                                 estimated_tokens=estimated_tokens(text, ingest), queries=asked,
                                 context_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())
    return Context(selection, text)
