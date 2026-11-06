"""Step g: embed the chunks, fingerprint the index, and save vectors and DocumentIndex.

A chunk over the embedding model's input limit is embedded in overlapping windows, never
cut (kaful.llm.embeddings). The fingerprint records everything besides chunk text that
changes what the index returns; the index hash covers both (kaful.artifacts.document_index).
Loading and searching a saved index is kaful.retrieval.
"""

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.document_index import (
    Chunk,
    DocumentIndex,
    DocumentRecord,
    IndexFingerprint,
    PageText,
    VectorsRef,
    index_hash_for,
)
from kaful.llm.embeddings import Embedder, Vectors
from kaful.paths import to_rel
from kaful.phases import PhaseName
from kaful.phases.p1_ingest.c_classify_pages import CLASSIFIER_VERSION
from kaful.phases.p1_ingest.f_chunk_pages import CHUNKER_VERSION
from kaful.runs import RunContext, sha256_bytes

VECTORS_FILE = "embeddings.npy"
PHASE: PhaseName = "ingest"


def save_vectors(path: Path, vectors: Vectors) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, vectors.astype(np.float32))
    data = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def embed_and_index(ctx: RunContext, embedder: Embedder, records: Sequence[DocumentRecord],
                    pages: Sequence[PageText], chunks: Sequence[Chunk],
                    extraction: str | None) -> tuple[DocumentIndex, ArtifactRef]:
    """Embed `chunks` (sorted by chunk id), save the vectors and the index; return both."""
    vectors, rows = embedder.embed_chunks([c.text for c in chunks], phase=PHASE, step="embed_chunks")
    split = len(rows) > len(chunks)
    path = ctx.run_dir / PHASE / VECTORS_FILE
    data = save_vectors(path, vectors)
    fingerprint = IndexFingerprint(classifier_version=CLASSIFIER_VERSION, chunker_version=CHUNKER_VERSION,
                                   extraction=extraction, embedding_model=embedder.model,
                                   embedding_dimensions=embedder.dimensions,
                                   llm_ruled_tables=tuple(sorted(r.doc_id for r in records if r.ruled_tables == "llm")),
                                   embedding_windows=embedder.windowing if split else None)
    index = DocumentIndex(
        run_id=ctx.manifest.run_id, machine_id=ctx.machine.machine_id,
        index_hash=index_hash_for(tuple(chunks), fingerprint), documents=tuple(records), pages=tuple(pages),
        chunks=tuple(chunks), fingerprint=fingerprint,
        vectors=VectorsRef(path=to_rel(ctx.machine_dir, path), sha256=sha256_bytes(data), model=embedder.model,
                           dimensions=embedder.dimensions, n_rows=len(rows), row_chunk=rows if split else ()),
    )
    return index, ctx.save_artifact(PHASE, index)
