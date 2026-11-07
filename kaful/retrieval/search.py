"""Vector search over a document index's chunks.

`VectorIndex.search` ranks chunks by cosine similarity; ties go to the smaller chunk id,
so results never depend on load order. A chunk embedded in several windows scores its
best window, so every part of it is searchable; results are whole chunks.
"""

from dataclasses import dataclass

import numpy as np

from kaful.artifacts.document_index import Chunk, DocumentIndex
from kaful.llm.embeddings import Vectors


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, index: DocumentIndex, vectors: Vectors) -> None:
        rows = index.vectors.chunk_of_rows() if index.vectors is not None else tuple(range(len(index.chunks)))
        if vectors.shape[0] != len(rows) or len(set(rows)) != len(index.chunks):
            raise ValueError("every vector row must map to a chunk, and every chunk needs a row")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.chunks = index.chunks
        self.rows = np.asarray(rows, dtype=np.int64)
        self.unit = vectors / np.where(norms == 0, 1.0, norms)

    def search(self, query: Vectors, k: int, doc_id: str | None = None) -> list[Hit]:
        """The k best chunks (of one document when `doc_id` is given)."""
        q = query.reshape(-1).astype(np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = np.full(len(self.chunks), -np.inf, dtype=np.float64)
        np.maximum.at(scores, self.rows, (self.unit @ q).astype(np.float64))
        candidates = [i for i in range(len(self.chunks)) if doc_id is None or self.chunks[i].doc_id == doc_id]
        order = sorted(candidates, key=lambda i: (-round(float(scores[i]), 6), self.chunks[i].chunk_id))
        return [Hit(self.chunks[i], float(scores[i])) for i in order[:k]]
