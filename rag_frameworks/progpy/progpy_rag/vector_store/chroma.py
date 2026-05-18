import hashlib
import struct
import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction, Embeddings, Documents
from typing import Optional

from .base import VectorStore, Chunk


class _OfflineEmbeddingFunction(EmbeddingFunction):
    """
    Hash-based embedding for offline dev/testing only.
    NOT suitable for production — semantic similarity will be poor.
    Replace with OpenAI embeddings (set OPENAI_API_KEY) for real use.
    """
    DIM = 384

    def __call__(self, input: Documents) -> Embeddings:
        result = []
        for text in input:
            # Deterministic pseudo-embedding from text hash
            h = hashlib.sha256(text.encode()).digest()
            # Repeat hash bytes to fill DIM floats
            raw = (h * ((self.DIM * 4 // len(h)) + 1))[: self.DIM * 4]
            floats = list(struct.unpack(f"{self.DIM}f", raw))
            # Normalize to unit vector
            magnitude = sum(x * x for x in floats) ** 0.5 or 1.0
            result.append([x / magnitude for x in floats])
        return result


class ChromaDBStore(VectorStore):
    """
    Local ChromaDB implementation of VectorStore.

    Uses OpenAI text-embedding-3-small when OPENAI_API_KEY is set.
    Falls back to an offline hash-based embedding for dev/testing.

    Swap guide (when moving to production):
        Replace this class with PineconeStore(VectorStore) in your pipeline.
        The interface is identical — upsert, query, count, delete_collection.
    """

    def __init__(
        self,
        collection_name: str = "progpy",
        persist_dir: str = "./data/chroma",
        openai_api_key: Optional[str] = None,
    ):
        self._client = chromadb.PersistentClient(path=persist_dir)

        if openai_api_key:
            self._ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small",
            )
            print("  Embeddings: OpenAI text-embedding-3-small")
        else:
            self._ef = _OfflineEmbeddingFunction()
            print("  Embeddings: offline hash-based (dev mode — set OPENAI_API_KEY for production)")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def upsert(self, chunks: list[Chunk]) -> None:
        """Idempotent upsert — safe to re-run the ingestion pipeline."""
        if not chunks:
            return

        for chunk in chunks:
            chunk.validate()

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(
        self,
        text: str,
        filter: Optional[dict] = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Semantic search with optional metadata pre-filter.

        filter examples:
            {"framework": "progpy"}
            {"framework": "progpy", "pattern": "composite"}
            {"$and": [{"framework": "progpy"}, {"domain": "thermal"}]}
        """
        kwargs = {
            "query_texts": [text],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter:
            # ChromaDB requires $and operator for multiple conditions
            if len(filter) > 1:
                kwargs["where"] = {"$and": [{k: v} for k, v in filter.items()]}
            else:
                kwargs["where"] = filter

        results = self._collection.query(**kwargs)

        chunks = []
        for doc, meta, dist, id_ in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunk = Chunk(text=doc, metadata=meta, chunk_id=id_)
            chunk.metadata["_score"] = round(1 - dist, 4)
            chunks.append(chunk)

        return chunks

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection.name)
        print(f"  Deleted collection: {self._collection.name}")