from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """
    A single unit of knowledge to be stored and retrieved.

    text        : the content that gets embedded and returned at retrieval time
    metadata    : filtering dimensions — framework, domain, type, pattern, source
    chunk_id    : stable unique ID (used for upserts so re-indexing is idempotent)
    """
    text: str
    metadata: dict
    chunk_id: str

    # --- required metadata keys ---
    # framework : str   e.g. "progpy"          — swap when you change frameworks
    # domain    : str   e.g. "thermal", "wear" — drives per-component retrieval
    # type      : str   e.g. "class", "method", "example", "guide", "api"
    # pattern   : str   e.g. "component", "composite", "degradation", "general"
    # source    : str   e.g. "github:progpy/src/progpy/models/battery.py"

    def validate(self) -> None:
        required = {"framework", "domain", "type", "pattern", "source"}
        missing = required - set(self.metadata.keys())
        if missing:
            raise ValueError(f"Chunk {self.chunk_id!r} missing metadata keys: {missing}")


class VectorStore(ABC):
    """
    Abstract interface for all vector store backends.

    To add a new backend (Pinecone, Weaviate, pgvector, etc.):
      1. Create a new file  vector_store/<backend>.py
      2. Subclass VectorStore and implement upsert + query
      3. Change one line in your pipeline: store = PineconeStore(...)

    Nothing else in the pipeline needs to change.
    """

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Insert or update chunks. Must be idempotent (safe to re-run)."""
        ...

    @abstractmethod
    def query(
        self,
        text: str,
        filter: Optional[dict] = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Semantic search over stored chunks.

        filter  : metadata filter applied BEFORE semantic search
                  e.g. {"framework": "progpy", "pattern": "component"}
        top_k   : number of chunks to return
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks currently stored."""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Wipe the collection. Useful when re-indexing from scratch."""
        ...