"""
tests/harness/fake_embedder.py
------------------------------
Offline stand-ins for the two embedding paths in the legacy pipeline.

    HashingEmbeddingFunction  replaces Chroma's OpenAIEmbeddingFunction in
                              utils.rag: hashed bag of words, so retrieval is
                              deterministic and still roughly lexical.
    frozen_progpy_store       replaces the ProgPy ChromaDBStore used by
                              utils.progpy_rag: serves the hits frozen in
                              tests/fixtures/progpy/framework_hits.json.

The recording run uses the same stand-ins, so recorded prompts match replay.
"""

import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from chromadb import Documents, EmbeddingFunction, Embeddings

from config import PROGPY_RAG_DIR

DIM = 256
FAKE_EMBEDDER_NAME = "kaful_test_hashing"
PROGPY_HITS_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "progpy" / "framework_hits.json"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def embed_text(text: str) -> list[float]:
    """Unit vector of signed token-hash counts (stable across processes)."""
    vec = [0.0] * DIM
    for tok in _tokens(text):
        h = hashlib.sha256(tok.encode()).digest()
        vec[int.from_bytes(h[:4], "big") % DIM] += 1.0 if h[4] & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """Accepts and ignores OpenAIEmbeddingFunction's constructor arguments."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        return [embed_text(t) for t in input]  # type: ignore[misc]

    @staticmethod
    def name() -> str:
        return FAKE_EMBEDDER_NAME

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()


def _hit(d: dict[str, Any]) -> Any:
    meta = {"type": d.get("type", ""), "name": d.get("name", "")}
    return type("Hit", (), {"chunk_id": d["chunk_id"], "text": d["text"], "metadata": meta})


@contextmanager
def frozen_progpy_store(path: Path = PROGPY_HITS_PATH) -> Iterator[None]:
    """Make ChromaDBStore.query answer from the frozen hits; unknown queries are errors."""
    hits: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))["hits"]
    rag_path = str(PROGPY_RAG_DIR)
    if rag_path not in sys.path:
        sys.path.insert(0, rag_path)  # progpy_rag's own imports need it (legacy layout)
    from vector_store import ChromaDBStore  # type: ignore[import-not-found]

    def init(self: Any, *_a: Any, **_k: Any) -> None:
        """No Chroma client: the real store is gitignored and bound to OpenAI."""

    def query(self: Any, text: str, filter: Any = None, top_k: int = 5) -> list[Any]:
        if text not in hits:
            raise LookupError(f"no frozen ProgPy hits for query {text!r} in {path.name}")
        return [_hit(d) for d in hits[text][:top_k]]

    saved = ChromaDBStore.__init__, ChromaDBStore.query
    ChromaDBStore.__init__, ChromaDBStore.query = init, query
    try:
        yield
    finally:
        ChromaDBStore.__init__, ChromaDBStore.query = saved
