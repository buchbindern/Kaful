"""Embeddings: the only way kaful embeds text. Same ledger and budget as the LLM gateway.

Vectors are cached by content (model, dimensions, sha256 of the text), one float32 .npy
file each, so rebuilding an index only embeds chunks whose text changed. Each batch of
uncached texts is checked against the run budget before it is sent (worst case:
characters / chars_per_token_worst tokens) and written to the ledger as one
`embedding` row; cache hits are one row with cost 0. Texts longer than
`embedding_max_chars` are cut to that length before embedding (and before hashing).
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from kaful.config import IngestSettings, LLMSettings
from kaful.llm.budget import RunBudget
from kaful.llm.gateway import OfflineMissError
from kaful.llm.ledger import LEDGER_SCHEMA_VERSION, Ledger, LedgerRow
from kaful.llm.pricing import PricingTable, TokenUsage, load_pricing

Vectors = npt.NDArray[np.float32]


class EmbeddingTransport(Protocol):
    def embed(self, texts: Sequence[str], model: str, dimensions: int) -> tuple[list[list[float]], int]:
        """Vectors in input order, and the billed input tokens."""
        ...


class VectorStore:
    """Content-addressed vector cache: <root>/<model>/<dims>/<sha[:2]>/<sha>.npy."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, model: str, dimensions: int, sha: str) -> Path:
        return self.root / model / str(dimensions) / sha[:2] / f"{sha}.npy"

    def get(self, model: str, dimensions: int, sha: str) -> Vectors | None:
        path = self.path(model, dimensions, sha)
        return np.load(path).astype(np.float32) if path.is_file() else None

    def put(self, model: str, dimensions: int, sha: str, vector: Vectors) -> None:
        path = self.path(model, dimensions, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, vector.astype(np.float32))
        tmp.replace(path)


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Embedder:
    settings: IngestSettings
    llm: LLMSettings
    ledger: Ledger
    budget: RunBudget
    transport: EmbeddingTransport | None
    store: VectorStore
    run_id: str | None = None
    machine_id: str | None = None
    pricing: PricingTable = field(default_factory=load_pricing)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    @property
    def model(self) -> str:
        return self.settings.embedding_model

    @property
    def dimensions(self) -> int:
        return self.settings.embedding_dimensions

    def embed(self, texts: Sequence[str], phase: str, step: str) -> Vectors:
        """One row per text, in input order (float32, shape (n, dimensions))."""
        cut = [t[: self.settings.embedding_max_chars] for t in texts]
        keys = [text_key(t) for t in cut]
        found = {k: v for k in sorted(set(keys)) if (v := self.store.get(self.model, self.dimensions, k)) is not None}
        misses = sorted({k: t for k, t in zip(keys, cut, strict=True) if k not in found}.items())
        if found:
            self._log(phase, step, n_texts=len(found), tokens=0, cost=0.0, cache="hit")
        if misses and self.transport is None:
            raise OfflineMissError(f"{len(misses)} text(s) have no cached {self.model} embedding and there is no transport")
        for start in range(0, len(misses), self.settings.embedding_batch_size):
            found.update(self._send(misses[start: start + self.settings.embedding_batch_size], phase, step))
        if not keys:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def _send(self, batch: Sequence[tuple[str, str]], phase: str, step: str) -> dict[str, Vectors]:
        assert self.transport is not None
        worst_tokens = sum(int(len(t) / self.llm.chars_per_token_worst) + 1 for _, t in batch)
        self.budget.check(self._cost(worst_tokens), f"{len(batch)} embedding(s) ({phase}/{step})")
        vectors, tokens = self.transport.embed([t for _, t in batch], self.model, self.dimensions)
        if len(vectors) != len(batch) or any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"{self.model}: expected {len(batch)} vectors of {self.dimensions} dimensions")
        cost = self._cost(tokens)
        self.budget.charge(cost)
        self._log(phase, step, n_texts=len(batch), tokens=tokens, cost=cost, cache="miss")
        out = {}
        for (key, _), vector in zip(batch, vectors, strict=True):
            array = np.asarray(vector, dtype=np.float32)
            self.store.put(self.model, self.dimensions, key, array)
            out[key] = array
        return out

    def _cost(self, tokens: int) -> float:
        return self.pricing.cost(self.model, TokenUsage(input_tokens=tokens))

    def _log(self, phase: str, step: str, *, n_texts: int, tokens: int, cost: float,
             cache: str) -> None:
        self.ledger.append(LedgerRow(
            schema_version=LEDGER_SCHEMA_VERSION, ts=self.now().isoformat(timespec="milliseconds"),
            run_id=self.run_id, machine_id=self.machine_id, event="embedding", phase=phase, step=step,
            provider=self.pricing.model(self.model).provider, model=self.model, status="ok",
            input_tokens=tokens, cost_usd=cost, pricing_as_of=self.pricing.as_of, n_texts=n_texts,
            response_cache="hit" if cache == "hit" else "miss",
        ))
