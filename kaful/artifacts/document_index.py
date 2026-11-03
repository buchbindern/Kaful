"""Phase 1 (ingest) artifact: raw page text, page labels and chunks per document."""

import hashlib
from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    ArtifactBase,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
    sorted_unique,
)
from kaful.paths import RelPath


class DocumentRecord(StrictModel):
    """One source document listed in machine.toml."""

    doc_id: Identifier
    path: RelPath
    sha256: Sha256
    n_pages: int = Field(ge=1)
    title: str | None = None


class PageText(StrictModel):
    """Raw text of one page, as extracted. Evidence quotes are checked against this."""

    doc_id: Identifier
    page: int = Field(ge=1)
    label: str | None = Field(default=None, description="Printed page label.")
    text: str
    extraction: Literal["pymupdf", "llm"]
    flags: Annotated[tuple[Literal["table", "figure", "garbled"], ...], sorted_unique(str)] = ()


class Chunk(StrictModel):
    """A retrievable span of pages. `chunk_id` is deterministic (see `chunk_id_for`)."""

    chunk_id: Sha256
    doc_id: Identifier
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: str
    chunker_version: NonEmptyStr

    @model_validator(mode="after")
    def _check_pages(self) -> "Chunk":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


def chunk_id_for(file_sha256: str, page_start: int, page_end: int, chunker_version: str) -> str:
    """Deterministic chunk id: hash of file sha + page range + chunker version."""
    key = f"{file_sha256}:{page_start}-{page_end}:{chunker_version}"
    return hashlib.sha256(key.encode()).hexdigest()


def index_hash_for(chunk_ids: tuple[str, ...]) -> str:
    """Index manifest hash: hash of the sorted chunk ids."""
    return hashlib.sha256("\n".join(sorted(chunk_ids)).encode()).hexdigest()


class DocumentIndex(ArtifactBase):
    """Every page's raw text plus the chunk index built from it.

    Here `index_hash` is this index's own hash, `index_hash_for(chunk ids)`; downstream
    artifacts record the same value.
    """

    kind: Literal["document_index"] = "document_index"
    documents: Annotated[tuple[DocumentRecord, ...], by_attr("doc_id")]
    pages: Annotated[
        tuple[PageText, ...], sorted_unique(lambda p: (p.doc_id, p.page), "pages")
    ]
    chunks: Annotated[tuple[Chunk, ...], by_attr("chunk_id")]

    @model_validator(mode="after")
    def _check_index(self) -> "DocumentIndex":
        doc_ids = {d.doc_id for d in self.documents}
        for doc_id in sorted({p.doc_id for p in self.pages} | {c.doc_id for c in self.chunks}):
            if doc_id not in doc_ids:
                raise ValueError(f"unknown doc_id {doc_id!r}")
        if self.index_hash != index_hash_for(tuple(c.chunk_id for c in self.chunks)):
            raise ValueError("index_hash does not match the chunk ids")
        return self
