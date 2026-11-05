"""Phase 1 (ingest) artifact: raw page text, page labels, extra layers and chunks per document.

`PageText.text` is always the raw PyMuPDF text layer: evidence quotes are checked
against it (hard rule 4). Content the text layer cannot carry (ruled tables as
markdown, LLM transcriptions of borderless tables, figures and scans) is stored
in separate, labelled `layers`, never mixed into `text`.
"""

import hashlib
import json
from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Fraction,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
    sorted_unique,
)
from kaful.paths import RelPath

# literature: secondary sources (theses, papers); its values are cited as `literature`, never `standard`.
DocumentKind = Literal["manual", "standard", "datasheet", "catalogue", "literature"]
PageClass = Literal["blank", "scanned", "garbled", "figure", "table_ruled", "table_unruled", "text"]
PAGE_CLASSES: tuple[PageClass, ...] = get_args(PageClass)
# Page classes whose content needs the LLM layer; everything else is raw text (+ table markdown).
NEEDS_LLM: frozenset[PageClass] = frozenset({"scanned", "garbled", "figure", "table_unruled"})
# Where a document's ruled tables come from: PyMuPDF markdown (free), or the LLM layer as well
# (machine.toml `ruled_tables = "llm"`, for documents whose tables have merged cells).
RuledTablesRoute = Literal["pymupdf", "llm"]
LabelSource = Literal["pdf", "printed", "none"]
LayerKind = Literal["table_md", "llm"]


class DocumentRecord(StrictModel):
    """One source document listed in machine.toml."""

    doc_id: Identifier
    path: RelPath
    kind: DocumentKind
    sha256: Sha256
    n_pages: int = Field(ge=1)
    title: str | None = None
    ruled_tables: RuledTablesRoute = "pymupdf"


class PageSignals(StrictModel):
    """Measurable page properties the classifier decides from (all from PyMuPDF)."""

    chars: int = Field(ge=0, description="Non-whitespace-stripped characters of the raw text.")
    lines: int = Field(ge=0)
    image_frac: Fraction = Field(description="Share of the page area covered by raster images.")
    drawings: int = Field(ge=0, description="Vector path objects.")
    curves: int = Field(ge=0, description="Bezier or diagonal path items (diagrams and plots, not table rules).")
    short_line_frac: Fraction = Field(description="Share of lines with <= 2 words or only numbers/dashes.")
    numeric_line_frac: Fraction = Field(description="Share of lines that are only numbers/dashes.")
    garble_ratio: Fraction
    ruled_tables: int = Field(ge=0, description="Tables found by find_tables that pass the quality check.")


class ReusedResponse(StrictModel):
    """An older prompt version's response, reused because the request differed only in a
    lower max_tokens and the response ended normally (LLMSettings.reuse)."""

    prompt_version: int = Field(ge=1)
    prompt_sha256: Sha256 = Field(description="The older request's hash: where the response is cached.")


class PageLayer(StrictModel):
    """Extra content for a page, with what produced it."""

    kind: LayerKind
    text: str
    producer: NonEmptyStr = Field(description="e.g. 'pymupdf 1.27.2.3 find_tables' or 'ingest.extract_page v1'.")
    model: str | None = Field(default=None, description="LLM layers: the model that produced it.")
    prompt_sha256: Sha256 | None = Field(default=None, description="LLM layers: the request hash (cache key).")
    reused_from: ReusedResponse | None = Field(default=None, description="LLM layers: set when the response was "
                                               "an older version's (the layer's own request was not sent).")


class PageText(StrictModel):
    """One page: raw text, printed label, class and extra layers."""

    doc_id: Identifier
    page: int = Field(ge=1, description="Physical 1-based page index.")
    label: str | None = Field(default=None, description="Printed page label.")
    label_source: LabelSource = "none"
    text: str = Field(description="Raw PyMuPDF text layer, unmodified.")
    page_class: PageClass
    signals: PageSignals
    layers: Annotated[tuple[PageLayer, ...], by_attr("kind")] = ()
    extraction_error: str | None = Field(default=None, description="Set when a needed LLM layer failed.")
    llm_by_document: bool = Field(default=False, description="A table_ruled page sent to the LLM because its "
                                  "document sets ruled_tables = 'llm'.")

    @property
    def needs_llm(self) -> bool:
        return self.page_class in NEEDS_LLM or self.llm_by_document


class Chunk(StrictModel):
    """A retrievable span of whole pages. `chunk_id` is deterministic (see `chunk_id_for`)."""

    chunk_id: Sha256
    doc_id: Identifier
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    label_start: str | None = None
    label_end: str | None = None
    section: str | None = Field(default=None, description="Title of the TOC section the chunk starts in.")
    text: str
    text_sha256: Sha256
    chunker_version: NonEmptyStr

    @model_validator(mode="after")
    def _check(self) -> "Chunk":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.text_sha256:
            raise ValueError("text_sha256 does not match text")
        return self


def chunk_id_for(file_sha256: str, page_start: int, page_end: int, chunker_version: str) -> str:
    """Deterministic chunk id: hash of file sha + page range + chunker version."""
    key = f"{file_sha256}:{page_start}-{page_end}:{chunker_version}"
    return hashlib.sha256(key.encode()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IndexFingerprint(StrictModel):
    """Everything besides chunk content that changes what the index returns."""

    classifier_version: NonEmptyStr
    chunker_version: NonEmptyStr
    extraction: str | None = Field(description="'<prompt id> v<N> <model>' (+ ' -> <model> when <rule>' with escalation) if any page has an LLM layer.")
    embedding_model: NonEmptyStr
    embedding_dimensions: int = Field(ge=1)
    llm_ruled_tables: Annotated[tuple[str, ...], SORTED_STRS] = Field(
        default=(), description="Documents whose table_ruled pages also get the LLM layer (ruled_tables = 'llm').")
    embedding_windows: str | None = Field(
        default=None, description="'<encoding> <window>/<overlap>' when some chunk is embedded in several windows.")


def index_hash_for(chunks: tuple[Chunk, ...], fingerprint: IndexFingerprint) -> str:
    """Index manifest hash: sorted (chunk_id, text_sha256) pairs plus the fingerprint.

    A chunk id alone does not change when an LLM layer's text changes (new model or
    prompt version), so the content hash of every chunk is part of the index hash. An empty
    `llm_ruled_tables` and an unset `embedding_windows` are left out, so indexes built
    before those fields (and indexes that need neither) keep their hash.
    """
    dumped = fingerprint.model_dump(mode="json")
    for name in ("llm_ruled_tables", "embedding_windows"):
        if not dumped[name]:
            del dumped[name]
    payload = {"chunks": sorted([c.chunk_id, c.text_sha256] for c in chunks), "fingerprint": dumped}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class VectorsRef(StrictModel):
    """The chunk embeddings file: float32 rows in `DocumentIndex.chunks` order, one per chunk,
    or (when a chunk is too long for one embedding) one per window, with `row_chunk`."""

    path: RelPath
    sha256: Sha256
    model: NonEmptyStr
    dimensions: int = Field(ge=1)
    n_rows: int = Field(ge=0)
    row_chunk: tuple[int, ...] = Field(
        default=(), description="Chunk index (into DocumentIndex.chunks) of each row; empty: row i is chunk i.")

    @model_validator(mode="after")
    def _check_rows(self) -> "VectorsRef":
        rows = self.row_chunk
        if rows and (len(rows) != self.n_rows or rows[0] != 0
                     or any(b - a not in (0, 1) for a, b in zip(rows, rows[1:], strict=False))):
            raise ValueError("row_chunk must give every row, in chunk order, starting at chunk 0 with no gaps")
        return self

    def chunk_of_rows(self) -> tuple[int, ...]:
        return self.row_chunk or tuple(range(self.n_rows))


class DocumentIndex(ArtifactBase):
    """Every page's raw text and layers, plus the chunk index built from it.

    Here `index_hash` is this index's own hash, `index_hash_for(chunks, fingerprint)`;
    downstream artifacts record the same value.
    """

    kind: Literal["document_index"] = "document_index"
    documents: Annotated[tuple[DocumentRecord, ...], by_attr("doc_id")]
    pages: Annotated[
        tuple[PageText, ...], sorted_unique(lambda p: (p.doc_id, p.page), "pages")
    ]
    chunks: Annotated[tuple[Chunk, ...], by_attr("chunk_id")]
    fingerprint: IndexFingerprint
    vectors: VectorsRef | None = None

    @model_validator(mode="after")
    def _check_index(self) -> "DocumentIndex":
        doc_ids = {d.doc_id for d in self.documents}
        for doc_id in sorted({p.doc_id for p in self.pages} | {c.doc_id for c in self.chunks}):
            if doc_id not in doc_ids:
                raise ValueError(f"unknown doc_id {doc_id!r}")
        if self.index_hash != index_hash_for(self.chunks, self.fingerprint):
            raise ValueError("index_hash does not match the chunks and fingerprint")
        if self.vectors is not None and len(set(self.vectors.chunk_of_rows())) != len(self.chunks):
            raise ValueError("vectors must have at least one row for every chunk")
        return self
