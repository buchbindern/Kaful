"""The ingest phase: machine.toml documents -> DocumentIndex (+ chunk embeddings).

1. validate the document list against disk (unlisted PDFs: warning, error with --strict);
2. read every page: raw text, label, signals, class, ruled-table markdown (offline);
3. add the LLM layer to pages that need it (`ingest.extract_page`, which must be locked);
4. chunk, then check coverage: every page with text in exactly one chunk (gate);
5. embed the chunks; save the vectors and the DocumentIndex; set the run's index_hash.

Pages whose extraction failed and coverage gaps fail the phase; with --allow-failures
the run continues and the failures stay listed in the manifest.
"""

import logging
from datetime import datetime

from kaful.artifacts.document_index import DocumentIndex, IndexFingerprint, VectorsRef, index_hash_for
from kaful.artifacts.manifest import PhaseRecord
from kaful.phases import PhaseName
from kaful.phases.ingest.chunk import CHUNKER_VERSION, chunk_document, coverage_gaps
from kaful.phases.ingest.classify import CLASSIFIER_VERSION
from kaful.phases.ingest.documents import DocumentError, check_documents
from kaful.phases.ingest.extract import EXTRACT_PROMPT_ID, add_llm_layers, routing
from kaful.phases.ingest.index import save_vectors
from kaful.phases.ingest.read import ReadDocument, read_document
from kaful.paths import to_rel
from kaful.prompts.registry import PromptNotLocked, PromptRegistry, read_lock, require_locked
from kaful.runs import RunContext, now_utc, sha256_bytes

log = logging.getLogger("kaful.ingest")

VECTORS_FILE = "embeddings.npy"


class IngestPhase:
    name: PhaseName = "ingest"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            return self._build(ctx, started)
        except (DocumentError, PromptNotLocked) as exc:
            log.error("ingest cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))

    def _build(self, ctx: RunContext, started: datetime) -> PhaseRecord:
        settings = ctx.settings.ingest
        check = check_documents(ctx.machine_dir, ctx.machine, strict=ctx.manifest.strict)
        documents = [read_document(ctx.machine_dir, record, settings) for record in check.records]
        documents, extraction = self._llm_layers(ctx, documents)
        chunks = sorted((c for d in documents for c in chunk_document(d, settings)), key=lambda c: c.chunk_id)
        pages = tuple(p for d in documents for p in d.pages)
        failures = [f"{p.doc_id} p{p.page}: extraction failed: {p.extraction_error}" for p in pages if p.extraction_error]
        failures += [f"coverage: {gap}" for gap in coverage_gaps(pages, chunks)]
        if ctx.embedder is None:
            raise RuntimeError("ingest needs an embedder")
        vectors = ctx.embedder.embed([c.text for c in chunks], phase=self.name, step="embed_chunks")
        path = ctx.run_dir / self.name / VECTORS_FILE
        data = save_vectors(path, vectors)
        fingerprint = IndexFingerprint(classifier_version=CLASSIFIER_VERSION, chunker_version=CHUNKER_VERSION,
                                       extraction=extraction, embedding_model=ctx.embedder.model,
                                       embedding_dimensions=ctx.embedder.dimensions)
        index_hash = index_hash_for(tuple(chunks), fingerprint)
        index = DocumentIndex(
            run_id=ctx.manifest.run_id, machine_id=ctx.machine.machine_id, index_hash=index_hash,
            documents=check.records, pages=pages, chunks=tuple(chunks), fingerprint=fingerprint,
            vectors=VectorsRef(path=to_rel(ctx.machine_dir, path), sha256=sha256_bytes(data),
                               model=ctx.embedder.model, dimensions=ctx.embedder.dimensions, n_rows=len(chunks)),
        )
        ref = ctx.save_artifact(self.name, index)
        ctx.manifest = ctx.manifest.model_copy(update={"index_hash": index_hash})
        log.info("index built", extra={"fields": {"pages": len(pages), "chunks": len(chunks),
                                                  "llm_pages": sum(p.needs_llm for p in pages),
                                                  "failures": len(failures), "warnings": list(check.warnings)}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=tuple(failures))

    @staticmethod
    def _llm_layers(ctx: RunContext, documents: list[ReadDocument]) -> tuple[list[ReadDocument], str | None]:
        if not any(p.needs_llm for d in documents for p in d.pages):
            return documents, None
        prompt = PromptRegistry().load(EXTRACT_PROMPT_ID)
        require_locked(prompt, read_lock())
        if ctx.gateway is None:
            raise RuntimeError("ingest needs the LLM gateway for pages that need extraction")
        choice = ctx.settings.llm.choice(prompt.id)
        layered = add_llm_layers(ctx.machine_dir, documents, prompt, choice, ctx.gateway, ctx.settings.ingest)
        return layered, routing(prompt, choice)
