"""The ingest phase: machine.toml documents -> DocumentIndex (+ chunk embeddings).

Steps, in the order `_build` runs them:
  a_check_documents     the machine.toml document list against the files on disk
                        (unlisted PDFs: warning, error with --strict)
  b_read_pages          every page's raw text, printed label and classifier signals, and
                        each document's table of contents        helpers: page_signals
  c_classify_pages      a class per page, which decides whether it needs the LLM layer
  d_add_ruled_tables    free PyMuPDF markdown for ruled tables    helpers: page_signals
  e_extract_hard_pages  the LLM layer (`ingest.extract_page`, which must be locked):
                        first pass, escalation where config's rules say
                                                                  helpers: render_pages
  f_chunk_pages         chunks, then the coverage check: every page with text in
                        exactly one chunk (gate)
  g_embed_and_index     embeddings (windows for chunks over the model's limit), the
                        fingerprint, the saved vectors and DocumentIndex
Shared types between steps: helpers/documents. Tooling not used during a run: tools/.
Loading and searching the saved index: kaful.retrieval.

Pages whose extraction failed and coverage gaps fail the phase before anything is
embedded (no index is saved); with --allow-failures the phase embeds and saves anyway and
the failures stay listed in the manifest. A missing gateway or embedder fails the phase.
"""

import logging
from datetime import datetime

from kaful.artifacts.document_index import PageText
from kaful.artifacts.manifest import PhaseRecord
from kaful.phases import PhaseName
from kaful.phases.p1_ingest.a_check_documents import DocumentError, check_documents
from kaful.phases.p1_ingest.b_read_pages import read_pages
from kaful.phases.p1_ingest.c_classify_pages import classify_pages
from kaful.phases.p1_ingest.d_add_ruled_tables import add_ruled_tables
from kaful.phases.p1_ingest.e_extract_hard_pages import (
    EXTRACT_PROMPT_ID,
    add_llm_layers,
    routing,
)
from kaful.phases.p1_ingest.f_chunk_pages import chunk_document, coverage_gaps
from kaful.phases.p1_ingest.g_embed_and_index import embed_and_index
from kaful.prompts.registry import (
    Prompt,
    PromptNotLocked,
    PromptRegistry,
    read_lock,
    require_locked,
)
from kaful.runs import RunContext, now_utc

log = logging.getLogger("kaful.ingest")


class MissingService(RuntimeError):
    """The run has no LLM gateway or embedder, and ingest needs it."""


def _reuse_notes(pages: tuple[PageText, ...], prompt_id: str, version: int) -> tuple[str, ...]:
    """One note per older version whose responses were reused, with the page count."""
    older = sorted(layer.reused_from.prompt_version for p in pages for layer in p.layers if layer.reused_from)
    return tuple(f"{older.count(v)} page(s): {prompt_id} v{v} responses reused under v{version} "
                 f"(the request differed only in a lower max_tokens; the response ended normally)"
                 for v in sorted(set(older)))


def _extraction_prompt() -> Prompt:
    prompt = PromptRegistry().load(EXTRACT_PROMPT_ID)
    require_locked(prompt, read_lock())
    return prompt


class IngestPhase:
    name: PhaseName = "ingest"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            return self._build(ctx, started)
        except (DocumentError, PromptNotLocked, MissingService) as exc:
            log.error("ingest cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))

    def _build(self, ctx: RunContext, started: datetime) -> PhaseRecord:
        settings, machine_dir = ctx.settings.ingest, ctx.machine_dir
        check = check_documents(machine_dir, ctx.machine, strict=ctx.manifest.strict)
        raw = [read_pages(machine_dir, record, settings) for record in check.records]
        documents = [classify_pages(document, settings.classifier) for document in raw]
        documents = [add_ruled_tables(machine_dir, document, settings.classifier) for document in documents]
        extraction, prompt = None, None
        if any(p.needs_llm for d in documents for p in d.pages):
            prompt = _extraction_prompt()
            if ctx.gateway is None:
                raise MissingService("ingest needs the LLM gateway for pages that need extraction")
            choice = ctx.settings.llm.choice(prompt.id)
            documents = add_llm_layers(machine_dir, documents, prompt, choice, ctx.gateway, settings)
            extraction = routing(prompt, choice)
        chunks = sorted((c for d in documents for c in chunk_document(d, settings)), key=lambda c: c.chunk_id)
        pages = tuple(p for d in documents for p in d.pages)
        failures = [f"{p.doc_id} p{p.page}: extraction failed: {p.extraction_error}" for p in pages if p.extraction_error]
        failures += [f"coverage: {gap}" for gap in coverage_gaps(pages, chunks)]
        if failures and not ctx.manifest.allow_failures:  # stop before paying for embeddings
            log.error("ingest failed before embedding", extra={"fields": {"failures": len(failures)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=tuple(failures))
        if ctx.embedder is None:
            raise MissingService("ingest needs an embedder")
        index, ref = embed_and_index(ctx, ctx.embedder, check.records, pages, chunks, extraction)
        ctx.manifest = ctx.manifest.model_copy(update={"index_hash": index.index_hash})
        log.info("index built", extra={"fields": {"pages": len(pages), "chunks": len(chunks),
                                                  "llm_pages": sum(p.needs_llm for p in pages),
                                                  "failures": len(failures), "warnings": list(check.warnings)}})
        notes = _reuse_notes(pages, prompt.id, prompt.version) if prompt is not None else ()
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=tuple(failures), notes=notes)
