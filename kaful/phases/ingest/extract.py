"""The LLM layer: `ingest.extract_page` on every page whose class needs it.

Each such page is sent as its PNG plus its raw text layer, through the gateway (one
Message Batch by default; cached responses cost nothing). The output is rendered to
markdown and stored as the page's `llm` layer; the raw text is never changed. A page
whose reply stays invalid after the gateway's one retry keeps raw text only and gets
`extraction_error`, which fails the phase unless failures are allowed.

Routing (the prompt's `escalate` entry in config): every page goes to the configured
model first; a page whose first pass meets an escalation rule (`output_has_tables`: its
output has any table; `first_pass_failed`: its reply stayed invalid, e.g. it ran out of
max_tokens) is re-run on the escalation model, and that output is used. Both passes are in
the ledger with their `route`, and the layer records the model that produced it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import PageLayer, PageText
from kaful.config import EscalationRule, IngestSettings, ModelChoice
from kaful.llm.gateway import Call, CallScope, Gateway, LLMOutputError, LLMResult
from kaful.paths import resolve_in
from kaful.phases.ingest.read import ReadDocument
from kaful.phases.ingest.render import page_png
from kaful.prompts.ingest.extract_page.models import Figure, PageExtraction, Table
from kaful.prompts.registry import Prompt

log = logging.getLogger("kaful.ingest.extract")

EXTRACT_PROMPT_ID = "ingest.extract_page"
PHASE = "ingest"
STEP = "extract_page"
ARTIFACT = "document_index"


@dataclass(frozen=True)
class PageCall:
    doc_index: int
    page_index: int
    call: Call


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _table_md(table: Table) -> str:
    width = max([len(table.header), *(len(r.cells) for r in table.rows)] or [1])
    header = table.header + [""] * (width - len(table.header))
    lines = [f"Table ({table.source}): {table.title}" if table.title else f"Table ({table.source})",
             "| " + " | ".join(_cell(h) for h in header) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(_cell(c) for c in r.cells + [""] * (width - len(r.cells))) + " |" for r in table.rows]
    return "\n".join(lines)


def _figure_md(figure: Figure) -> str:
    lines = [f"Figure ({figure.kind}): {figure.caption}" if figure.caption else f"Figure ({figure.kind})",
             figure.description]
    if figure.text_in_image:
        lines.append("Text in figure: " + "; ".join(figure.text_in_image))
    lines += [f"- {v.label}: {v.value}{' ' + v.unit if v.unit else ''}"
              + (" (read from curve)" if v.how == "read_from_curve" else "") for v in figure.values]
    return "\n".join(lines)


def render_extraction(output: PageExtraction) -> str:
    """The extraction as markdown, in output order: tables, figures, then OCR text."""
    parts = [_table_md(t) for t in output.tables] + [_figure_md(f) for f in output.figures]
    if output.ocr_text:
        parts.append("OCR text:\n" + output.ocr_text.strip())
    return "\n\n".join(parts)


def escalation(outcome: LLMResult | str, rules: Sequence[EscalationRule]) -> EscalationRule | None:
    """The first rule (in config order) a first-pass outcome meets, or None."""
    for rule in rules:
        if rule == "first_pass_failed" and isinstance(outcome, str):
            return rule
        if rule == "output_has_tables" and isinstance(outcome, LLMResult):
            tables = outcome.output.model_dump(mode="json").get("tables") or []  # by field, not class: eval copies
            if any(t.get("header") or t.get("rows") for t in tables):
                return rule
    return None


def routing(prompt: Prompt, choice: ModelChoice) -> str:
    """The routing rule as recorded in the index fingerprint."""
    rule = f"{prompt.id} v{prompt.version} {choice.model}"
    return rule + (f" -> {choice.escalate.model} when {choice.escalate.rule()}" if choice.escalate else "")


def extraction_calls(machine_dir: Path, documents: Sequence[ReadDocument], prompt: Prompt,
                     settings: IngestSettings, route: str | None = None) -> list[PageCall]:
    """One call per page that needs the LLM layer, in (document, page) order."""
    calls = []
    for d, document in enumerate(documents):
        wanted = [i for i, page in enumerate(document.pages) if page.needs_llm]
        if not wanted:
            continue
        with fitz.open(resolve_in(machine_dir, document.record.path)) as doc:
            for i in wanted:
                page = document.pages[i]
                scope = CallScope(PHASE, STEP, component=f"{page.doc_id}:p{page.page}", artifact=ARTIFACT)
                images = {"page_image": page_png(doc[page.page - 1], settings)}
                calls.append(PageCall(d, i, Call(prompt, {"raw_text": page.text}, scope, images=images, route=route)))
    return calls


def _results(gateway: Gateway, calls: Sequence[PageCall]) -> list[LLMResult | str]:
    """A result or an error message per call. Run as one batch; if any reply stays invalid,
    re-run each call alone (the valid ones are cache hits, so nothing is paid twice)."""
    try:
        return list(gateway.run_many([c.call for c in calls]))
    except LLMOutputError:
        log.warning("some page extractions are invalid; collecting per page")
    outcomes: list[LLMResult | str] = []
    for c in calls:
        try:
            outcomes.append(gateway.call(c.call))
        except LLMOutputError as exc:
            outcomes.append(str(exc))
    return outcomes


def _routed(gateway: Gateway, calls: Sequence[PageCall], choice: ModelChoice) -> list[tuple[LLMResult | str, str]]:
    """(outcome, model) per call: the first pass, with escalated pages replaced by the escalation's outcome."""
    outcomes = [(o, choice.model) for o in _results(gateway, calls)]
    if choice.escalate is None:
        return outcomes
    strong = choice.escalated()
    why = {i: r for i, (o, _) in enumerate(outcomes) if (r := escalation(o, choice.escalate.when)) is not None}
    up = sorted(why)
    second = [replace(calls[i], call=replace(calls[i].call, choice=strong,
                                             route=f"escalated from {choice.model}: {why[i]}")) for i in up]
    for i, outcome in zip(up, _results(gateway, second), strict=True):
        outcomes[i] = (outcome, strong.model)
    log.info("pages escalated", extra={"fields": {"model": strong.model, "pages": len(up), "of": len(calls),
                                                  "by_rule": {r: list(why.values()).count(r)
                                                              for r in choice.escalate.when}}})
    return outcomes


def add_llm_layers(machine_dir: Path, documents: Sequence[ReadDocument], prompt: Prompt, choice: ModelChoice,
                   gateway: Gateway, settings: IngestSettings) -> list[ReadDocument]:
    """Documents with an `llm` layer (or `extraction_error`) on every page that needs one."""
    first = f"first pass; escalate to {choice.escalate.model} when {choice.escalate.rule()}" if choice.escalate else None
    calls = extraction_calls(machine_dir, documents, prompt, settings, route=first)
    pages = [list(d.pages) for d in documents]
    for page_call, (outcome, model) in zip(calls, _routed(gateway, calls, choice), strict=True):
        page: PageText = pages[page_call.doc_index][page_call.page_index]
        if isinstance(outcome, str):
            pages[page_call.doc_index][page_call.page_index] = page.model_copy(update={"extraction_error": outcome})
            continue
        assert isinstance(outcome.output, PageExtraction)
        layer = PageLayer(kind="llm", text=render_extraction(outcome.output), producer=f"{prompt.id} v{prompt.version}",
                          model=model, prompt_sha256=outcome.prompt_sha256)
        layers = tuple(sorted((*page.layers, layer), key=lambda item: item.kind))
        pages[page_call.doc_index][page_call.page_index] = page.model_copy(update={"layers": layers})
    return [ReadDocument(d.record, tuple(p), d.toc) for d, p in zip(documents, pages, strict=True)]
