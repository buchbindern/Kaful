"""M6 grounding metrics: a frozen set of legacy evidence quotes, scored against the old
(legacy) index and the new DocumentIndex, plus page coverage.

Per PLAN M6 the old index is reference only, not a target: it is an LLM transcription,
partly hand-tweaked, and the quotes were copied from it. Scores reported:
- old index (legacy Chroma documents, plus the saved phase-2 context where the live
  index is partial, e.g. CNC);
- new index, any layer (raw text + table markdown + LLM layer);
- new index, raw text only (what hard rule 4 accepts as `manual` evidence).
The gate items are elsewhere (classifier recall, extract_page evals, coverage, and the
hand review of old-only quotes, whose template `old_only_review` writes).

Matching uses the shared evidence matcher (`kaful.evidence`, `MATCHER_VERSION`) with
`grounding_near_ratio`. Pages are joined with the next page, so quotes across a page
break match.
"""

import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from kaful.artifacts.base import Fraction, Identifier, NonEmptyStr, Sha256, StrictModel, by_attr
from kaful.artifacts.document_index import PageText
from kaful.config import IngestSettings
from kaful.evidence import Matcher as _Matcher
from kaful.evidence import MATCHER_VERSION, Passage, QuoteMatch, tokens
from kaful.phases.p1_ingest.f_chunk_pages import has_text
from kaful.runs import sha256_bytes

LEGACY_DIRS = ("schema", "twin")
EVIDENCE_KEY = "evidence"
CHROMA_FILE = Path("rag") / "vector_db" / "chroma.sqlite3"
SAVED_CONTEXT = Path("outputs") / "schema" / "step_a_chunks.json"


# ── The frozen quote set ─────────────────────────────────────────────────────

class Quote(StrictModel):
    quote_id: NonEmptyStr = Field(description="First 12 hex chars of sha256 of the normalized quote.")
    text: NonEmptyStr
    sources: tuple[str, ...] = Field(description="'<checkpoint path>#<json path>' where it occurs (sorted).")


class QuoteSet(StrictModel):
    schema_version: Literal[1] = 1
    machine_id: Identifier
    description: NonEmptyStr
    quotes: Annotated[tuple[Quote, ...], by_attr("quote_id")]


def _evidence(node: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for key in sorted(node):
            value = node[key]
            if key == EVIDENCE_KEY and isinstance(value, str) and value.strip():
                yield f"{path}.{key}", value.strip()
            else:
                yield from _evidence(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _evidence(value, f"{path}[{i}]")


def collect_quotes(machine_dir: Path, machine_id: str, min_tokens: int) -> QuoteSet:
    """Distinct legacy evidence quotes (>= min_tokens tokens) from outputs/{schema,twin}, without snapshots."""
    found: dict[str, tuple[str, set[str]]] = {}
    for subdir in LEGACY_DIRS:
        for file in sorted((machine_dir / "outputs" / subdir).rglob("*.json")):
            rel = file.relative_to(machine_dir).as_posix()
            if "SNAPSHOT" in rel:
                continue
            for path, text in _evidence(json.loads(file.read_text(encoding="utf-8")), ""):
                norm = " ".join(tokens(text))
                if len(norm.split()) >= min_tokens:
                    found.setdefault(norm, (text, set()))[1].add(f"{rel}#{path}")
    quotes = [Quote(quote_id=sha256_bytes(norm.encode())[:12], text=text, sources=tuple(sorted(srcs)))
              for norm, (text, srcs) in found.items()]
    return QuoteSet(machine_id=machine_id, quotes=tuple(sorted(quotes, key=lambda q: q.quote_id)),
                    description="Legacy evidence quotes from outputs/schema and outputs/twin checkpoints "
                                "(snapshots excluded), frozen for the M6 grounding metric.")


# ── Matching ─────────────────────────────────────────────────────────────────

class Matcher(_Matcher):
    """The shared evidence matcher, configured from the ingest grounding settings."""

    def __init__(self, passages: Sequence[Passage], settings: IngestSettings) -> None:
        super().__init__(passages, settings.grounding_near_ratio, settings.grounding_candidate_overlap)


def page_passages(pages: Sequence[PageText], layers: bool) -> list[Passage]:
    """Each page joined with the next page of the same document; with or without extra layers."""
    def text(page: PageText) -> str:
        return "\n".join([page.text, *(layer.text for layer in page.layers)]) if layers else page.text

    out = []
    for i, page in enumerate(pages):
        following = pages[i + 1] if i + 1 < len(pages) and pages[i + 1].doc_id == page.doc_id else None
        out.append(Passage(f"{page.doc_id}:p{page.page}", text(page) + ("\n" + text(following) if following else "")))
    return out


# ── The old (legacy) index ───────────────────────────────────────────────────

@dataclass(frozen=True)
class OldIndex:
    passages: tuple[Passage, ...]
    pages: frozenset[tuple[str, int]]
    partial: bool


def read_old_index(machine_dir: Path, doc_files: dict[str, str], total_pages: int) -> OldIndex:
    """Legacy Chroma documents (read-only), plus the saved phase-2 context when the live index
    covers < 95% of the listed pages. `doc_files` maps file name -> doc_id."""
    uri = f"file:{(machine_dir / CHROMA_FILE).resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute(
            "SELECT d.id, d.string_value, f.string_value, m.string_value, s.int_value, e.int_value "
            "FROM embedding_metadata d "
            "LEFT JOIN embedding_metadata f ON f.id = d.id AND f.key = 'pdf_file' "
            "LEFT JOIN embedding_metadata m ON m.id = d.id AND m.key = 'manual_id' "
            "LEFT JOIN embedding_metadata s ON s.id = d.id AND s.key = 'start_page' "
            "LEFT JOIN embedding_metadata e ON e.id = d.id AND e.key = 'end_page' "
            "WHERE d.key = 'chroma:document' ORDER BY d.id").fetchall()
    passages: list[Passage] = []
    pages: set[tuple[str, int]] = set()
    only = next(iter(doc_files.values())) if len(doc_files) == 1 else None
    for row_id, text, pdf_file, _manual, start, end in rows:
        doc_id = doc_files.get(pdf_file or "", only)
        passages.append(Passage(f"chroma:{row_id}:{doc_id}:p{start}-{end}", text or ""))
        if doc_id is not None and start is not None and end is not None:
            pages.update((doc_id, p) for p in range(int(start), int(end) + 1))
    partial = len(pages) < 0.95 * total_pages
    if partial and (machine_dir / SAVED_CONTEXT).is_file():
        saved = json.loads((machine_dir / SAVED_CONTEXT).read_text(encoding="utf-8"))
        passages += [Passage(f"saved_context:{c.get('chunk_id')}", str(c.get("text", ""))) for c in saved]
    return OldIndex(tuple(passages), frozenset(pages), partial)


# ── Report ───────────────────────────────────────────────────────────────────

class Score(StrictModel):
    verbatim: int
    near: int
    unsupported: int

    @property
    def supported(self) -> int:
        return self.verbatim + self.near


class QuoteResult(StrictModel):
    quote_id: NonEmptyStr
    old: QuoteMatch
    old_where: str | None
    new_any: QuoteMatch | None
    raw_only: QuoteMatch


class Coverage(StrictModel):
    pages_total: int
    pages_empty: int = Field(description="Pages with no text in any layer (the only pages not chunked).")
    pages_in_chunks: int
    share: Fraction = Field(description="Pages with text in at least one chunk / pages with text.")


class GroundingReport(StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["grounding_report"] = "grounding_report"
    machine_id: Identifier
    quote_set_sha256: Sha256
    n_quotes: int
    run_id: str | None = Field(description="The ingest run scored; None for a raw-text preview.")
    matcher_version: NonEmptyStr = Field(default="matcher-v1", description="Reports before matcher-v2 omit it.")
    index_hash: str | None
    old: Score
    old_partial: bool
    old_page_coverage: Fraction
    new_any_layer: Score | None
    raw_only: Score
    new_coverage: Coverage | None
    old_only: tuple[str, ...] = Field(description="Quotes the old index supports and the new any-layer index does not.")
    quotes: Annotated[tuple[QuoteResult, ...], by_attr("quote_id")]
    note: NonEmptyStr = ("The old index is reference only: it is an LLM transcription, partly hand-tweaked, and "
                         "these quotes were copied from it. Scores are context, not a gate.")


def score(classes: Sequence[QuoteMatch]) -> Score:
    return Score(verbatim=classes.count("verbatim"), near=classes.count("near"),
                 unsupported=classes.count("unsupported"))


def coverage(pages: Sequence[PageText], chunk_pages: set[tuple[str, int]]) -> Coverage:
    content = [p for p in pages if has_text(p)]
    covered = sum(1 for p in content if (p.doc_id, p.page) in chunk_pages)
    return Coverage(pages_total=len(pages), pages_empty=len(pages) - len(content), pages_in_chunks=covered,
                    share=covered / len(content) if content else 1.0)


def build_report(quote_set: QuoteSet, quote_set_sha256: str, old: OldIndex, pages: Sequence[PageText],
                 settings: IngestSettings, *, run_id: str | None, index_hash: str | None,
                 chunk_pages: set[tuple[str, int]] | None) -> GroundingReport:
    """Score every quote against the old index, the raw text and (with a run) every layer."""
    old_matcher = Matcher(old.passages, settings)
    raw_matcher = Matcher(page_passages(pages, layers=False), settings)
    any_matcher = Matcher(page_passages(pages, layers=True), settings) if run_id else None
    results = []
    for quote in quote_set.quotes:
        old_class, old_where = old_matcher.match(quote.text)
        raw_class, _ = raw_matcher.match(quote.text)
        any_class = any_matcher.match(quote.text)[0] if any_matcher else None
        results.append(QuoteResult(quote_id=quote.quote_id, old=old_class, old_where=old_where, new_any=any_class,
                                   raw_only=raw_class))
    old_only = tuple(r.quote_id for r in results if r.old != "unsupported"
                     and (r.new_any if r.new_any is not None else r.raw_only) == "unsupported")
    total = len(pages)
    return GroundingReport(
        machine_id=quote_set.machine_id, quote_set_sha256=quote_set_sha256, n_quotes=len(results), run_id=run_id,
        matcher_version=MATCHER_VERSION,
        index_hash=index_hash, old=score([r.old for r in results]), old_partial=old.partial,
        old_page_coverage=len(old.pages) / total if total else 0.0,
        new_any_layer=score([r.new_any for r in results if r.new_any is not None]) if run_id else None,
        raw_only=score([r.raw_only for r in results]),
        new_coverage=coverage(pages, chunk_pages) if chunk_pages is not None else None,
        old_only=old_only, quotes=tuple(results),
    )


# ── Hand review of old-only quotes ───────────────────────────────────────────

EXCERPT_CHARS = 900
ALSO_ON_MIN_RECALL = 0.8  # a page holding this share of the quote's words is listed under all_pages
OLD_RANGE_RE = re.compile(r":(?P<doc>[a-z][a-z0-9_]*):p(?P<start>\d+)-(?P<end>\d+)$")


def page_ref(page: PageText) -> str:
    """'doc:p<physical page>' plus the printed label when there is one, e.g. 'fanuc:p590 (label 566)'."""
    return f"{page.doc_id}:p{page.page}" + (f" (label {page.label})" if page.label else "")


def _recall(wanted: set[str], page: PageText) -> float:
    return len(wanted & set(tokens("\n".join([page.text, *(x.text for x in page.layers)])))) / len(wanted)


def _candidate(quote: str, pages: Sequence[PageText], old_where: str | None) -> PageText | None:
    """The page whose text (all layers) holds most of the quote's tokens, within the old chunk's pages if known."""
    match = OLD_RANGE_RE.search(old_where or "")
    pool = [p for p in pages if match is None or (p.doc_id == match["doc"]
                                                   and int(match["start"]) <= p.page <= int(match["end"]))] or list(pages)
    wanted = set(tokens(quote))
    if not wanted or not pool:
        return None
    return max(pool, key=lambda p: (_recall(wanted, p), -p.page))


def all_pages(quote: str, pages: Sequence[PageText]) -> list[PageText]:
    """Every page holding most of the quote's words (manuals repeat passages), in document order."""
    wanted = set(tokens(quote))
    return [p for p in pages if wanted and _recall(wanted, p) >= ALSO_ON_MIN_RECALL]


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def old_only_review(report: GroundingReport, quote_set: QuoteSet, pages: Sequence[PageText]) -> str:
    """TOML table for the M6 hand review: is each old-only quote's information in the new index?"""
    by_id = {q.quote_id: q for q in quote_set.quotes}
    results = {r.quote_id: r for r in report.quotes}
    lines = [f"# M6 hand review of old-only quotes: {report.machine_id}, run {report.run_id}.",
             "# For each quote: is its information present in the new index, regardless of wording?",
             '# present = "yes" | "no" | "partial"; reviewer = who decided; note = why (optional).',
             "# Pages are physical (1-based PDF page) with the printed label in brackets; all_pages lists",
             f"# every page holding >= {ALSO_ON_MIN_RECALL:.0%} of the quote's words (manuals repeat passages).", ""]
    for quote_id in report.old_only:
        quote, result = by_id[quote_id], results[quote_id]
        page = _candidate(quote.text, pages, result.old_where)
        excerpt = "" if page is None else "\n".join([page.text, *(x.text for x in page.layers)])[:EXCERPT_CHARS]
        lines += ["[[quotes]]", f"quote_id = {_toml_str(quote_id)}", f"quote = {_toml_str(quote.text)}",
                  f"old_where = {_toml_str(result.old_where or '')}",
                  f"candidate_page = {_toml_str('' if page is None else page_ref(page))}",
                  f"all_pages = [{', '.join(_toml_str(page_ref(p)) for p in all_pages(quote.text, pages))}]",
                  f"candidate_excerpt = {_toml_str(excerpt)}", 'present = ""', 'reviewer = ""', 'note = ""', ""]
    return "\n".join(lines)
