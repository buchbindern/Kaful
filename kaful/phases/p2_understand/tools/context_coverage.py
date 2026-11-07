"""How much of the golden evidence a context policy shows the understand prompts (offline).

The golden evidence pages, per prompt they matter to:
- sensor_channels: pages of channel rows (candidate channels' page; legacy fields' legacy_pages,
  only when the machine has one document, since legacy pages name no document) and values rows;
- fault_modes: pages of `yes` fault modes and candidate fault modes (with their other_pages).
Components rows carry no pages, so components coverage is not measured.

A page counts as covered when a selected chunk spans it. Query vectors come from the embedding
cache (a missing one is an error, never an API call), so measuring costs nothing.

Not used during a run (a tool).
"""

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from kaful.artifacts.base import StrictModel
from kaful.artifacts.document_index import Chunk, DocumentIndex
from kaful.config import IngestSettings, RetrievalTopic, UnderstandSettings
from kaful.llm.embeddings import Embedder, Vectors
from kaful.phases.p2_understand.a_select_context import document_order, estimated_tokens, render, select_context

Page = tuple[str, int]
TOPICS: tuple[RetrievalTopic, ...] = ("fault_modes", "sensor_channels")
OTHER_PAGE = re.compile(r"\bp(\d+) \(label")


class TopicCoverage(StrictModel):
    covered: int
    total: int
    missed: tuple[str, ...] = Field(description="'doc pN (label L)', sorted by page.")


class PolicyCoverage(StrictModel):
    policy: str
    estimated_tokens: dict[str, int] = Field(description="Prompt (or 'shared') -> context estimate.")
    topics: dict[str, TopicCoverage]


def _pages(text: str) -> list[int]:
    return [int(p) for p in re.findall(r"\d+", text)]


def golden_evidence(golden: dict[str, Any], doc_ids: Sequence[str]) -> dict[str, set[Page]]:
    """Golden evidence pages per prompt (see the module docstring)."""
    single = doc_ids[0] if len(doc_ids) == 1 else None
    out: dict[str, set[Page]] = {t: set() for t in TOPICS}
    for row in golden["legacy_fields"]:
        if row["verdict"] == "channel" and single:
            out["sensor_channels"] |= {(single, p) for p in _pages(str(row.get("legacy_pages", "")))}
    for row in golden["candidate_channels"]:
        if row["verdict"] == "channel" and row.get("doc_id") and row.get("page"):
            out["sensor_channels"].add((row["doc_id"], int(row["page"])))
    for row in golden.get("values", []):
        out["sensor_channels"].add((row["doc_id"], int(row["page"])))
    for row in golden["fault_modes"] + golden.get("candidate_fault_modes", []):
        if row["verdict"] == "yes" and row.get("doc_id") and str(row.get("page", "")):
            out["fault_modes"].add((row["doc_id"], int(row["page"])))
            out["fault_modes"] |= {(row["doc_id"], int(p)) for p in OTHER_PAGE.findall(row.get("other_pages", ""))}
    return out


def _covered(chunks: Sequence[Chunk], wanted: set[Page], labels: dict[Page, str | None]) -> TopicCoverage:
    spanned = {(c.doc_id, p) for c in chunks for p in range(c.page_start, c.page_end + 1)}
    missed = sorted(wanted - spanned, key=lambda dp: (dp[0], dp[1]))
    names = tuple(f"{d} p{p}" + (f" (label {labels.get((d, p))})" if labels.get((d, p)) else "") for d, p in missed)
    return TopicCoverage(covered=len(wanted) - len(missed), total=len(wanted), missed=names)


def _chunks(index: DocumentIndex, ids: Sequence[str]) -> list[Chunk]:
    by_id = {c.chunk_id: c for c in index.chunks}
    return [by_id[i] for i in ids]


def measure(index: DocumentIndex, vectors: Vectors | None, embedder: Embedder | None, golden_path: Path,
            policies: dict[str, tuple[UnderstandSettings, bool]], ingest: IngestSettings) -> list[PolicyCoverage]:
    """Coverage of each policy: (settings, per_prompt). Per-prompt policies ask only that prompt's queries."""
    golden = tomllib.loads(golden_path.read_text(encoding="utf-8"))
    wanted = golden_evidence(golden, sorted({d.doc_id for d in index.documents}))
    labels = {(p.doc_id, p.page): p.label for p in index.pages}
    out = []
    for name, (settings, per_prompt) in policies.items():
        topics, tokens = {}, {}
        for topic in TOPICS:
            used = settings.model_copy(update={"queries": {topic: settings.queries[topic]}}) if per_prompt else settings
            context = select_context(index, vectors, embedder, used, ingest)
            tokens[topic if per_prompt else "shared"] = context.selection.estimated_tokens
            topics[topic] = _covered(_chunks(index, context.selection.chunk_ids), wanted[topic], labels)
        out.append(PolicyCoverage(policy=name, estimated_tokens=tokens, topics=topics))
    return out


def whole_text(index: DocumentIndex) -> str:
    return render(document_order(index.chunks))


def whole_estimate(index: DocumentIndex, settings: UnderstandSettings) -> int:
    return estimated_tokens(whole_text(index), settings)
