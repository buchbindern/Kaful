"""Evidence verification and provenance for understand items (hard rules 3, 4 and 15).

Each quote is checked against the raw text of the page it cites (`kaful.evidence`). An
item's claimed source follows the kind of document its best quote comes from
(manual, datasheet and catalogue -> manual; standard -> standard; literature -> literature).
It keeps that source only if at least one quote is verbatim or near in the raw text;
otherwise it is downgraded to `llm_judgement`, with `downgraded_from` set, and the
report lists it. Confidence = agreement x the weight of the best evidence class.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

from kaful.artifacts.base import Evidence, EvidenceClass, SourceKind
from kaful.artifacts.document_index import DocumentIndex, PageText
from kaful.config import UnderstandSettings
from kaful.evidence import verify_quote

ACCEPTED: frozenset[EvidenceClass] = frozenset({"verbatim", "near"})
EVIDENCE_RANK: tuple[EvidenceClass, ...] = ("verbatim", "near", "in_layer", "unsupported", "unverified")
DOCUMENT_SOURCE: Mapping[str, SourceKind] = {
    "manual": "manual", "datasheet": "manual", "catalogue": "manual", "standard": "standard",
    "literature": "literature",
}


class ProvenanceFields(TypedDict):
    source: SourceKind
    ref: str
    confidence: float
    evidence: tuple[Evidence, ...]
    downgraded_from: SourceKind | None


@dataclass(frozen=True)
class Provenance:
    source: SourceKind
    ref: str
    confidence: float
    evidence: tuple[Evidence, ...]
    downgraded_from: SourceKind | None

    def fields(self) -> ProvenanceFields:
        return {"source": self.source, "ref": self.ref, "confidence": self.confidence, "evidence": self.evidence,
                "downgraded_from": self.downgraded_from}


class PageLookup:
    """Pages by (doc_id, page), plus each document's kind."""

    def __init__(self, index: DocumentIndex) -> None:
        self.pages = {(p.doc_id, p.page): p for p in index.pages}
        self.kinds = {d.doc_id: d.kind for d in index.documents}

    def page(self, doc_id: str, page: int) -> PageText | None:
        return self.pages.get((doc_id, page))

    def following(self, doc_id: str, page: int) -> PageText | None:
        return self.pages.get((doc_id, page + 1))


def check_quote(doc_id: str, page: int, quote: str, pages: PageLookup, settings: UnderstandSettings) -> Evidence:
    cited = pages.page(doc_id, page)
    if cited is None:
        return Evidence(doc_id=doc_id, page=page, quote=quote, verification="unsupported")
    verdict = verify_quote(quote, cited, pages.following(doc_id, page), settings.evidence_near_ratio,
                           settings.evidence_candidate_overlap)
    return Evidence(doc_id=doc_id, page=page, page_label=cited.label, quote=quote, verification=verdict)


def provenance(quotes: Sequence[tuple[str, int, str]], agreement: float, pages: PageLookup,
               settings: UnderstandSettings) -> Provenance:
    evidence = tuple(check_quote(d, p, q, pages, settings) for d, p, q in quotes)
    best = min(evidence, key=lambda e: (EVIDENCE_RANK.index(e.verification), e.doc_id, e.page, e.quote))
    claimed = DOCUMENT_SOURCE.get(pages.kinds.get(best.doc_id, ""), "manual")
    accepted = best.verification in ACCEPTED
    label = f" (label {best.page_label})" if best.page_label else ""
    ref = f"{best.doc_id} p.{best.page}{label}"
    confidence = round(agreement * settings.evidence_weights[best.verification], 6)
    if accepted:
        return Provenance(claimed, ref, confidence, evidence, None)
    return Provenance("llm_judgement", ref, confidence, evidence, claimed)
