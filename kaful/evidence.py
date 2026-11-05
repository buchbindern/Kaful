"""Evidence quotes checked by code against document pages (hard rule 4).

Matching is on normalized tokens (NFKC, case, dashes and quotes folded, line-break
hyphens joined only between letters since matcher-v3 (so "10-|30 μm" stays a range), and
since matcher-v2 a decimal comma read as a decimal point: 2,0 = 2.0;
a comma before exactly three digits is a thousands separator and is kept, so 1,000 != 1.000):
`verbatim` = the quote's tokens appear contiguously; `near` = a token
window matches with SequenceMatcher ratio >= `near_ratio`.

`verify_quote` classifies one quote against its cited page (joined with the next page
of the same document, so quotes across a page break match):
- verbatim / near: found in the page's raw text, which is what rule 4 accepts;
- near: found only in the page's ruled-table markdown, which PyMuPDF derives from the
  same text layer (no LLM), so it counts like a reformatted raw-text match;
- in_layer: found only in an LLM transcription layer (a table or figure read from the
  page image); not accepted as `manual`;
- unsupported: found nowhere on the cited page.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from kaful.artifacts.base import EvidenceClass
from kaful.artifacts.document_index import PageText

QuoteMatch = Literal["verbatim", "near", "unsupported"]
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
DECIMAL_COMMA = re.compile(r"(?<=[0-9]),(?![0-9]{3}(?![0-9]))")
LINE_BREAK_HYPHEN = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[^\W\d_])")  # letter - newline - letter
MATCHER_VERSION = "matcher-v3"  # v2: decimal commas equal decimal points; v3: line-break hyphens join letters only
DASHES = re.compile(r"[‐-―−]")


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = DASHES.sub("-", text).replace("‘", "'").replace("’", "'")
    text = LINE_BREAK_HYPHEN.sub("", text)
    return [DECIMAL_COMMA.sub(".", t) if t[0].isdigit() else t for t in TOKEN_RE.findall(text)]


@dataclass(frozen=True)
class Passage:
    """Text to match against, with where it came from."""

    key: str
    text: str


class Matcher:
    def __init__(self, passages: Sequence[Passage], near_ratio: float, candidate_overlap: float) -> None:
        self.near_ratio = near_ratio
        self.candidate_overlap = candidate_overlap
        self.passages = [(p, tokens(p.text)) for p in passages]
        self.joined = [(p, f" {' '.join(t)} ", set(t)) for p, t in self.passages]

    def match(self, quote: str) -> tuple[QuoteMatch, str | None]:
        q = tokens(quote)
        if not q:
            return "unsupported", None
        needle = f" {' '.join(q)} "
        for passage, text, _ in self.joined:
            if needle in text:
                return "verbatim", passage.key
        best, where = 0.0, None
        for (passage, toks), (_, _, vocab) in zip(self.passages, self.joined, strict=True):
            if len(set(q) & vocab) < self.candidate_overlap * len(set(q)):
                continue
            ratio = _best_window(q, toks)
            if ratio > best:
                best, where = ratio, passage.key
        return ("near", where) if best >= self.near_ratio else ("unsupported", None)


def _best_window(q: list[str], toks: list[str]) -> float:
    n = len(q)
    step = max(1, n // 4)
    best = 0.0
    for start in range(0, max(1, len(toks) - n + 1), step):
        ratio = SequenceMatcher(None, q, toks[start:start + n + n // 5], autojunk=False).ratio()
        best = max(best, ratio)
    return best


def _with_next(page: PageText, following: PageText | None, text: str, next_text: str) -> str:
    return text + ("\n" + next_text if following is not None and following.doc_id == page.doc_id else "")


def verify_quote(quote: str, page: PageText, following: PageText | None, near_ratio: float,
                 candidate_overlap: float) -> EvidenceClass:
    """Classify `quote` against its cited `page` (and the page after it); see the module docstring."""
    def match(text: str, next_text: str) -> QuoteMatch:
        passage = Passage(f"{page.doc_id}:p{page.page}", _with_next(page, following, text, next_text))
        return Matcher([passage], near_ratio, candidate_overlap).match(quote)[0]

    def layers(p: PageText | None, kind: str) -> str:
        return "" if p is None else "\n".join(layer.text for layer in p.layers if layer.kind == kind)

    raw = match(page.text, following.text if following is not None else "")
    if raw != "unsupported":
        return raw
    if match(layers(page, "table_md"), layers(following, "table_md")) != "unsupported":
        return "near"
    if match(layers(page, "llm"), layers(following, "llm")) != "unsupported":
        return "in_layer"
    return "unsupported"
