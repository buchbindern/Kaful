"""One id per component across samples (used by b_merge_samples; approved 2026-11-07).

Samples name the same part differently (eversys: brew_chamber, brewing_chamber, brewing_unit).
Two ids are joined when
- their names have the same form: the same words once a plural "s" and an "-ing" are dropped
  (brew_chamber, brewing_chamber); or
- they quote the same passage (same page, one quote inside the other) and the passage does not
  name both of them as phrases (a quote that says "coffee boiler and steam boiler" is about two
  parts).
Two ids that one sample lists side by side are never joined: that sample says they are
different parts (left and right grinders). The id kept is the one most samples used, then the
one its own quotes name, then the first alphabetically; the others become its aliases, and every
merge is reported with the reason it was made.
"""

import re
from collections.abc import Sequence

from kaful.prompts.understand.components.models import ExtractedComponent

SUFFIXES = (("ing", 6), ("s", 4))  # suffix, the shortest word it is dropped from


def _stem(word: str) -> str:
    for suffix, shortest in SUFFIXES:
        if len(word) >= shortest and word.endswith(suffix):
            word = word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _names(words: list[str], phrase: list[str]) -> bool:
    return any(words[i:i + len(phrase)] == phrase for i in range(len(words) - len(phrase) + 1))


class _Groups:
    """Union-find over ids that refuses to join groups any sample lists side by side."""

    def __init__(self, ids: Sequence[str], apart: set[frozenset[str]]) -> None:
        self.parent = {i: i for i in ids}
        self.apart = apart

    def find(self, i: str) -> str:
        while self.parent[i] != i:
            i = self.parent[i]
        return i

    def members(self, root: str) -> list[str]:
        return [i for i in self.parent if self.find(i) == root]

    def join(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb or any(frozenset((x, y)) in self.apart for x in self.members(ra) for y in self.members(rb)):
            return False
        self.parent[max(ra, rb)] = min(ra, rb)
        return True


def _links(items: dict[str, list[ExtractedComponent]]) -> list[tuple[str, str, str]]:
    """Candidate joins (a, b, reason), name forms first, each in sorted order."""
    ids = sorted(items)
    form = {i: _words(i.replace("_", " ")) for i in ids}
    links = [(a, b, "same name form") for n, a in enumerate(ids) for b in ids[n + 1:] if form[a] == form[b]]
    quotes = {i: sorted({(q.doc_id, q.page, " ".join(_words(q.quote))) for c in items[i] for q in c.evidence})
              for i in ids}
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            for doc, page, qa in quotes[a]:
                shared = [qb for d, p, qb in quotes[b] if (d, p) == (doc, page) and (qa in qb or qb in qa)]
                words = qa.split()
                if shared and not (_names(words, form[a]) and _names(words, form[b])):
                    links.append((a, b, f"same quote, {doc} p{page}"))
                    break
    return links


def _keep(group: list[str], items: dict[str, list[ExtractedComponent]], counts: dict[str, int]) -> str:
    def named(i: str) -> bool:
        return any(_names(_words(q.quote), _words(i.replace("_", " "))) for c in items[i] for q in c.evidence)
    return min(group, key=lambda i: (-counts[i], not named(i), i))


def canonical_ids(samples: Sequence[Sequence[ExtractedComponent]]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Each id -> the id it is merged into (itself if none), and one line per merge with its reasons."""
    items: dict[str, list[ExtractedComponent]] = {}
    counts: dict[str, int] = {}
    apart: set[frozenset[str]] = set()
    for sample in samples:
        here = sorted({c.id for c in sample})
        for c in sample:
            items.setdefault(c.id, []).append(c)
        for i in here:
            counts[i] = counts.get(i, 0) + 1
        apart |= {frozenset((a, b)) for n, a in enumerate(here) for b in here[n + 1:]}
    groups = _Groups(sorted(items), apart)
    reasons: dict[str, str] = {}
    for a, b, why in _links(items):
        if groups.join(a, b):
            reasons.setdefault(a, why)
            reasons.setdefault(b, why)
    mapping, merges = {}, []
    for root in sorted({groups.find(i) for i in items}):
        group = sorted(groups.members(root))
        kept = _keep(group, items, counts)
        mapping.update({i: kept for i in group})
        others = [i for i in group if i != kept]
        if others:
            merges.append(f"{kept} <- " + ", ".join(f"{i} ({reasons[i]})" for i in others))
    return mapping, tuple(merges)
