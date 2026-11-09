"""One id per component, one name per channel and one mechanism per fault mode, across samples (used by b_merge_samples;
approved 2026-11-07, extended 2026-11-08).

Samples name the same thing differently (eversys: brew_chamber, brewing_chamber, brewing_unit;
flow and ticks_per_second for one status-bar reading). Two ids are joined when
- their names have the same form: the same words once a plural "s" and an "-ing" are dropped,
  and, for components, a trailing generic head word (`merge_generic_heads`: grinder,
  grinder module); or
- they quote the same passage (same page, one quote inside the other) and the passage does not
  name both of them as separate phrases (a quote that says "coffee boiler and steam boiler" is
  about two parts; "bean hopper with lid" names one); or
- for channels, a quote must be the same, not one inside the other, and must not name both
  printed names (a status-bar line quoted whole names every reading on it); and they may also join on the same reading quoted on
  different pages of one document (a screen repeated through the manual), when the reading
  occurs at most once in any line of each page ("0,0 bar" twice in one status bar identifies
  no single channel).
- for components, one is an abbreviation of the other (added M7c, 2026-11-09: CDV dpc / digital
  process controller): its id is the initials of the other's words and a printed name or quote of either
  prints the full words followed by it ("Digital Process Controller (DPC)"); a full id that ends in its
  own initials (digital_temperature_controller_dtc) has the form of the id without them.
- for components, after the joins above, a series or type of another built component folds into it
  (`variant_base`; added M7c, 2026-11-09: a manual for a product line lists CNC spindle_motor_alpha_i_series
  and spm_type_a beside spindle_motor and spm). This fold overrides "listed side by side".
Channels join only when their units agree and both or neither are numeric; fault modes only
within one component (added 2026-11-08: coffee_outlet clogging / residue buildup, one p14 sentence). Two ids that one sample lists side by side
are never joined: that sample says they are different (left and right grinders). The id kept
is the one most samples used, then the one its own quotes name, then the first alphabetically;
the others become its aliases, and every merge is reported with the reason it was made.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from kaful.artifacts.document_index import PageText
from kaful.config import UnderstandSettings
from kaful.evidence import tokens

SUFFIXES = (("ing", 6), ("s", 4))  # suffix, the shortest word it is dropped from

NUMERIC = ("float", "int")
Quoted = tuple[str, int, str]  # doc_id, page, quote as words joined by spaces
Occurrences = Callable[[str, int, str], int]  # how often a quote occurs on (doc_id, page)


def _stem(word: str) -> str:
    for suffix, shortest in SUFFIXES:
        if len(word) >= shortest and word.endswith(suffix):
            word = word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def _spans(words: list[str], phrase: list[str]) -> list[range]:
    n = len(phrase)
    return [range(i, i + n) for i in range(len(words) - n + 1) if n and words[i:i + n] == phrase]


def _names(words: list[str], phrase: list[str]) -> bool:
    return bool(_spans(words, phrase))


def _names_both(words: list[str], a: list[str], b: list[str]) -> bool:
    """Whether the passage names both phrases at places that do not overlap."""
    return any(not set(x) & set(y) for x in _spans(words, a) for y in _spans(words, b))


@dataclass(frozen=True)
class Kind:
    """How one kind of item is compared: its id, generic head words, and when two may join."""

    id: Callable[[Any], str]
    phrase: Callable[[Any], str] = lambda c: c.id.replace("_", " ")  # how a quote names the item
    heads: tuple[str, ...] = ()
    compatible: Callable[[Any, Any], bool] = lambda a, b: True
    occurrences: Occurrences | None = None  # set for channels: a repeated reading may join across pages
    label: Callable[[Any], str] | None = None  # components: the printed name, read for abbreviations
    exact_quotes: bool = False  # channels: the same quote, not one inside the other


class Pages(Protocol):
    def page(self, doc_id: str, page: int) -> PageText | None: ...


def page_occurrences(pages: Pages) -> Occurrences:
    """How often a quote occurs on a page: the most in any one line of its texts (raw text or a layer;
    ingest writes a figure's text once as a line and again as a list, which is not a second reading)."""
    def count(doc_id: str, page: int, quote: str) -> int:
        found = pages.page(doc_id, page)
        if found is None:
            return 0
        q = tokens(quote)
        texts = (found.text, *(layer.text for layer in found.layers))
        return max((len(_spans(tokens(line), q)) for t in texts for line in t.splitlines()), default=0)
    return count


DEFAULT_HEADS: tuple[str, ...] = UnderstandSettings.model_fields["merge_generic_heads"].default
DEFAULT_VARIANT_WORDS: tuple[str, ...] = UnderstandSettings.model_fields["merge_variant_words"].default


def component_kind(heads: Sequence[str] = DEFAULT_HEADS) -> Kind:
    return Kind(id=lambda c: c.id, heads=tuple(_stem(h) for h in heads), label=lambda c: c.name)


def fault_mode_id(f: Any) -> str:
    return f"{f.component} {' '.join(tokens(f.mechanism))}"


def fault_mode_kind() -> Kind:
    """Fault modes join only within one component, on the same quote that does not name both mechanisms."""
    return Kind(id=fault_mode_id, phrase=lambda f: f.mechanism, compatible=lambda a, b: a.component == b.component)


def channel_kind(occurrences: Occurrences) -> Kind:
    def compatible(a: Any, b: Any) -> bool:
        return a.unit == b.unit and (a.dtype in NUMERIC) == (b.dtype in NUMERIC)
    return Kind(id=lambda c: c.name, phrase=lambda c: c.printed_name, compatible=compatible,
                occurrences=occurrences, exact_quotes=True)


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


def _initials(words: Sequence[str]) -> str:
    return "".join(w[0] for w in words)


def _form(i: str, heads: tuple[str, ...], abbreviations: bool = False) -> list[str]:
    words = _words(i.replace("_", " "))
    if abbreviations and len(words) > 2 and words[-1] == _initials(words[:-1]):
        words = words[:-1]
    while len(words) > 1 and words[-1] in heads:
        words = words[:-1]
    return words


def generic_match(i: str, others: Sequence[str], heads: Sequence[str] = DEFAULT_HEADS) -> str | None:
    """The only id of `others` whose form ends in `i`'s one-word form (filters -> inline_gas_filter), else None."""
    form = _form(i, tuple(_stem(h) for h in heads))
    if len(form) != 1:
        return None
    hits = [o for o in sorted(others) if _form(o, tuple(_stem(h) for h in heads))[-1:] == form]
    return hits[0] if len(hits) == 1 else None


def variant_base(i: str, texts: Sequence[str], others: Sequence[str], heads: Sequence[str] = DEFAULT_HEADS,
                 markers: Sequence[str] = DEFAULT_VARIANT_WORDS) -> str | None:
    """The id of `others` that `i` is a series or type of, else None: its name form is a run of i's, i's other
    words include a marker word (series, type) and its own words none, and one of i's `texts` (printed names,
    quotes) prints all its words and the marker. The longest such id wins (spm_amplifier_type_a ->
    spm_amplifier, not spm); a tie folds nowhere."""
    stems, mark = tuple(_stem(h) for h in heads), {_stem(m) for m in markers}
    form = _form(i, stems)
    found = []
    for o in sorted(set(others) - {i}):
        base = _form(o, stems)
        rest = [w for n, w in enumerate(form) if not any(n in run for run in _spans(form, base))]
        printed = any(set(base) <= set(_words(t)) and mark & set(_words(t)) for t in texts)
        if _spans(form, base) and mark & set(rest) and not mark & set(base) and printed:
            found.append((len(base), o))
    longest = [o for n, o in found if n == max(found)[0]] if found else []
    return longest[0] if len(longest) == 1 else None


def variant_bases(samples: Sequence[Sequence[Any]], heads: Sequence[str] = DEFAULT_HEADS,
                  markers: Sequence[str] = DEFAULT_VARIANT_WORDS) -> tuple[dict[str, str], tuple[str, ...]]:
    """Each component id that is a series or type of another id in the samples -> that id (`variant_base`,
    against ids that are not variants themselves), and one line per base with the ids folded into it."""
    items: dict[str, list[Any]] = {}
    for c in (c for sample in samples for c in sample):
        items.setdefault(c.id, []).append(c)
    texts = {i: sorted({c.name for c in cs} | {q.quote for c in cs for q in c.evidence}) for i, cs in items.items()}
    found = {i: variant_base(i, texts[i], sorted(items), heads, markers) for i in sorted(items)}
    folds = {i: b for i, b in found.items() if b is not None and found[b] is None}
    lines = [f"{b} <- " + ", ".join(f"{i} (a series or type of it)" for i in sorted(i for i in folds if folds[i] == b))
             for b in sorted(set(folds.values()))]
    return folds, tuple(lines)


def _abbreviates(short: list[str], full: list[str], texts: list[list[str]]) -> bool:
    """`short` is the initials of `full`, and some text prints the full words followed by it."""
    return (len(short) == 1 and len(short[0]) >= 2 and len(full) >= 2 and _initials(full) == short[0]
            and any(_names(words, full + short) for words in texts))


def _same_quote(a: list[Quoted], b: list[Quoted], fa: list[str], fb: list[str], exact: bool) -> str | None:
    for doc, page, qa in a:
        for d, p, qb in b:
            inside = qa == qb if exact else (qa in qb or qb in qa)
            if (d, p) == (doc, page) and inside and not _names_both(qa.split(), fa, fb):
                return f"same quote, {doc} p{page}"
    return None


def _same_reading(a: list[Quoted], b: list[Quoted], count: Occurrences) -> str | None:
    for doc, page, qa in a:
        for d, p, qb in b:
            if d == doc and p != page and qa == qb and count(doc, page, qa) <= 1 and count(d, p, qb) <= 1:
                return f"same reading, {doc} p{min(page, p)} and p{max(page, p)}"
    return None


def _named(versions: list[Any], kind: Kind, form: list[str]) -> list[str]:
    """The words a quote would name the item by: its printed name for channels, its id form for components."""
    phrase = _words(kind.phrase(versions[0]))
    return form if phrase == _words(kind.id(versions[0]).replace("_", " ")) else phrase


def _links(items: dict[str, list[Any]], kind: Kind) -> list[tuple[str, str, str]]:
    """Candidate joins (a, b, reason), name forms first, each in sorted order."""
    ids = sorted(items)
    form = {i: _form(i, kind.heads, kind.label is not None) for i in ids}
    quotes = {i: sorted({(q.doc_id, q.page, " ".join(_words(q.quote))) for c in items[i] for q in c.evidence})
              for i in ids}
    raw = {i: sorted({(q.doc_id, q.page, q.quote) for c in items[i] for q in c.evidence}) for i in ids}
    pairs = [(a, b) for n, a in enumerate(ids) for b in ids[n + 1:] if kind.compatible(items[a][0], items[b][0])]
    links = [(a, b, "same name form") for a, b in pairs if form[a] == form[b]]
    if kind.label is not None:
        label = kind.label
        texts = {i: [_words(label(c)) for c in items[i]] + [q.split() for _, _, q in quotes[i]] for i in ids}
        links += [(a, b, "abbreviation") for a, b in pairs
                  if _abbreviates(form[a], form[b], texts[a] + texts[b])
                  or _abbreviates(form[b], form[a], texts[a] + texts[b])]
    for a, b in pairs:
        why = _same_quote(quotes[a], quotes[b], _named(items[a], kind, form[a]), _named(items[b], kind, form[b]),
                          kind.exact_quotes)
        if why is None and kind.occurrences is not None:
            why = _same_reading(raw[a], raw[b], kind.occurrences)
        if why is not None:
            links.append((a, b, why))
    return links


def _keep(group: list[str], items: dict[str, list[Any]], counts: dict[str, int]) -> str:
    def named(i: str) -> bool:
        return any(_names(_words(q.quote), _words(i.replace("_", " "))) for c in items[i] for q in c.evidence)
    return min(group, key=lambda i: (-counts[i], not named(i), i))


def canonical_ids(samples: Sequence[Sequence[Any]], kind: Kind | None = None,
                  ) -> tuple[dict[str, str], tuple[str, ...]]:
    """Each id -> the id it is merged into (itself if none), and one line per merge with its reasons.

    `kind` defaults to components with the default generic head words."""
    kind = kind or component_kind()
    items: dict[str, list[Any]] = {}
    counts: dict[str, int] = {}
    apart: set[frozenset[str]] = set()
    for sample in samples:
        here = sorted({kind.id(c) for c in sample})
        for c in sample:
            items.setdefault(kind.id(c), []).append(c)
        for i in here:
            counts[i] = counts.get(i, 0) + 1
        apart |= {frozenset((a, b)) for n, a in enumerate(here) for b in here[n + 1:]}
    groups = _Groups(sorted(items), apart)
    reasons: dict[str, str] = {}
    for a, b, why in _links(items, kind):
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
