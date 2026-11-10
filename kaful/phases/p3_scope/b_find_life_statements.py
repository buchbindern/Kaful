"""Step b: the life signal (L) per component, and the machine-wide life statements, found by code (no LLM).

- stated: a sentence or table row of a machine document (kinds in understand.machine_kinds) that names the
  component (helpers/text.names), has a life word, and states a number with a time or usage unit (in the
  text, or under a table header that names the unit) or a period word (daily ... yearly, also by ditto).
- life_model: a page of a reference document (any other kind) that the component's counted fault modes cite,
  that prints every word of one of the component's names, and has a line with a life-rating or life-model
  phrase (L10, rating life, dynamic load rating, fatigue life ...). The line is the quote. The document must
  be cited by the component's own fault modes, so a bearing catalogue never gives the spindle motor a life.
- declared: machine.toml [[life]] for the component.
A stated unit that names no component, or names the machine (`machine_words`; eversys p84 "The service of
Eversys coffee machines ... 12 months or every 50'000 products"), is a machine-wide statement, linked to the
components it names and those whose counted fault modes cite the same page. For the components citing the
page it also counts for L, as basis `machine_wide` (a lower bound on life, flagged; approved 2026-11-09).
Only a statement that names the machine, or prints no word of any component's name, links through a cited
page: one that prints a name word of a part (air_compressor p29 "Have all flexibles/flexible joints
inspected", for discharge_flexible) is about that part. It is linked to the parts it loosely names and gives
no one a life (approved 2026-11-09).
Every quote is checked against its page (kaful.evidence.verify_quote); a table row read from an LLM layer
may be `in_layer`, which is kept and shown: scope only flags that a life exists, M9 verifies values.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from kaful.artifacts.base import Evidence
from kaful.artifacts.document_index import DocumentIndex, PageText
from kaful.artifacts.machine_model import Component, MachineModel
from kaful.artifacts.scope import LifeEvidence, MachineLifeStatement
from kaful.config import MachineConfig, ScopeSettings, UnderstandSettings
from kaful.evidence import verify_quote
from kaful.phases.p3_scope.a_collect_signals import counts
from kaful.phases.p3_scope.helpers.text import (
    TABLE_LAYER_KINDS,
    Row,
    has_phrase,
    names,
    sentences,
    stem,
    table_rows,
    words,
)
from kaful.phases.p3_scope.helpers.text import named as named_by

NUMBER = r"\d[\d,.'’]*"


@dataclass(frozen=True)
class Found:
    by_component: dict[str, list[LifeEvidence]]
    machine_wide: list[MachineLifeStatement]


def _ref(page: PageText) -> str:
    label = f" (label {page.label})" if page.label and page.label != str(page.page) else ""
    return f"{page.doc_id} p{page.page}{label}"


class _Rules:
    def __init__(self, settings: ScopeSettings) -> None:
        units = "|".join(sorted((re.escape(u) for u in settings.life_units), key=len, reverse=True))
        self.amount = re.compile(rf"(?<![\w.]){NUMBER}\s*(?:{units})\b", re.IGNORECASE)
        self.number = re.compile(rf"(?<![\w.]){NUMBER}(?![\w])")
        self.unit_words = {words(u)[0] for u in settings.life_units if words(u)}
        self.life_words = tuple(sorted(words(w)[0] for w in settings.life_words))
        self.periods = {words(p)[0] for p in settings.life_periods}
        self.models = [(p, words(p)) for p in settings.life_model_phrases]
        self.max_words = settings.statement_max_words
        self.machine = {words(w)[0] for w in settings.machine_words}
        self.min_name_word = settings.name_word_min_chars

    def has_life_word(self, text_words: Iterable[str]) -> bool:
        """A life word, also inflected (replace: replaced, replacement; change: changed)."""
        return any(w.startswith(self.life_words) for w in text_words)

    def sentence(self, text: str) -> bool:
        if len(words(text)) > self.max_words:
            return False
        return self.has_life_word(words(text)) and bool(self.amount.search(text))

    def row(self, row: Row) -> bool:
        w = set(words(row.text))
        if not self.has_life_word(w):
            return False
        if w & self.periods or set(words(row.inherited)) & self.periods:
            return True
        return any(self.number.fullmatch(cell.strip()) and set(words(head)) & self.unit_words
                   for cell, head in zip(row.cells, row.header, strict=False))

    def life_model(self, lines: Sequence[str]) -> tuple[str, str] | None:
        """(phrase, line): the line with the strongest life-model phrase (earliest in the settings), first such
        line on ties."""
        split = [(ln.strip(), words(ln)) for ln in lines]
        for phrase, phrase_words in self.models:
            hit = next((ln for ln, w in split if has_phrase(w, phrase_words)), None)
            if hit:
                return phrase, hit
        return None


def _evidence(page: PageText, following: PageText | None, quote: str, u: UnderstandSettings) -> Evidence:
    return Evidence(doc_id=page.doc_id, page=page.page, page_label=page.label, quote=quote,
                    verification=verify_quote(quote, page, following, u.evidence_near_ratio,
                                              u.evidence_candidate_overlap))


def _units(page: PageText, rules: _Rules) -> list[tuple[str, str]]:
    """(quote, note) for each sentence or table row of the page that states a life or interval."""
    found = [(s, "") for s in sentences(page.text) if rules.sentence(s)]
    for layer in (layer for layer in page.layers if layer.kind in TABLE_LAYER_KINDS):
        found += [(r.text, f"table row; period {r.inherited!r} by ditto" if r.inherited else "table row")
                  for r in table_rows(layer.text) if rules.row(r)]
    unique: dict[str, str] = {}
    for quote, note in found:
        unique.setdefault(quote, note)
    return list(unique.items())


def _cited(model: MachineModel) -> tuple[dict[str, set[str]], dict[tuple[str, int], set[str]]]:
    """Per component, the documents its counted fault modes cite; per (doc, page), the components citing it."""
    docs: dict[str, set[str]] = {}
    pages: dict[tuple[str, int], set[str]] = {}
    for mode in (m for m in model.fault_modes if counts(m)):
        for ev in mode.evidence:
            docs.setdefault(mode.component, set()).add(ev.doc_id)
            pages.setdefault((ev.doc_id, ev.page), set()).add(mode.component)
    return docs, pages


def _name_words(forms: Sequence[Sequence[str]], heads: Sequence[str], min_chars: int) -> set[str]:
    generic = {stem(h) for h in heads}
    return {w for f in forms for w in f if len(w) >= min_chars and w not in generic}


def _stated(index: DocumentIndex, machine_docs: set[str], comps: Sequence[Component], heads: Sequence[str],
            rules: _Rules, u: UnderstandSettings, cited_pages: dict[tuple[str, int], set[str]], found: Found) -> None:
    pages = {(p.doc_id, p.page): p for p in index.pages}
    forms = {c.id: names(c, heads) for c in comps}
    name_words = {c.id: _name_words(forms[c.id], heads, rules.min_name_word) for c in comps}
    for page in sorted((p for p in index.pages if p.doc_id in machine_docs), key=lambda p: (p.doc_id, p.page)):
        following = pages.get((page.doc_id, page.page + 1))
        for quote, note in _units(page, rules):
            hits = sorted(c.id for c in comps if named_by(quote, c, forms[c.id]))
            ev = _evidence(page, following, quote, u)
            if hits and not set(words(quote)) & rules.machine:
                for cid in hits:
                    found.by_component[cid].append(LifeEvidence(basis="stated", ref=_ref(page), evidence=ev, note=note))
                continue
            said = set(words(quote))
            loose = sorted(cid for cid in name_words if name_words[cid] & said)
            about_a_part = not hits and bool(loose) and not said & rules.machine
            citing = set() if about_a_part else cited_pages.get((page.doc_id, page.page), set())
            linked = loose if about_a_part else sorted({*hits, *citing})
            found.machine_wide.append(MachineLifeStatement(evidence=ev, linked=tuple(linked)))
            for cid in sorted(citing):
                found.by_component[cid].append(LifeEvidence(basis="machine_wide", ref=_ref(page), evidence=ev,
                                                            note="machine-wide statement on a page its fault modes cite"))


def _models(index: DocumentIndex, reference_docs: set[str], comps: Sequence[Component], heads: Sequence[str],
            rules: _Rules, u: UnderstandSettings, cited_docs: dict[str, set[str]], found: Found) -> None:
    pages = {(p.doc_id, p.page): p for p in index.pages}
    for c in comps:
        docs = cited_docs.get(c.id, set()) & reference_docs
        forms = names(c, heads)
        for page in sorted((p for p in index.pages if p.doc_id in docs), key=lambda p: (p.doc_id, p.page)):
            on_page = set(words(page.text))
            form = next((f for f in forms if set(f) <= on_page), None)
            hit = rules.life_model(page.text.splitlines())
            if form is None or hit is None:
                continue
            ev = _evidence(page, pages.get((page.doc_id, page.page + 1)), hit[1], u)
            found.by_component[c.id].append(LifeEvidence(basis="life_model", ref=_ref(page), evidence=ev, phrase=hit[0],
                                                         note=f"the page prints every word of {' '.join(form)!r}"))


def find_life(index: DocumentIndex, model: MachineModel, machine: MachineConfig, settings: ScopeSettings,
              understand: UnderstandSettings) -> Found:
    rules = _Rules(settings)
    kinds = {d.doc_id: d.kind for d in index.documents}
    machine_docs = {d for d, k in kinds.items() if k in understand.machine_kinds}
    comps = sorted(model.components, key=lambda c: c.id)
    found = Found({c.id: [] for c in comps}, [])
    cited_docs, cited_pages = _cited(model)
    heads = understand.merge_generic_heads
    _stated(index, machine_docs, comps, heads, rules, understand, cited_pages, found)
    _models(index, set(kinds) - machine_docs, comps, heads, rules, understand, cited_docs, found)
    for d in machine.life:
        if d.component in found.by_component:
            found.by_component[d.component].append(LifeEvidence(
                basis="declared", ref=f"machine.toml [[life]] {d.kind}",
                note=f"{d.value:g} {d.unit}" + (f"; {d.note}" if d.note else "")))
    statements = {(s.evidence.doc_id, s.evidence.page, s.evidence.quote): s for s in found.machine_wide}
    return Found({k: _unique(v) for k, v in found.by_component.items()},
                 [statements[k] for k in sorted(statements)])


def _unique(items: list[LifeEvidence]) -> list[LifeEvidence]:
    keyed = {(e.basis, e.ref, e.evidence.quote if e.evidence else ""): e for e in items}
    return [keyed[k] for k in sorted(keyed)]
