"""Step b: the life statements of the full-model components, read back from the pages scope cited (no LLM).

For each life signal scope found (kinds `stated` and `machine_wide`), the quote is found again on its page:
- a schedule-table row (kaful.doctext.table_rows) gives a number under a header that names a unit ("4000"
  under "Running hours") and a period ("Yearly", "2-yearly", or by ditto);
- a sentence gives "number unit" ("every 2000 hours").
A value in the machine's time unit is used as is (pint converts h / min / s for an hour clock; "product"
matches "products"); a calendar period (day ... year) is kept as printed and needs a duty cycle (step c).
- kind: the first kind of settings.kind_words whose word prefixes the quote has (replace -> replacement
  interval; check, clean, inspect -> inspection interval, which bounds nothing); scope only passes quotes with
  a life word, so a quote with none of them is a service interval.
- consumable: a replacement interval of a component a word of whose name ends with a consumable word.
- source: `manual` only for a verbatim quote (hard rule 4). Otherwise it is downgraded to llm_judgement and,
  on a scanned page (no text layer), used from the LLM transcription with a wider prior and flagged; on a
  page with a text layer it is recorded but not used (Natalie 2026-11-09).
  Confidence is the understand evidence weight of the quote's verification class (code, rule 15).
machine.toml [[life]] declarations, and [[maintenance]] intervals (clean, descale: a cleaning interval;
replace: a replacement interval), become statements with source `deployment`. A `life_model` signal has no
value (a model, not a life): its component is flagged for ratings (step c).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pint

from kaful.artifacts.base import Evidence
from kaful.artifacts.document_index import DocumentIndex, PageText
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.param_set import LifeKind, LifeStatement
from kaful.artifacts.scope import LifeEvidence, Scope
from kaful.config import KafulSettings, MachineConfig, ParameterizeSettings
from kaful.doctext import TABLE_LAYER_KINDS, sentences, table_rows, words
from kaful.units import UnitError, parse_unit, registry

NUMBER_RE = re.compile(r"^\d[\d,.'’]*$")
AMOUNT_RE = re.compile(r"(?<![\w.])(\d[\d,.'’]*)\s*([A-Za-z]+)\b")
PERIOD_RE = re.compile(r"^(?:(\d+)\s*-\s*)?([a-z]+)$")
BOUNDS: dict[str, str] = {"rated_life": "point", "mtbf": "point", "replacement_interval": "lower",
                          "service_interval": "lower", "cleaning_interval": "lower", "inspection_interval": "none"}
DECLARED_KINDS = {"clean": "cleaning_interval", "descale": "cleaning_interval", "replace": "replacement_interval"}


@dataclass
class Read:
    statements: list[LifeStatement] = field(default_factory=list)
    notes: dict[str, list[str]] = field(default_factory=dict)  # component -> what could not be read


def _number(text: str) -> float | None:
    clean = text.replace(",", "").replace("'", "").replace("’", "")
    try:
        return float(clean) if NUMBER_RE.match(text.strip()) else None
    except ValueError:
        return None


def in_time_unit(value: float, unit: str, time_unit: str, settings: ParameterizeSettings) -> float | None:
    """`value` `unit` in the machine's time unit, or None for a calendar period or a foreign unit."""
    u = unit.lower()
    if u in settings.calendar_days or u.rstrip("s") in settings.calendar_days:
        return None
    if u.rstrip("s") == time_unit.lower().rstrip("s"):
        return value
    try:
        return float(registry().Quantity(value, parse_unit(u)).to(parse_unit(time_unit)).magnitude)
    except (UnitError, pint.errors.DimensionalityError):
        return None


def _period(text: str, settings: ParameterizeSettings) -> tuple[float, str] | None:
    m = PERIOD_RE.match(text.strip().lower())
    if m and m.group(2) in settings.calendar_days:
        return float(m.group(1) or 1), m.group(2)
    return None


def _amounts(page: PageText, quote: str, unit_words: set[str], settings: ParameterizeSettings) -> list[tuple[float, str]]:
    """(value, unit) the quote states: table-row cells under unit headers and its period, or sentence amounts."""
    found: list[tuple[float, str]] = []
    for layer in (layer for layer in page.layers if layer.kind in TABLE_LAYER_KINDS):
        for row in (r for r in table_rows(layer.text) if r.text == quote):
            for cell, head in zip(row.cells, row.header, strict=False):
                value, hit = _number(cell), sorted(set(words(head)) & unit_words)
                if value is not None and hit:
                    found.append((value, hit[0]))
            for text in (row.cells[0] if row.cells else "", row.inherited):
                period = _period(text, settings)
                if period is not None:
                    found.append(period)
    if not found:
        for sentence in (s for s in sentences(quote) if s):
            for number, unit in AMOUNT_RE.findall(sentence):
                value = _number(number)
                if value is not None and unit.lower() in unit_words | {u + "s" for u in unit_words}:
                    found.append((value, unit.lower()))
    return sorted(set(found))


def kind_of(quote: str, settings: ParameterizeSettings) -> LifeKind:
    said = words(quote)
    for kind, prefixes in settings.kind_words.items():
        if any(w.startswith(tuple(prefixes)) for w in said):
            return kind  # type: ignore[return-value]
    return "service_interval"


def _statement(sid: str, component: str, consumable_name: bool, e: LifeEvidence, ev: Evidence, value: float,
               unit: str, time_unit: str, s: KafulSettings, scanned: bool) -> LifeStatement:
    p = s.parameterize
    kind = kind_of(ev.quote, p)
    verbatim = ev.verification == "verbatim"
    transcribed = not verbatim and scanned and ev.verification == "in_layer"
    return LifeStatement(
        id=sid, component=component, kind=kind, basis=e.basis, value=value, unit=unit, bound=BOUNDS[kind],
        in_time_unit=in_time_unit(value, unit, time_unit, p), usable=verbatim or transcribed, transcribed=transcribed,
        consumable=consumable_name and kind == "replacement_interval",
        source="manual" if verbatim else "llm_judgement", downgraded_from=None if verbatim else "manual",
        ref=e.ref, confidence=s.understand.evidence_weights[ev.verification], evidence=(ev,))


def _best(amounts: Sequence[tuple[float, str]], time_unit: str, p: ParameterizeSettings) -> tuple[float, str]:
    """The amount in the time unit if there is one (a schedule's running hours beat its calendar period)."""
    direct = [a for a in amounts if in_time_unit(a[0], a[1], time_unit, p) is not None]
    return (direct or list(amounts))[0]


def read(index: DocumentIndex, model: MachineModel, scope: Scope, machine: MachineConfig,
         settings: KafulSettings) -> Read:
    p, time_unit = settings.parameterize, machine.confirmed_time_unit()
    pages = {(pg.doc_id, pg.page): pg for pg in index.pages}
    unit_words = {words(u)[0] for u in settings.scope.life_units if words(u)} | set(p.calendar_days)
    components = {c.id: c for c in scope.full_model_components(model)}
    out = Read()
    for cid, component in sorted(components.items()):
        consumable = any(w.endswith(tuple(p.consumable_words)) for w in words(component.name))
        decision = next(d for d in scope.decisions if d.component == cid)
        numbered: dict[str, int] = {}
        for e in (x for x in decision.signals.life if x.basis in ("stated", "machine_wide")):
            assert e.evidence is not None
            n = numbered[e.basis] = numbered.get(e.basis, 0) + 1
            page = pages[(e.evidence.doc_id, e.evidence.page)]
            amounts = _amounts(page, e.evidence.quote, unit_words, p)
            if not amounts:
                out.notes.setdefault(cid, []).append(f"no value read from {e.ref}: “{e.evidence.quote}”")
                continue
            value, unit = _best(amounts, time_unit, p)
            out.statements.append(_statement(f"{cid}_{e.basis}_{n}", cid, consumable, e, e.evidence, value, unit,
                                             time_unit, settings, page.page_class == "scanned"))
        for m in (m for m in machine.maintenance if m.component == cid and m.interval is not None):
            assert m.interval is not None and m.unit is not None
            kind = DECLARED_KINDS[m.action]
            out.statements.append(LifeStatement(
                id=f"{cid}_maintenance_{m.action}", component=cid, kind=kind, basis="declared",
                value=m.interval, unit=m.unit, bound="lower", in_time_unit=in_time_unit(m.interval, m.unit, time_unit, p),
                consumable=consumable and kind == "replacement_interval", source="deployment",
                ref=f"machine.toml [[maintenance]] {cid} {m.action}", confidence=1.0))
        for n, d in enumerate(d for d in machine.life if d.component == cid):
            out.statements.append(LifeStatement(
                id=f"{cid}_declared_{n + 1}", component=cid, kind=d.kind, basis="declared", value=d.value, unit=d.unit,
                bound=BOUNDS[d.kind], in_time_unit=in_time_unit(d.value, d.unit, time_unit, p),
                consumable=consumable and d.kind == "replacement_interval", source="deployment",
                ref=f"machine.toml [[life]] {d.kind}", confidence=1.0))
    return out
