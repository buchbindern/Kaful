"""Step e: connections from machine.toml [[connection]] (M13b): one Connection per (from-component, to-component)
pair whose ends both have a binding, e.g. water_filter -> coffee_boiler and water_filter -> steam_boiler.

Understand finds flow paths, not couplings, so every connection is declared and reviewed. Its provenance is the
link's existence: the cited quote is checked against the page's raw text by code (rule 4); `manual` when found
(verbatim or near), else `deployment` with the downgrade recorded and flagged; `deployment` with no citation.
Confidence is the evidence weight of the quote's verification class (settings.understand.evidence_weights; 1 for a
declaration alone, which states only that the engineer asserts it). Its strength is the parameter parameterize step g
made, `<from_component>.<slot>`; the gates check it exists. A connection with an end that has no binding is listed
in `skipped`, not bound.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from kaful.artifacts.base import Evidence, SourceKind
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.twin_definition import Connection, TemplateBinding
from kaful.config import DeclaredConnection, MachineConfig, UnderstandSettings
from kaful.evidence import verify_quote
from kaful.templates.couplings import COUPLINGS

DECLARED_REF = "machine.toml [[connection]]"
RAW_TEXT_CLASSES = ("verbatim", "near")
FLAG_DOWNGRADED = "connection {id}: its quote is not in the cited page's raw text, so the link is only declared"
FLAG_COUPLED = "rate scaled by {source}'s damage fraction ({form}, connection {id})"


@dataclass
class Bound:
    connections: list[Connection] = field(default_factory=list)
    flags: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # "<from> -> <to>" with an end that has no binding


def connection_id(source: str, target: str) -> str:
    return f"{source}_to_{target}"


def _evidence(d: DeclaredConnection, index: DocumentIndex | None, u: UnderstandSettings) -> Evidence | None:
    if d.quote is None or d.doc_id is None or d.page is None:
        return None
    pages = {(p.doc_id, p.page): p for p in index.pages} if index is not None else {}
    page = pages.get((d.doc_id, d.page))
    if page is None:
        return Evidence(doc_id=d.doc_id, page=d.page, quote=d.quote, verification="unsupported")
    verdict = verify_quote(d.quote, page, pages.get((page.doc_id, page.page + 1)), u.evidence_near_ratio,
                           u.evidence_candidate_overlap)
    return Evidence(doc_id=page.doc_id, page=page.page, page_label=page.label, quote=d.quote, verification=verdict)


def bind_connections(machine: MachineConfig, bindings: Sequence[TemplateBinding], index: DocumentIndex | None,
                     u: UnderstandSettings) -> Bound:
    out = Bound()
    binding_of = {b.component: b.id for b in bindings}
    for d in sorted(machine.connection, key=lambda d: (d.from_component, d.form)):
        ev = _evidence(d, index, u)
        for target in d.to_components:
            if d.from_component not in binding_of or target not in binding_of:
                out.skipped.append(f"{d.from_component} -> {target}")
                continue
            cid = connection_id(d.from_component, target)
            declared = f"{DECLARED_REF} {d.from_component} -> {target}"
            source: SourceKind = "deployment"
            downgraded: SourceKind | None = None
            ref = declared
            if ev is not None and ev.verification in RAW_TEXT_CLASSES:
                source, ref = "manual", f"{ev.doc_id} p{ev.page} (label {ev.page_label})"
            elif ev is not None:
                downgraded = "manual"
                out.flags.setdefault(target, []).append(FLAG_DOWNGRADED.format(id=cid))
            out.connections.append(Connection(
                id=cid, from_binding=binding_of[d.from_component], to_binding=binding_of[target], form=d.form,
                form_version=COUPLINGS[d.form].version,
                params={s.name: f"{d.from_component}.{s.name}" for s in COUPLINGS[d.form].params},
                basis="declared", justification=d.reason, source=source, ref=ref, downgraded_from=downgraded,
                evidence=(ev,) if ev is not None else (),
                confidence=u.evidence_weights[ev.verification] if ev is not None else 1.0))
            out.flags.setdefault(target, []).append(FLAG_COUPLED.format(source=d.from_component, form=d.form,
                                                                        id=cid))
    out.connections.sort(key=lambda c: c.id)
    return out
