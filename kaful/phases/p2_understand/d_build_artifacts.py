"""Step d: build MachineModel, SensorContract and UnderstandReport items from merged,
verified extractions.

Everything code changes or removes is recorded (`Log`) for the UnderstandReport:
references to things that were not extracted, one-sided ranges, a second usage clock,
sample periods that are not backed by a verified quote, units pint cannot parse, and
channels whose only evidence is a screenshot of the reading.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from kaful.artifacts.base import Evidence, ValueRole
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath
from kaful.artifacts.sensor_contract import SecondaryRole, SensorChannel
from kaful.artifacts.understand_report import ItemCheck, ItemKind
from kaful.config import DeclaredChannel, DeclaredComponent
from kaful.phases.p2_understand.b_merge_samples import Merged, MergedComponents, normalized
from kaful.phases.p2_understand.c_verify_quotes import ACCEPTED, Checker, Provenance, check_quote, provenance
from kaful.prompts.understand.components.models import ExtractedComponent, ExtractedFlowPath
from kaful.prompts.understand.quotes import Quote
from kaful.prompts.understand.fault_modes.models import ExtractedFaultMode
from kaful.prompts.understand.sensor_channels.models import ExtractedChannel, StatedPeriod
from kaful.units import UnitError, parse_unit, to_seconds
from kaful.value_roles import settle_role

class SamplingFields(TypedDict, total=False):
    sampling: Literal["periodic", "event_driven", "unknown"]
    sample_period_s: float
    sample_period_evidence: Evidence


class TypeFields(TypedDict, total=False):
    unit: str | None
    unit_status: Literal["not_printed"]
    range: tuple[float, float] | None
    range_evidence: Evidence | None
    categories: tuple[str, ...] | None


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
DEPLOYMENT_REF = "machine.toml [[instrumentation]] {name}"
DECLARED_COMPONENT_REF = "machine.toml [[components]] {id}"
JUDGEMENT_REF = "llm_judgement: understand.sensor_channels"


@dataclass
class Log:
    items: list[ItemCheck] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    unit_problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def item(self, kind: ItemKind, id: str, agreement: float, prov: Provenance) -> None:
        self.items.append(ItemCheck(kind=kind, id=id, agreement=agreement,
                                    evidence=tuple(e.verification for e in prov.evidence), source=prov.source,
                                    downgraded_from=prov.downgraded_from, confidence=prov.confidence))


def _reference_only(m: Merged[Any], check: Checker) -> str | None:
    """The reference documents an item is quoted from, when none of its quotes is from a machine document."""
    docs = sorted({d for d, _, _ in m.quotes})
    if any(check.pages.kinds.get(d) in check.settings.machine_kinds for d in docs):
        return None
    return ", ".join(docs)


def components(merged: Sequence[Merged[ExtractedComponent]], check: Checker, log: Log,
               aliases: dict[str, tuple[str, ...]] | None = None) -> list[Component]:
    kept = []
    for m in merged:
        refs = _reference_only(m, check)
        if refs is None:
            kept.append(m)
        else:
            log.dropped.append(f"component {m.item.id}: quoted only from reference documents ({refs}); "
                               "not a part of this machine")
    merged = kept
    ids = {m.item.id for m in merged}
    parents = {m.item.id: (m.item.parent if m.item.parent in ids else None) for m in merged}
    for m in merged:
        if m.item.parent is not None and parents[m.item.id] is None:
            log.dropped.append(f"component {m.item.id}: parent {m.item.parent!r} was not extracted")
    for start in sorted(parents):  # break parent cycles that merging can create
        seen, node = [start], parents[start]
        while node is not None and node not in seen:
            seen.append(node)
            node = parents[node]
        if node is not None:
            log.dropped.append(f"component {node}: parent {parents[node]!r} removed (cycle {' -> '.join(seen)})")
            parents[node] = None
    out = []
    for m in merged:
        prov = check.provenance(m)
        log.item("component", m.item.id, m.agreement, prov)
        out.append(Component(id=m.item.id, name=m.item.name, description=m.item.description,
                             category=m.item.category, parent=parents[m.item.id],
                             aliases=(aliases or {}).get(m.item.id, ()), **prov.fields()))
    return out


def flow_paths(merged: Sequence[Merged[ExtractedFlowPath]], nodes: set[str], check: Checker,
               log: Log) -> list[FlowPath]:
    out = []
    for m in merged:
        ends = (m.item.from_node, m.item.to_node)
        missing = [e for e in ends if e not in nodes]
        if missing:
            log.dropped.append(f"flow path {ends[0]} -> {ends[1]}: unknown end(s) {missing}")
            continue
        prov = check.provenance(m)
        path_id = f"{ends[0]}_to_{ends[1]}_{m.item.medium}"
        log.item("flow_path", path_id, m.agreement, prov)
        out.append(FlowPath(id=path_id, from_component=ends[0], to_component=ends[1], medium=m.item.medium,
                            quantity=m.item.quantity, **prov.fields()))
    return out


def _sampling(c: ExtractedChannel, check: Checker, log: Log) -> SamplingFields:
    if c.role == "event":
        return {"sampling": "event_driven"}
    period: StatedPeriod | None = c.sample_period
    if period is None:
        return {"sampling": "unknown"}
    evidence = _valued(period.evidence, period.role, check, log, f"channel {c.name}: sample period")
    if evidence.value_role == "example":
        return {"sampling": "unknown"}
    q = period.evidence
    stated = {float(n.replace(",", ".")) for n in NUMBER_RE.findall(q.quote)}
    try:
        seconds = to_seconds(period.value, period.unit)
    except UnitError as exc:
        log.notes.append(f"channel {c.name}: sample period unit: {exc}; sampling unknown")
        return {"sampling": "unknown"}
    if evidence.verification not in ACCEPTED or period.value not in stated:
        log.notes.append(f"channel {c.name}: sample period {period.value} {period.unit} not backed by its quote "
                         f"({evidence.verification}); sampling unknown")
        return {"sampling": "unknown"}
    return {"sampling": "periodic", "sample_period_s": seconds, "sample_period_evidence": evidence}


def _valued(q: Quote, proposed: ValueRole, check: Checker, log: Log, what: str) -> Evidence:
    """The quote, verified, with its value role (code may downgrade the proposal to example)."""
    evidence = check_quote(q.doc_id, q.page, q.quote, check.pages, check.settings)
    role, why = settle_role(proposed, evidence, check.pages.page(q.doc_id, q.page), check.settings)
    if why is not None:
        log.notes.append(f"{what}: role {proposed} -> example ({why})")
    if role == "example":
        log.notes.append(f"{what}: an example value, not stored")
    return evidence.model_copy(update={"value_role": role})


def _range(c: ExtractedChannel, check: Checker, log: Log) -> tuple[tuple[float, float] | None, Evidence | None]:
    if c.range_min is None or c.range_max is None:
        if c.range_min is not None or c.range_max is not None:
            log.notes.append(f"channel {c.name}: one-sided range ({c.range_min}, {c.range_max}) not stored")
        return None, None
    assert c.range_evidence is not None and c.range_role is not None  # the output model guarantees both
    evidence = _valued(c.range_evidence, c.range_role, check, log, f"channel {c.name}: range")
    return (None, None) if evidence.value_role == "example" else ((c.range_min, c.range_max), evidence)


def _type_fields(c: ExtractedChannel, check: Checker, log: Log) -> TypeFields | None:
    if c.categories and c.dtype != "category":
        log.notes.append(f"channel {c.name}: categories given for dtype {c.dtype}; dropped")
        c = c.model_copy(update={"categories": None})
    if c.dtype not in ("float", "int"):
        if c.dtype == "category" and not c.categories:
            log.dropped.append(f"channel {c.name}: category channel without categories")
            return None
        return {"categories": tuple(sorted(set(c.categories))) if c.categories else None}
    if not c.unit:
        log.notes.append(f"channel {c.name}: no unit printed; kept with unit_status not_printed and no range")
        return {"unit": None, "unit_status": "not_printed"}
    try:
        parse_unit(c.unit)
    except UnitError as exc:
        log.unit_problems.append(f"channel {c.name}: {exc}")
    span, evidence = _range(c, check, log)
    return {"unit": c.unit, "range": span, "range_evidence": evidence}


def _secondary(m: Merged[ExtractedChannel], check: Checker) -> tuple[SecondaryRole, ...]:
    """The channel's second role: verified like any quote when one is given, else an unquoted judgement."""
    c = m.item
    if c.secondary_role is None or c.secondary_reason is None:
        return ()
    if c.secondary_evidence is not None:
        q = c.secondary_evidence
        prov = provenance(((q.doc_id, q.page, q.quote),), m.agreement, check.pages, check.settings)
    else:
        confidence = round(m.agreement * check.settings.evidence_weights["unverified"], 6)
        prov = Provenance("llm_judgement", JUDGEMENT_REF, confidence, (), None)
    return (SecondaryRole(role=c.secondary_role, reason=c.secondary_reason, **prov.fields()),)


def channels(merged: Sequence[Merged[ExtractedChannel]], declared: Sequence[DeclaredChannel], check: Checker,
             log: Log) -> list[SensorChannel]:
    kept = []
    for m in merged:
        refs = _reference_only(m, check)
        if refs is None:
            kept.append(m)
        else:
            log.dropped.append(f"channel {m.item.name}: quoted only from reference documents ({refs}); "
                               "not a channel of this machine")
    merged = kept
    clocks = sorted((m for m in merged if m.item.role == "usage_clock"), key=lambda m: (-m.agreement, m.item.name))
    declared_names = {d.name for d in declared}
    out = []
    for m in merged:
        c = m.item
        if c.name in declared_names:
            log.notes.append(f"channel {c.name}: declared in machine.toml; the declaration replaces the extraction")
            continue
        role = c.role
        if role == "usage_clock" and c is not clocks[0].item:
            log.notes.append(f"channel {c.name}: second usage_clock (kept {clocks[0].item.name}); now an observation")
            role = "observation"
        typed = _type_fields(c, check, log)
        if typed is None:
            continue
        prov, shown = check.channel_provenance(m)
        if shown is not None:
            log.notes.append(f"channel {c.name}: shown in a screenshot ({prov.ref}); exists, its values are examples")
        log.item("channel", c.name, m.agreement, prov)
        out.append(SensorChannel(name=c.name, description=c.description, role=role, dtype=c.dtype,
                                 secondary_roles=_secondary(m, check), **typed, **_sampling(c, check, log),
                                 **prov.fields()))
    for d in declared:
        out.append(SensorChannel(**d.model_dump(), source="deployment", ref=DEPLOYMENT_REF.format(name=d.name),
                                 confidence=1.0))
        log.items.append(ItemCheck(kind="channel", id=d.name, agreement=1.0, evidence=(), source="deployment",
                                   confidence=1.0))
    return sorted(out, key=lambda ch: ch.name)


def fault_modes(merged: Sequence[Merged[ExtractedFaultMode]], component_ids: set[str], channel_names: set[str],
                check: Checker, log: Log) -> list[FaultMode]:
    out: dict[str, FaultMode] = {}
    for m in merged:
        f = m.item
        if f.component not in component_ids:
            log.dropped.append(f"fault mode {f.id}: unknown component {f.component!r}")
            continue
        refs = {}
        for name in ("driven_by", "observable_by"):
            given = sorted(set(getattr(f, name)))
            refs[name] = tuple(c for c in given if c in channel_names)
            for unknown in sorted(set(given) - set(refs[name])):
                log.dropped.append(f"fault mode {f.id}: {name} {unknown!r} is not a channel")
        mode_id = f.id if f.id not in out else f"{f.component}_{normalized(f.mechanism)}"
        prov = check.provenance(m)
        log.item("fault_mode", mode_id, m.agreement, prov)
        out[mode_id] = FaultMode(id=mode_id, component=f.component, name=f.name, mechanism=f.mechanism,
                                 effect=f.effect, **refs, **prov.fields())
    return [out[k] for k in sorted(out)]


def declared_components(declared: Sequence[DeclaredComponent], built: Sequence[Component],
                        log: Log) -> list[Component]:
    """The built components plus the ones machine.toml declares (source `deployment`); a declaration replaces
    an extraction of the same id, and a declared parent the merge renamed attaches to the kept id."""
    names = {d.id for d in declared}
    kept_as = {alias: c.id for c in built for alias in c.aliases} | {c.id: c.id for c in built} | {n: n for n in names}
    out = []
    for c in built:
        if c.id in names:
            log.notes.append(f"component {c.id}: declared in machine.toml; the declaration replaces the extraction")
            log.items = [i for i in log.items if (i.kind, i.id) != ("component", c.id)]
        else:
            out.append(c)
    for d in declared:
        parent = kept_as.get(d.parent) if d.parent is not None else None
        if d.parent is not None and parent is None:
            log.dropped.append(f"component {d.id}: declared parent {d.parent!r} is neither extracted nor declared")
        if d.note:
            log.notes.append(f"component {d.id}: declared in machine.toml ({d.note})")
        out.append(Component(id=d.id, name=d.name, description=d.description, category=d.category, parent=parent,
                             source="deployment", ref=DECLARED_COMPONENT_REF.format(id=d.id), confidence=1.0))
        log.items.append(ItemCheck(kind="component", id=d.id, agreement=1.0, evidence=(), source="deployment",
                                   confidence=1.0))
    return sorted(out, key=lambda c: c.id)


def build_components(merged: MergedComponents, declared: Sequence[DeclaredComponent], check: Checker,
                     log: Log) -> tuple[list[Component], list[str], list[FlowPath]]:
    """Components (with the declared ones), the boundaries a flow path uses, and flow paths (a boundary nothing
    uses is noted, not kept)."""
    built = declared_components(declared, components(merged.components, check, log, merged.aliases), log)
    boundaries = [b for b in merged.boundaries if b not in {c.id for c in built}]
    flows = flow_paths(merged.flow_paths, {c.id for c in built} | set(boundaries), check, log)
    used = {f.from_component for f in flows} | {f.to_component for f in flows}
    log.notes += [f"boundary {b}: no flow path uses it; not kept" for b in boundaries if b not in used]
    return built, [b for b in boundaries if b in used], flows
