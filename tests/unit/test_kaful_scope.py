"""Scope (M8): signals, the life detector, decisions, overrides and gates, on a small two-document machine."""

from typing import Any

import pytest

from kaful.artifacts.base import Evidence
from kaful.artifacts.document_index import DocumentIndex, DocumentRecord, PageLayer, PageText, index_hash_for
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath, MachineModel
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.config import DeclaredLife, DeclaredScope, KafulSettings, MachineConfig
from kaful.phases.p3_scope.c_decide import FLAG_NOTHING_TO_CALIBRATE
from kaful.phases.p3_scope.helpers.text import table_rows
from kaful.phases.p3_scope.phase import build_scope
from tests.harness.kaful_factories import BASE, FINGERPRINT, SHA, SIGNALS

SETTINGS = KafulSettings()

SCHEDULE = """Table (text_layer)
| Period | Running hours | Operation |
| --- | --- | --- |
| Daily |  | Check oil level |
| “ | — | Check the filter element (FE) |
| Yearly | 4000 | Replace filter element (FE) |
| “ | — | Have the hose inspected |"""

PAGES = {
    ("manual", 1): "The shaft seal must be replaced every 2000 hours of operation. Keep the area clean.",
    ("manual", 2): "Maintenance schedule",
    ("manual", 3): "The service of the machine is due every 12 months or every 50'000 products.",
    ("manual", 4): "The motor drives the pump. The motor may overheat.",
    ("catalogue", 1): "Angular contact ball bearings\nThe basic rating life (L10) of a bearing\nflaking of raceways",
    ("catalogue", 2): "Rating life of motors in general",
}


def _index() -> DocumentIndex:
    pages = tuple(PageText(doc_id=d, page=p, label=str(p), label_source="printed", text=t, page_class="text",
                           signals=SIGNALS,
                           layers=(PageLayer(kind="llm", text=SCHEDULE, producer="test"),) if (d, p) == ("manual", 2)
                           else ()) for (d, p), t in sorted(PAGES.items()))
    docs = (DocumentRecord(doc_id="catalogue", path="manual/cat.pdf", kind="catalogue", sha256=SHA, n_pages=2),
            DocumentRecord(doc_id="manual", path="manual/pump.pdf", kind="manual", sha256=SHA, n_pages=4))
    return DocumentIndex(**BASE, index_hash=index_hash_for((), FINGERPRINT), documents=docs, pages=pages, chunks=(),
                         fingerprint=FINGERPRINT)


def _quoted(doc: str, page: int, quote: str) -> dict[str, Any]:
    return {"source": "manual", "ref": f"{doc} p{page}", "confidence": 0.9,
            "evidence": (Evidence(doc_id=doc, page=page, quote=quote, verification="verbatim"),)}


def _component(cid: str, name: str, parent: str | None = None) -> Component:
    return Component(id=cid, name=name, description="d", category="part", parent=parent,
                     **_quoted("manual", 4, "The motor drives the pump."))


def _mode(mid: str, component: str, doc: str, page: int, quote: str, observable: tuple[str, ...] = ()) -> FaultMode:
    return FaultMode(id=mid, component=component, name=mid, mechanism="wear", effect="e", observable_by=observable,
                     **_quoted(doc, page, quote))


def _model() -> MachineModel:
    return MachineModel(**BASE, n_extractions=3, components=(
        _component("bearings", "Angular contact ball bearings", parent="motor"),
        _component("filter_element_fe", "Filter element (FE)"),
        _component("motor", "Motor"),
        _component("pump", "Pump"),
        _component("shaft_seal", "Shaft seal"),
        _component("valve", "Valve"),
    ), flow_paths=(FlowPath(id="pump_to_seal", from_component="pump", to_component="shaft_seal", medium="fluid",
                            source="llm_judgement", ref="j", confidence=0.3),),
        fault_modes=(
        _mode("bearings_flaking", "bearings", "catalogue", 1, "flaking of raceways"),
        _mode("filter_clogging", "filter_element_fe", "manual", 2, "Maintenance schedule"),
        _mode("motor_overheating", "motor", "manual", 4, "The motor may overheat."),
        _mode("seal_wear", "shaft_seal", "manual", 1, "The shaft seal must be replaced", observable=("leak_flow",)),
    ))


def _contract() -> SensorContract:
    return SensorContract(**BASE, channels=(SensorChannel(
        name="leak_flow", description="Leakage flow", role="observation", dtype="float", unit="L/min",
        sampling="unknown", source="manual", ref="manual p1", confidence=0.9,
        evidence=(Evidence(doc_id="manual", page=1, quote="leak", verification="verbatim"),)),))


def _machine(**extra: Any) -> MachineConfig:
    return MachineConfig(machine_id=BASE["machine_id"], display_name="Test pump", time_unit="h", **extra)


def _scope(machine: MachineConfig | None = None) -> tuple[Scope, tuple[str, ...]]:
    return build_scope({**BASE}, _index(), _model(), _contract(), machine or _machine(), SETTINGS)


def _decisions(scope: Scope) -> dict[str, tuple[str, str]]:
    return {d.component: (d.decision, d.decided_by) for d in scope.decisions}


def test_full_model_needs_a_fault_mode_and_an_observing_channel_or_a_life() -> None:
    scope, failures = _scope()
    assert failures == ()
    assert _decisions(scope) == {
        "bearings": ("full_model", "rule"),  # a catalogue life model on a page its fault mode cites
        "filter_element_fe": ("full_model", "rule"),  # a schedule row names it by its designation
        "motor": ("connection_only", "rule"),  # G alone (and the parent of the bearings)
        "pump": ("connection_only", "rule"),  # on a flow path with the seal
        "shaft_seal": ("full_model", "rule"),  # observed by leak_flow, and a stated interval
        "valve": ("excluded", "rule"),
    }
    by_id = {d.component: d for d in scope.decisions}
    assert FLAG_NOTHING_TO_CALIBRATE in by_id["motor"].flags
    assert [c.channel for c in by_id["shaft_seal"].signals.observed_by] == ["leak_flow"]
    assert [e.basis for e in by_id["shaft_seal"].signals.life] == ["stated"]


def test_a_life_model_counts_only_from_a_reference_document_the_components_fault_modes_cite() -> None:
    """The catalogue's "Rating life of motors" page must not give the motor a life: its fault modes cite the manual."""
    by_id = {d.component: d for d in _scope()[0].decisions}
    [life] = by_id["bearings"].signals.life
    assert (life.basis, life.ref, life.evidence.quote if life.evidence else "") == (
        "life_model", "catalogue p1", "The basic rating life (L10) of a bearing")
    assert by_id["motor"].signals.life == ()


def test_schedule_rows_and_machine_wide_statements() -> None:
    scope, _ = _scope()
    filt = {d.component: d for d in scope.decisions}["filter_element_fe"].signals.life
    assert sorted(e.evidence.quote for e in filt if e.evidence) == [
        "Yearly 4000 Replace filter element (FE)"]  # the "Check ... (FE)" row has no life word
    assert [(s.evidence.page, s.evidence.quote, s.linked) for s in scope.machine_life] == [
        (2, "Have the hose inspected", ("filter_element_fe",)),  # names no component; its page has the filter's mode
        (3, "The service of the machine is due every 12 months or every 50'000 products.", ())]


def test_overrides_and_declared_lives() -> None:
    machine = _machine(scope=(DeclaredScope(component="motor", decision="full_model", reason="engineer: it runs hot"),),
                       life=(DeclaredLife(component="valve", kind="service_interval", value=500, unit="h"),))
    scope, failures = _scope(machine)
    by_id = {d.component: d for d in scope.decisions}
    assert failures == ()
    assert (by_id["motor"].decision, by_id["motor"].decided_by, by_id["motor"].reason) == (
        "full_model", "override", "engineer: it runs hot")
    assert any(f.startswith("rule said connection_only") for f in by_id["motor"].flags)
    assert [e.basis for e in by_id["valve"].signals.life] == ["declared"]
    assert by_id["valve"].decision == "excluded"  # a life without a fault mode is not a full model


def test_an_override_or_life_for_an_unknown_component_fails_the_gate() -> None:
    machine = _machine(scope=(DeclaredScope(component="ghost", decision="excluded", reason="r"),),
                       life=(DeclaredLife(component="ghost", kind="rated_life", value=1, unit="h"),))
    assert _scope(machine)[1] == ("machine.toml [[scope]] names unknown component ghost",
                                  "machine.toml [[life]] names unknown component ghost")


def test_a_ditto_mark_repeats_the_period_of_the_row_above() -> None:
    rows = table_rows(SCHEDULE)
    assert [(r.cells[0], r.inherited) for r in rows] == [
        ("Daily", ""), ("“", "Daily"), ("Yearly", ""), ("“", "Yearly")]


def test_a_long_run_on_sentence_is_not_a_statement() -> None:
    """CNC p672: a parameter table read without full stops became one 'sentence' naming the SPM."""
    from kaful.phases.p3_scope.b_find_life_statements import _Rules

    rules = _Rules(SETTINGS.scope)
    assert rules.sentence("Replace the seal every 2000 h.")
    assert rules.sentence("The seal is replaced after 2000 h.")  # inflected life word
    assert not rules.sentence("The seal lasts 2000 h.")  # no life word
    assert not rules.sentence("Replace " + "x " * SETTINGS.scope.statement_max_words + "every 2000 h.")


@pytest.mark.parametrize("seed", ["0", "1"])
def test_scope_does_not_depend_on_hash_order(seed: str) -> None:
    import os
    import subprocess
    import sys

    code = ("from tests.unit.test_kaful_scope import _scope; from kaful.runs import dump_json; "
            "print(dump_json(_scope()[0]))")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": seed})
    from kaful.runs import dump_json
    assert out.stdout.strip() == dump_json(_scope()[0]).strip()
