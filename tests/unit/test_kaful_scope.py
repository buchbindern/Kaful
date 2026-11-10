"""Scope (M8): signals, the life detector, decisions, overrides and gates, on a small two-document machine."""

from typing import Any

import pytest

from kaful.artifacts.base import Evidence
from kaful.artifacts.document_index import DocumentIndex, DocumentRecord, PageLayer, PageText, index_hash_for
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath, MachineModel
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.config import DeclaredLife, DeclaredScope, KafulSettings, MachineConfig
from kaful.phases.p3_scope.c_decide import (
    FLAG_LIFE_MODEL_ONLY,
    FLAG_MACHINE_WIDE,
    FLAG_NAMED_ONLY,
    FLAG_NOTHING_TO_CALIBRATE,
    FLAG_THRESHOLD_EVENT,
)
from kaful.doctext import table_rows
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


def _channel(name: str, description: str) -> SensorChannel:
    return SensorChannel(name=name, description=description, role="observation", dtype="float", unit="L/min",
                         sampling="unknown", source="manual", ref="manual p1", confidence=0.9,
                         evidence=(Evidence(doc_id="manual", page=1, quote="leak", verification="verbatim"),))


def _contract(*extra: SensorChannel) -> SensorContract:
    return SensorContract(**BASE, channels=tuple(sorted((_channel("leak_flow", "Leakage flow"), *extra),
                                                        key=lambda c: c.name)))


def _machine(**extra: Any) -> MachineConfig:
    return MachineConfig(machine_id=BASE["machine_id"], display_name="Test pump", time_unit="h", **extra)


def _scope(machine: MachineConfig | None = None, contract: SensorContract | None = None,
           model: MachineModel | None = None) -> tuple[Scope, tuple[str, ...]]:
    return build_scope({**BASE}, _index(), model or _model(), contract or _contract(), machine or _machine(), SETTINGS)


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
    assert sorted((e.basis, e.evidence.quote) for e in filt if e.evidence) == [
        ("machine_wide", "Have the hose inspected"),  # on a page its fault mode cites: a lower bound
        ("stated", "Yearly 4000 Replace filter element (FE)")]  # the "Check ... (FE)" row has no life word
    assert [(s.evidence.page, s.evidence.quote, s.linked) for s in scope.machine_life] == [
        (2, "Have the hose inspected", ("filter_element_fe",)),  # names no component; its page has the filter's mode
        (3, "The service of the machine is due every 12 months or every 50'000 products.", ())]


def test_a_life_model_without_a_channel_is_flagged_with_the_inputs_m9_needs() -> None:
    by_id = {d.component: d for d in _scope()[0].decisions}
    assert FLAG_LIFE_MODEL_ONLY.format(phrase="L10") in by_id["bearings"].flags
    assert not any("life model available" in f for f in by_id["shaft_seal"].flags)  # a stated life


def test_a_channel_that_names_the_component_as_the_thing_measured_observes_it() -> None:
    """Approved 2026-11-09: kept apart from fault-mode links, at lower confidence. A name followed by another
    part word only qualifies that part (eversys ticks_per_second: 'the coffee boiler flowmeter')."""
    contract = _contract(_channel("motor_temperature", "Winding temperature, shown on the display."),
                         _channel("flow_ticks", "Flow of the motor flowmeter in ticks."),
                         _channel("outlet_reading", "Pressure at the outlet of the pump, in bar."),
                         _channel("valve_position", "Position of the valve."))
    by_id = {d.component: d for d in _scope(contract=contract)[0].decisions}
    motor = by_id["motor"]
    assert (motor.decision, [(c.channel, c.named_in) for c in motor.signals.named_by]) == (
        "full_model", [("motor_temperature", "name")])
    assert FLAG_NAMED_ONLY in motor.flags and FLAG_NOTHING_TO_CALIBRATE not in motor.flags
    assert motor.signals.named_by[0].confidence == pytest.approx(0.9 * SETTINGS.scope.named_channel_factor)
    assert by_id["pump"].signals.named_by == ()  # no fault mode on the pump: a named channel decides nothing
    assert by_id["valve"].decision == "excluded"
    seal = by_id["shaft_seal"]
    assert seal.signals.named_by == () and seal.signals.observed_by[0].confidence == pytest.approx(0.9)


def test_a_name_is_the_thing_measured_only_before_a_quantity_word_punctuation_or_nothing() -> None:
    from kaful.doctext import measures, stem

    quantities = {stem(w) for w in SETTINGS.scope.channel_quantity_words}
    boiler = [["coffee", "boiler"]]
    assert measures("Coffee boiler temperature shown on the display.", boiler, quantities)
    assert measures("Pressure dewpoint temperature of the coffee boiler, reported", boiler, quantities)
    assert measures("coffee boiler", boiler, quantities)
    assert not measures("Flow rate of the coffee boiler flowmeter in ticks per second.", boiler, quantities)


def test_a_threshold_event_observes_and_a_machine_wide_statement_on_a_cited_page_is_a_life() -> None:
    """Approved 2026-11-09: a fault mode's warning or limit event counts as O (its own kind), and a machine-wide
    statement on a page the component's fault modes cite counts as L (a lower bound); both are flagged."""
    event = _channel("warning", "Warning messages").model_copy(update={"role": "event", "dtype": "category", "unit": None, "categories": ("W-1",), "sampling": "event_driven"})
    model = _model()
    modes = (*model.fault_modes,
             _mode("motor_insulation", "motor", "manual", 4, "The motor may overheat.", observable=("warning",)),
             _mode("valve_wear", "valve", "manual", 3, "The service of the machine is due every 12 months"))
    by_id = {d.component: d for d in _scope(contract=_contract(event),
                                             model=model.model_copy(update={"fault_modes": modes}))[0].decisions}
    motor, valve = by_id["motor"], by_id["valve"]
    assert motor.decision == "full_model" and [c.channel for c in motor.signals.threshold_events] == ["warning"]
    assert FLAG_THRESHOLD_EVENT.format(channels="warning") in motor.flags
    assert valve.decision == "full_model" and [e.basis for e in valve.signals.life] == ["machine_wide"]
    assert FLAG_MACHINE_WIDE.format(refs="manual p3") in valve.flags


def test_a_statement_that_names_a_different_part_does_not_link_through_a_cited_page() -> None:
    """Approved 2026-11-09: air_compressor restrictors got a life from the p29 row "Have all flexibles/flexible
    joints inspected". Only a statement that names the machine, or names no part at all, links through a cited
    page; one that prints a word of another part's name is about that part, and gives no one a life."""
    model = _model()
    hose = _component("discharge_hose", "Discharge hose")
    scope, _ = _scope(model=model.model_copy(update={"components": tuple(sorted(
        (*model.components, hose), key=lambda c: c.id))}))
    filt = {d.component: d for d in scope.decisions}["filter_element_fe"].signals.life
    assert [e.basis for e in filt] == ["stated"]  # no machine_wide life from the hose row
    assert [(s.evidence.page, s.linked) for s in scope.machine_life] == [
        (2, ("discharge_hose",)),  # linked to the part it loosely names, for the review
        (3, ())]  # names the machine: still links through a cited page (none cites p3 here)


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


GOLDEN = """machine_id = "%s"
[[components]]
id = "ball_bearings"
aliases = ["bearings"]
verdict = "child"
[[components]]
id = "shaft_seal"
verdict = "top_level"
[[components]]
id = "motor"
verdict = "top_level"
[[components]]
id = "missing_part"
verdict = "top_level"
[[fault_modes]]
component = "bearings"
what_degrades = "raceway flaking"
verdict = "yes"
[[fault_modes]]
component = "shaft_seal"
what_degrades = "sudden seal burst"
verdict = "no"
[[fault_modes]]
component = "motor"
what_degrades = "winding ageing"
documented = false
verdict = "yes"
[[fault_modes]]
component = "missing_part"
what_degrades = "wear"
verdict = "yes"
""" % BASE["machine_id"]


def test_the_scorer_counts_recall_given_understand_and_strict_and_lenient_precision(tmp_path: Any) -> None:
    from kaful.phases.p3_scope.tools.score import score

    golden = tmp_path / "golden.toml"
    golden.write_text(GOLDEN)
    result = score(golden, _scope()[0])
    assert (result.recall, result.found, result.not_extracted) == (1.0, ("ball_bearings <- bearings",), ("missing_part",))
    assert result.precision_strict.wrong == ("shaft_seal -> shaft_seal",) and result.unflagged == ()
    assert result.unmatched_full_models == ("filter_element_fe",)  # unreviewed: counts against precision
    assert result.precision_strict.value == result.precision_lenient.value == pytest.approx(1 / 3)
    motor = _machine(scope=(DeclaredScope(component="motor", decision="full_model", reason="r"),))
    lenient = score(golden, _scope(motor)[0]).precision_lenient  # a documented = false row counts when lenient
    assert lenient.correct == ("bearings -> ball_bearings", "motor -> motor")
