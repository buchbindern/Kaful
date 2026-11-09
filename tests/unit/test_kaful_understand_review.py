"""The onboarding review (M7 exit item, Natalie 2026-11-08), offline on a tiny_pump run: the page lists failed
checks first and items from lowest confidence up, with page links; a filled record becomes machine.toml
declarations that load, and confirmations change nothing."""

import tomllib
from pathlib import Path

import pytest

from kaful.artifacts.document_index import DocumentIndex
from kaful.config import load_machine_config
from kaful.phases.p2_understand.tools.onboarding_review import (
    ReviewEntry,
    ReviewRecord,
    apply_review,
    insert_declaration,
    record_toml,
    review_order,
    write_review,
)
from kaful.pipeline import RunContext
from kaful.runs import load_artifact
from tests.harness.kaful_llm import ScriptedTransport
from tests.harness.kaful_understand import ScriptedUnderstanding
from tests.unit.test_kaful_understand_phase import _artifacts, _run

SEAL = {"id": "seal_cartridge", "name": "Mechanical seal", "description": "Shaft seal", "category": "seal",
        "parent": "booster_pump_p1"}
ACCEL = {"name": "housing_accel_rms", "description": "Retrofitted accelerometer", "role": "observation",
         "dtype": "float", "unit": "mm/s", "sampling": "unknown"}


def _write(ctx: RunContext) -> tuple[Path, Path]:
    model, contract, report = _artifacts(ctx)
    return write_review(ctx.machine_dir, model, contract, report,
                        load_artifact(ctx.machine_dir, model.inputs[0], DocumentIndex))


def _reviewed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunContext, Path, Path, str]:
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(ScriptedUnderstanding()))
    page, record = _write(ctx)
    return ctx, page, record, page.read_text()


def test_the_page_puts_failed_checks_and_low_confidence_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, record, text = _reviewed(tmp_path, monkeypatch)
    assert text.index("Failed checks") < text.index("Review first") < text.index("Component tree")
    assert "observable_by &#x27;cavitation_index&#x27; is not a channel" in text
    assert 'href="../../manual/tp200_manual.pdf#page=' in text and "/Users/" not in text
    entries = ReviewRecord.model_validate(tomllib.loads(record.read_text())).items
    assert entries[0].id == "wear_ring" and all(e.action == "" for e in entries)  # downgraded first


def test_review_order_is_downgraded_then_rising_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(ScriptedUnderstanding()))
    _, _, report = _artifacts(ctx)
    order = review_order(report)
    downgraded = [i for i in order if i.downgraded_from is not None]
    assert order[:len(downgraded)] == downgraded
    rest = order[len(downgraded):]
    assert [i.confidence for i in rest] == sorted(i.confidence for i in rest)


def test_an_existing_record_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _, record, _ = _reviewed(tmp_path, monkeypatch)
    record.write_text(record.read_text().replace('reviewer = ""', 'reviewer = "E. Engineer"'))
    _write(ctx)
    assert 'reviewer = "E. Engineer"' in record.read_text()


def test_apply_writes_sorted_declarations_that_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, _, record_path, _ = _reviewed(tmp_path, monkeypatch)
    machine_dir = ctx.machine_dir
    record = ReviewRecord(machine_id="tiny_pump", run_id="r", reviewer="E. Engineer", reviewed_on="2026-11-09",
                          items=(ReviewEntry(kind="component", id="booster_pump_p1", action="confirm"),
                                 ReviewEntry(kind="component", id="seal_cartridge", action="add", declare=SEAL),
                                 ReviewEntry(kind="component", id="impeller", action="correct",
                                             declare={**SEAL, "id": "impeller", "name": "Impeller"}),
                                 ReviewEntry(kind="channel", id="housing_accel_rms", action="add", declare=ACCEL),
                                 ReviewEntry(kind="fault_mode", id="seal_leak", action="add", note="weeps at 2000 h")))
    record_path.write_text(record_toml(record))
    done = apply_review(machine_dir, record_path)
    assert done == ["channel housing_accel_rms: add -> machine.toml [[instrumentation]]",
                    "component booster_pump_p1: confirm recorded",
                    "component impeller: correct -> machine.toml [[components]]",
                    "component seal_cartridge: add -> machine.toml [[components]]",
                    "fault_mode seal_leak: add recorded (no declaration format for this kind)"]
    config = load_machine_config(machine_dir)
    assert [c.id for c in config.components] == ["impeller", "seal_cartridge"]
    assert config.components[1].note == "onboarding review (E. Engineer, 2026-11-09)"
    assert [c.name for c in config.instrumentation] == ["housing_accel_rms"]
    with pytest.raises(ValueError, match="already declared"):
        apply_review(machine_dir, record_path)


def test_a_correction_needs_a_valid_declaration() -> None:
    with pytest.raises(ValueError, match="needs a declare table"):
        ReviewEntry(kind="component", id="impeller", action="correct")
    with pytest.raises(ValueError, match="declare.id must be"):
        ReviewEntry(kind="component", id="impeller", action="correct", declare=SEAL)
    with pytest.raises(ValueError):
        ReviewEntry(kind="channel", id="x", action="add", declare={"name": "x"})


def test_insert_keeps_a_table_sorted_before_a_later_table() -> None:
    text = '[[components]]\nid = "b"\nname = "B"\n\n[instances]\nspindle = 1\n'
    out = insert_declaration(text, "components", "id", "c", '[[components]]\nid = "c"\n')
    assert out.index('id = "b"') < out.index('id = "c"') < out.index("[instances]")
    out = insert_declaration(out, "components", "id", "a", '[[components]]\nid = "a"\n')
    assert out.index('id = "a"') < out.index('id = "b"')


def test_carriers_are_marked_and_unconfirmed_parts_are_listed_apart(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """M7b (Natalie, 2026-11-09): a carrier that only a datasheet names is not dropped silently; the page lists it
    as a part the machine manuals do not confirm, so an engineer can declare it. A kept carrier is marked."""
    from kaful.phases.p2_understand.tools.onboarding_review import review_html

    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(ScriptedUnderstanding()))
    model, contract, report = _artifacts(ctx)
    gone = ("component pump_bearings (pump bearings): datasheet-only part, not confirmed in the machine manuals "
            "(skf_sheet); a fault-mode carrier; declare it in machine.toml [[components]] if the machine has it")
    items = tuple(i.model_copy(update={"found_by": "fault_modes"}) if i.id == "wear_ring" else i for i in report.items)
    report = report.model_copy(update={"dropped": tuple(sorted((*report.dropped, gone))), "items": items})
    text = review_html(model, contract, report, load_artifact(ctx.machine_dir, model.inputs[0], DocumentIndex))
    failed, apart = text.index("Failed checks"), text.index("Parts only reference documents name (1)")
    assert failed < apart < text.index("Review first") and text.index("pump_bearings") > apart
    assert "component (carrier, from fault modes)" in text
