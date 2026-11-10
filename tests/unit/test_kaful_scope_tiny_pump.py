"""Scope on tiny_pump end to end (offline), the M2 over-decomposition eval case (PLAN M8): the manual says the
skid has two main components; legacy kept 11 full models. Scope must keep exactly 2 (the inlet filter F1, and
the impeller under the booster pump P1: the wear lives in the impeller, so P1 is connection_only as its
parent; Natalie 2026-11-09), with the sensors as channels, not components, and no paid call."""

import tomllib
from pathlib import Path
from typing import Any

import pytest

from kaful import pipeline
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.phases import PhaseName
from kaful.phases.p1_ingest import phase as ingest_phase
from kaful.phases.p2_understand import phase as understand_phase
from kaful.phases.p2_understand.tools.onboarding_review import (
    ReviewEntry,
    ReviewRecord,
    apply_review,
    record_toml,
    write_review,
)
from kaful.runs import RunContext, load_artifact
from tests.harness.kaful_ingest import FakeEmbeddings
from tests.harness.kaful_llm import ScriptedTransport
from tests.harness.kaful_understand import (
    CHANNELS,
    MACHINE_ID,
    ScriptedUnderstanding,
    locked_prompts,
    make_tiny_pump,
    understand_settings,
)

PHASES: tuple[PhaseName, ...] = ("ingest", "understand", "scope")


def _scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[RunContext, Scope, int]:
    make_tiny_pump(tmp_path)
    for module in (ingest_phase, understand_phase):
        monkeypatch.setattr(module, "read_lock", locked_prompts)
    transport = ScriptedTransport(ScriptedUnderstanding())
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=PHASES, budget_usd=1.0)
    ctx, base = pipeline.start_run(understand_settings(tmp_path), request, transport=transport,
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, PHASES, base)
    [ref] = ctx.manifest.phase("scope").artifacts
    return ctx, load_artifact(ctx.machine_dir, ref, Scope), len(transport.sent)


def test_tiny_pump_keeps_two_full_models_and_no_sensor_is_a_component(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, scope, sent = _scope(tmp_path, monkeypatch)
    assert ctx.manifest.phase("scope").status == "passed"
    assert sent == 9  # understand's 3 prompts x 3 samples; scope sends nothing
    decisions = {d.component: (d.decision, d.decided_by) for d in scope.decisions}
    assert decisions == {
        "booster_pump_p1": ("connection_only", "rule"),  # parent of the impeller
        "impeller": ("full_model", "rule"),  # wear observed by discharge_pressure and motor_current
        "inlet_cartridge_filter_f1": ("full_model", "rule"),  # clogging observed by filter_dp
        "pump_bearings": ("excluded", "rule"),
        "wear_ring": ("excluded", "rule"),  # its only evidence is a paraphrase (downgraded)
    }
    assert scope.in_scope() == ("impeller", "inlet_cartridge_filter_f1")
    assert not {name for name, *_ in CHANNELS} & set(decisions)


def test_downstream_per_component_work_sees_only_the_full_models(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    """PLAN M8 accept: later phases iterate over Scope.full_model_components only (eversys $0.16 and CNC $0.07
    of legacy per-component spend on triaged-out components -> $0)."""
    ctx, scope, _ = _scope(tmp_path, monkeypatch)
    [ref] = [r for r in ctx.manifest.phase("understand").artifacts if r.kind == "machine_model"]
    model = load_artifact(ctx.machine_dir, ref, MachineModel)
    assert [c.id for c in scope.full_model_components(model)] == ["impeller", "inlet_cartridge_filter_f1"]
    smaller = model.model_copy(update={"components": tuple(c for c in model.components if c.id != "impeller")})
    with pytest.raises(ValueError, match="impeller"):
        scope.full_model_components(smaller)


def test_a_reviewed_scope_correction_becomes_an_override_on_the_next_run(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit criterion 8: review, then apply, then machine.toml [[scope]], then a rerun shows `override`, and the
    manifest records it."""
    ctx, scope, _ = _scope(tmp_path, monkeypatch)
    refs = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
    model = load_artifact(ctx.machine_dir, refs["machine_model"], MachineModel)
    page, record_path = write_review(
        ctx.machine_dir, model, load_artifact(ctx.machine_dir, refs["sensor_contract"], SensorContract),
        load_artifact(ctx.machine_dir, refs["understand_report"], UnderstandReport),
        load_artifact(ctx.machine_dir, model.inputs[0], DocumentIndex), scope, ctx.manifest.run_id)
    text = page.read_text()
    assert "Scope decisions" in text and "inlet_cartridge_filter_f1" in text
    record = ReviewRecord.model_validate(tomllib.loads(record_path.read_text()))
    assert [e.id for e in record.items if e.kind == "scope"] == [
        "impeller", "inlet_cartridge_filter_f1", "booster_pump_p1"]  # none flagged: full models, then connection only
    entries = [e if (e.kind, e.id) != ("scope", "booster_pump_p1") else ReviewEntry(
        kind="scope", id="booster_pump_p1", action="correct", declare={
            "component": "booster_pump_p1", "decision": "full_model", "reason": "engineer: model the pump casing"})
        for e in record.items]
    record_path.write_text(record_toml(record.model_copy(update={"items": tuple(entries), "reviewer": "E. Engineer",
                                                                 "reviewed_on": "2026-11-09"})))
    assert apply_review(ctx.machine_dir, record_path) == ["scope booster_pump_p1: correct -> machine.toml [[scope]]"]

    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("scope",), budget_usd=1.0,
                                  base_run=ctx.manifest.run_id)
    again, base = pipeline.start_run(understand_settings(tmp_path), request, transport=ScriptedTransport(
        ScriptedUnderstanding()), embedding_transport=FakeEmbeddings())
    pipeline.execute(again, ("scope",), base)
    [ref] = again.manifest.phase("scope").artifacts
    pump = {d.component: d for d in load_artifact(again.machine_dir, ref, Scope).decisions}["booster_pump_p1"]
    assert (pump.decision, pump.decided_by, pump.reason) == ("full_model", "override", "engineer: model the pump casing")
    assert [(o.kind, o.target, o.value) for o in again.manifest.overrides] == [
        ("scope", "booster_pump_p1", "full_model"),
        ("base_run", ctx.manifest.run_id, "inputs differ")]  # machine.toml changed since the base run


JUDGED = '{"life": {"unit": "hour", "low": 5000, "high": 40000}, "basis": "centrifugal pump impellers in clean water"}'


def _with_judgement(request: dict[str, Any]) -> str:
    """ScriptedUnderstanding plus parameterize.life_judgement (the impeller has only an inspection interval)."""
    if "You estimate how long" in request["system"][0]["text"]:
        return JUDGED
    return _SCRIPT(request)


_SCRIPT = ScriptedUnderstanding()


def test_a_reviewed_param_becomes_a_declaration_and_the_next_parameterize_run_uses_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M9 checkpoint 4: review (judged lives first) -> apply -> machine.toml [[param]] -> parameterize uses it
    (source deployment), and the manifest records the override."""
    from kaful.artifacts.param_set import ParamSet
    from kaful.phases.p4_parameterize import phase as parameterize_phase
    from kaful.phases.p4_parameterize.tools.report import build_report
    from kaful.prompts.registry import LockEntry, PromptRegistry

    global _SCRIPT
    _SCRIPT = ScriptedUnderstanding()
    make_tiny_pump(tmp_path)
    judgement = PromptRegistry().load("parameterize.life_judgement")
    lock = locked_prompts()
    lock.prompts["parameterize.life_judgement"] = LockEntry(version=judgement.version,
                                                            content_sha256=judgement.content_sha256)
    for module in (ingest_phase, understand_phase, parameterize_phase):
        monkeypatch.setattr(module, "read_lock", lambda: lock)
    phases: tuple[PhaseName, ...] = ("ingest", "understand", "scope", "parameterize")
    settings = understand_settings(tmp_path)
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=phases, budget_usd=1.0)
    ctx, base = pipeline.start_run(settings, request, transport=ScriptedTransport(_with_judgement),
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, phases, base)
    assert ctx.manifest.phase("parameterize").status == "passed", ctx.manifest.phase("parameterize").failures
    refs = {r.kind: r for p in ("understand", "scope", "parameterize") for r in ctx.manifest.phase(p).artifacts}
    param_set = load_artifact(ctx.machine_dir, refs["param_set"], ParamSet)
    assert param_set.get("impeller.life_nominal").source == "llm_judgement"
    # "Replace the cartridge every 2000 operating hours" does not name "inlet cartridge filter (F1)" as printed
    # (scope's naming rule), so the filter's life is judged too: a naming gap, recorded in M9.md.
    assert param_set.get("inlet_cartridge_filter_f1.life_nominal").source == "llm_judgement"
    model = load_artifact(ctx.machine_dir, refs["machine_model"], MachineModel)
    params = (param_set, build_report(param_set, settings.parameterize))
    page, record_path = write_review(
        ctx.machine_dir, model, load_artifact(ctx.machine_dir, refs["sensor_contract"], SensorContract),
        load_artifact(ctx.machine_dir, refs["understand_report"], UnderstandReport),
        load_artifact(ctx.machine_dir, model.inputs[0], DocumentIndex),
        load_artifact(ctx.machine_dir, refs["scope"], Scope), ctx.manifest.run_id, params)
    assert "Parameters (parameterize run" in page.read_text()
    record = ReviewRecord.model_validate(tomllib.loads(record_path.read_text()))
    lives = [e.id for e in record.items if e.kind == "param"][:2]
    assert lives == ["impeller.life_nominal", "inlet_cartridge_filter_f1.life_nominal"]  # judged first
    entries = [e if (e.kind, e.id) != ("param", "impeller.life_nominal") else ReviewEntry(
        kind="param", id="impeller.life_nominal", action="correct", declare={
            "param": "impeller.life_nominal", "unit": "hour", "low": 16000, "high": 64000,
            "reason": "engineer: impellers in this duty last about 4 years of continuous running"})
        for e in record.items]
    record_path.write_text(record_toml(record.model_copy(update={"items": tuple(entries), "reviewer": "E. Engineer",
                                                                 "reviewed_on": "2026-11-10"})))
    assert "param impeller.life_nominal: correct -> machine.toml [[param]]" in apply_review(ctx.machine_dir, record_path)

    again_request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("parameterize",),
                                        budget_usd=1.0, base_run=ctx.manifest.run_id)
    again, base = pipeline.start_run(settings, again_request, transport=ScriptedTransport(_with_judgement),
                                     embedding_transport=FakeEmbeddings())
    pipeline.execute(again, ("parameterize",), base)
    [ref] = again.manifest.phase("parameterize").artifacts
    life = load_artifact(again.machine_dir, ref, ParamSet).get("impeller.life_nominal")
    assert (life.source, life.prior.quantile(0.5)) == ("deployment", pytest.approx(32000))
    assert ("param", "impeller.life_nominal") in [(o.kind, o.target) for o in again.manifest.overrides]
