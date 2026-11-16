"""Verify (M11), checkpoint 2: quarantine and the reduced twin, the phase outcomes, the manifest record, the review
section, determinism, and tiny_pump end to end through verify (offline)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kaful import pipeline
from kaful.artifacts.base import ArtifactBase
from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth
from kaful.artifacts.twin_definition import Connection
from kaful.artifacts.verification import CheckResult, VerificationReport
from kaful.config import DeclaredMaintenance, KafulSettings, SynthesizeSettings
from kaful.paths import resolve_in
from kaful.phases import PhaseName
from kaful.phases.p6_verify import phase as verify_phase
from kaful.phases.p6_verify.d_quarantine import quarantine, reduce
from kaful.phases.p6_verify.tools.review import verify_html
from kaful.runs import dump_json, load_artifact
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_build import _twin
from tests.unit.test_kaful_scope import _contract, _machine
from tests.unit.test_kaful_verify_checks import doubled_life

SETTINGS = KafulSettings()
CLEANED = (DeclaredMaintenance(component="shaft_seal", action="clean", interval=1000, unit="h"),)


def _verify(ps_change: Any = None, allow: bool = False, settings: KafulSettings = SETTINGS,
            twin_change: Any = None) -> verify_phase.Verified:
    twin, ps, failures = _twin(_machine(maintenance=CLEANED))
    assert failures == ()
    twin = twin_change(twin) if twin_change else twin
    return verify_phase.verify({**BASE}, {**BASE}, twin, ps_change(ps) if ps_change else ps, _contract(), settings,
                               allow)


def test_a_twin_that_passes_is_verified_whole() -> None:
    done = _verify()
    assert done.status == "passed" and done.failures == () and done.report.passed
    assert done.twin is not None and done.twin.components == ("bearings", "filter_element_fe", "shaft_seal")
    assert [r.name for r in done.report.runs] == ["documented_maintenance@q0.05", "documented_maintenance@q0.5",
                                                  "documented_maintenance@q0.95", "nominal@q0.5"]


def test_a_failing_component_is_quarantined_and_the_rest_is_checked_again_without_it() -> None:
    done = _verify(doubled_life)
    assert done.status == "quarantined" and done.report.passed
    [q] = done.report.quarantined
    assert (q.component, q.checks) == ("shaft_seal", ("life_consistency",)) and "too fast" in q.reason
    twin = done.twin
    assert twin is not None and twin.components == ("bearings", "filter_element_fe")
    assert {b.component for b in twin.bindings} == {"bearings", "filter_element_fe"}
    assert all("shaft_seal" not in x for x in (*(e.name for e in twin.events), *(r.id for r in twin.maintenance),
                                               *(i.name for i in twin.external_inputs), *twin.flags))
    assert not [e for p in twin.profiles for e in p.events if e.channel == "maintenance_shaft_seal"]
    assert done.report.recheck and {r.status for r in done.report.recheck} <= {"pass", "skipped"}
    assert {r.target for r in done.report.recheck} == {"bearings", "composite", "filter_element_fe"}
    assert done.failures[0].startswith("quarantined shaft_seal: life_consistency: nominal@q0.5: the event fires")
    assert done.notes == ("verified twin: 2 of 3 components (1 quarantined)",)


def test_quarantine_that_leaves_nothing_fails_the_run() -> None:
    def all_doubled(ps: Any) -> Any:
        for component in ("bearings", "filter_element_fe", "shaft_seal"):
            ps = doubled_life(ps, component)
        return ps

    done = _verify(all_doubled)
    assert done.status == "failed" and done.twin is None and not done.report.passed
    assert done.failures[-1] == "quarantine leaves no component"


def test_a_static_failure_stops_the_run_and_is_never_quarantined() -> None:
    def wrong_unit(ps: Any) -> Any:
        return ps.model_copy(update={"parameters": tuple(x.model_copy(update={"unit": "mm / h"})
                                                         if x.param_id == "shaft_seal.k" else x for x in ps.parameters)})

    done = _verify(wrong_unit)
    assert done.status == "failed" and done.twin is None and done.report.quarantined == ()
    assert "shaft_seal units: binding shaft_seal: k unit 'mm / h' is not '1 / h'" in done.failures


def test_allow_failures_keeps_every_component_and_records_it() -> None:
    done = _verify(doubled_life, allow=True)
    assert done.status == "failed" and done.report.allowed_failures and done.report.quarantined == ()
    assert done.twin is not None and "shaft_seal" in done.twin.components and done.report.passed


def test_a_component_fed_by_a_quarantined_one_is_quarantined_with_it() -> None:
    """No connection exists until M13; the rule is in place for when they do."""
    twin, _, _ = _twin()
    wired = twin.model_copy(update={"connections": (Connection(
        id="seal_to_bearings", source_port="shaft_seal.leak", target_port="bearings.contamination",
        source_unit="L/h", target_unit="L/h", match="exact", score=1.0),)})
    failed = CheckResult(check="monotone", target="shaft_seal", status="fail", message="D falls")
    held = quarantine(wired, [failed])
    assert [(q.component, q.checks, q.dependents) for q in held] == [
        ("bearings", ("depends_on_quarantined",), ()), ("shaft_seal", ("monotone",), ("bearings",))]
    assert reduce(wired, held, {**BASE}).components == ("filter_element_fe",)


def test_the_review_section_lists_the_quarantined_component_first_with_how_to_resolve_it() -> None:
    page = verify_html(_verify(doubled_life).report)
    assert "1 quarantined" in page and "<h3>Quarantined (1)</h3>" in page
    assert page.index("shaft_seal") < page.index("Checks per component")
    assert "check the [[param]] for its life or k" in page


@pytest.mark.parametrize("seed", ["0", "1"])
def test_the_report_and_the_verified_twin_do_not_depend_on_hash_order(seed: str) -> None:
    code = ("from tests.unit.test_kaful_verify_phase import _verify; from tests.unit.test_kaful_verify_checks import "
            "doubled_life; from kaful.runs import dump_json; d = _verify(doubled_life); "
            "print(dump_json(d.report) + dump_json(d.twin))")
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONHASHSEED": seed})
    done = _verify(doubled_life)
    assert done.twin is not None
    assert out.stdout.strip() == (dump_json(done.report) + dump_json(done.twin)).strip()


VERIFY_PHASES: tuple[PhaseName, ...] = ("ingest", "understand", "scope", "parameterize", "build", "verify")


def _tiny_pump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, KafulSettings]:
    from kaful.phases.p1_ingest import phase as ingest_phase
    from kaful.phases.p2_understand import phase as understand_phase
    from kaful.phases.p4_parameterize import phase as parameterize_phase
    from kaful.prompts.registry import LockEntry, PromptRegistry
    from tests.harness.kaful_ingest import FakeEmbeddings
    from tests.harness.kaful_llm import ScriptedTransport
    from tests.harness.kaful_understand import (
        MACHINE_ID,
        ScriptedUnderstanding,
        locked_prompts,
        make_tiny_pump,
        understand_settings,
    )
    from tests.unit import test_kaful_scope_tiny_pump as tp

    tp._SCRIPT = ScriptedUnderstanding()
    make_tiny_pump(tmp_path)
    judgement = PromptRegistry().load("parameterize.life_judgement")
    lock = locked_prompts()
    lock.prompts["parameterize.life_judgement"] = LockEntry(version=judgement.version,
                                                            content_sha256=judgement.content_sha256)
    for module in (ingest_phase, understand_phase, parameterize_phase):
        monkeypatch.setattr(module, "read_lock", lambda: lock)
    settings = understand_settings(tmp_path)
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=VERIFY_PHASES, budget_usd=1.0)
    ctx, base = pipeline.start_run(settings, request, transport=ScriptedTransport(tp._with_judgement),
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, VERIFY_PHASES, base)
    return ctx, settings


def _artifact(ctx: Any, phase: PhaseName, kind: str, model: type[ArtifactBase]) -> Any:
    [ref] = [r for r in ctx.manifest.phase(phase).artifacts if r.kind == kind]
    return load_artifact(ctx.machine_dir, ref, model)


def test_tiny_pump_passes_verify_end_to_end_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kaful.artifacts.twin_definition import TwinDefinition

    ctx, _ = _tiny_pump(tmp_path, monkeypatch)
    record = ctx.manifest.phase("verify")
    assert record.status == "passed", record.failures
    report = _artifact(ctx, "verify", "verification_report", VerificationReport)
    twin = _artifact(ctx, "verify", "twin_definition", TwinDefinition)
    assert twin.components == ("impeller", "inlet_cartridge_filter_f1") and twin.run_id == ctx.manifest.run_id
    assert {r.status for r in report.results} <= {"pass", "skipped"}
    assert {r.target for r in report.results if r.check == "life_consistency"} == set(twin.components)
    assert record.notes == ("verified twin: all 2 components",)

    from kaful.artifacts.document_index import DocumentIndex
    from kaful.artifacts.machine_model import MachineModel
    from kaful.artifacts.sensor_contract import SensorContract
    from kaful.artifacts.understand_report import UnderstandReport
    from kaful.phases.p2_understand.tools.onboarding_review import write_review

    model = _artifact(ctx, "understand", "machine_model", MachineModel)
    page, _ = write_review(ctx.machine_dir, model, _artifact(ctx, "understand", "sensor_contract", SensorContract),
                           _artifact(ctx, "understand", "understand_report", UnderstandReport),
                           load_artifact(ctx.machine_dir, model.inputs[0], DocumentIndex), run_id=ctx.manifest.run_id,
                           verification=report)
    assert f"Verification (verify run {ctx.manifest.run_id}): passed" in page.read_text()


def test_a_quarantine_is_recorded_in_the_manifest_and_a_later_run_can_reuse_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The phase plumbing: step c is made to fail the impeller, as a broken twin would."""
    from kaful.phases.p6_verify import c_check_dynamics

    real = c_check_dynamics.dynamic_results

    def impeller_fails(twin: Any, ps: Any, sims: Any, v: Any) -> tuple[CheckResult, ...]:
        out = real(twin, ps, sims, v)
        if len(twin.components) < 2:
            return out
        return tuple(r.model_copy(update={"status": "fail", "message": "broken"})
                     if (r.target, r.check) == ("impeller", "monotone") else r for r in out)

    monkeypatch.setattr(verify_phase, "dynamic_results", impeller_fails)
    ctx, settings = _tiny_pump(tmp_path, monkeypatch)
    record = ctx.manifest.phase("verify")
    assert (record.status, ctx.manifest.status) == ("quarantined", "passed")
    assert record.failures == ("quarantined impeller: monotone: broken",)
    assert record.notes == ("verified twin: 1 of 2 components (1 quarantined)",)
    assert {r.kind for r in record.artifacts} == {"twin_definition", "verification_report"}

    later = pipeline.RunRequest(machine_id=ctx.machine.machine_id, argv=("run",), only=("synthesize",),
                                budget_usd=1.0, base_run=ctx.manifest.run_id)
    again, base = pipeline.start_run(settings, later)
    pipeline.execute(again, ("synthesize",), base)
    assert again.manifest.phase("verify").status == "reused"
    synthesized = again.manifest.phase("synthesize")
    assert synthesized.status == "passed", synthesized.failures
    dataset = _artifact(again, "synthesize", "synthetic_dataset", SyntheticDataset)
    assert dataset.excluded_components == ("impeller",)  # quarantined: not simulated


def test_tiny_pump_runs_end_to_end_through_synthesize_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx, settings = _tiny_pump(tmp_path, monkeypatch)
    small = settings.model_copy(update={"synthesize": SynthesizeSettings(n_units=2)})
    later = pipeline.RunRequest(machine_id=ctx.machine.machine_id, argv=("run",), only=("synthesize",),
                                budget_usd=1.0, base_run=ctx.manifest.run_id)
    again, base = pipeline.start_run(small, later)
    pipeline.execute(again, ("synthesize",), base)
    record = again.manifest.phase("synthesize")
    assert record.status == "passed", record.failures
    assert record.notes[0].startswith("Pipeline sanity check, not validation")
    dataset = _artifact(again, "synthesize", "synthetic_dataset", SyntheticDataset)
    assert [u.unit_id for u in dataset.units] == ["unit_00", "unit_01"] and dataset.excluded_components == ()
    for unit in dataset.units:
        truth = SyntheticTruth.model_validate_json(resolve_in(again.machine_dir, unit.truth.path).read_text())
        assert truth.run_id == again.manifest.run_id and set(truth.true_params)
        assert resolve_in(again.machine_dir, unit.observations.path).is_file()
