"""Understand phase end to end on tiny_pump (offline): ingest, then three prompts x N samples,
merged with agreement, evidence checked against raw page text, artifacts saved."""

import argparse
from pathlib import Path
from typing import Any

import pytest

from kaful import pipeline
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.phases.p1_ingest import phase as ingest_phase
from kaful.phases.p2_understand import phase as understand_phase
from kaful.phases.p2_understand.tools import commands as understand_tools
from kaful.phases.p2_understand.tools.golden_score import GoldenScore
from kaful.prompts.registry import PromptLock
from kaful.runs import load_artifact
from tests.harness.kaful_ingest import FakeEmbeddings
from kaful.llm.transport import Completion
from tests.harness.kaful_llm import DEFAULT_USAGE, ScriptedTransport
from tests.harness.kaful_understand import (
    MACHINE_ID,
    PERIOD_LINE,
    ScriptedUnderstanding,
    locked_prompts,
    make_tiny_pump,
    understand_settings,
)

ACCEL = """
[[instrumentation]]
name = "housing_accel_rms"
description = "Retrofitted accelerometer on the pump bearing housing"
role = "observation"
dtype = "float"
unit = "mm/s"
range = [0.0, 50.0]
sampling = "periodic"
sample_period_s = 1.0
"""


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transport: ScriptedTransport,
         lock: PromptLock | None = None, instrumentation: str = "") -> pipeline.RunContext:
    make_tiny_pump(tmp_path, instrumentation)
    for module in (ingest_phase, understand_phase):
        monkeypatch.setattr(module, "read_lock", lambda: lock or locked_prompts())
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest", "understand"), budget_usd=1.0)
    ctx, base = pipeline.start_run(understand_settings(tmp_path), request, transport=transport,
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, ("ingest", "understand"), base)
    return ctx


def _artifacts(ctx: pipeline.RunContext) -> tuple[MachineModel, SensorContract, UnderstandReport]:
    refs = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
    return (load_artifact(ctx.machine_dir, refs["machine_model"], MachineModel),
            load_artifact(ctx.machine_dir, refs["sensor_contract"], SensorContract),
            load_artifact(ctx.machine_dir, refs["understand_report"], UnderstandReport))


def test_understand_builds_checked_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = ScriptedTransport(ScriptedUnderstanding())
    ctx = _run(tmp_path, monkeypatch, transport)
    record = ctx.manifest.phase("understand")
    assert (record.status, record.failures) == ("passed", ())
    assert len(transport.sent) == 9  # 3 prompts x 3 samples, no retries
    machine, contract, report = _artifacts(ctx)

    assert [(c.id, c.parent) for c in machine.components] == [
        ("booster_pump_p1", None), ("impeller", "booster_pump_p1"), ("inlet_cartridge_filter_f1", None),
        ("pump_bearings", "booster_pump_p1"), ("wear_ring", "booster_pump_p1")]
    assert machine.boundaries == ("discharge_outlet", "supply_inlet") and len(machine.flow_paths) == 3
    wear_ring = next(c for c in machine.components if c.id == "wear_ring")
    assert (wear_ring.source, wear_ring.downgraded_from, wear_ring.evidence[0].verification) == (
        "llm_judgement", "manual", "unsupported")
    bearings = next(i for i in report.items if i.id == "pump_bearings")
    assert bearings.agreement == pytest.approx(1 / 3) and bearings.confidence == pytest.approx(1 / 3, abs=1e-6)

    names = [c.name for c in contract.channels]
    assert names == ["discharge_pressure", "filter_dp", "flow_rate", "inlet_pressure", "motor_current",
                     "motor_temperature", "rated_flow", "vibration_rms"]
    filter_dp = contract.channel("filter_dp")
    assert (filter_dp.unit, filter_dp.range, filter_dp.sampling, filter_dp.sample_period_s) == (
        "bar", (0.0, 2.0), "periodic", 10.0)
    assert filter_dp.sample_period_evidence is not None and filter_dp.sample_period_evidence.quote == PERIOD_LINE
    assert filter_dp.sample_period_evidence.value_role == "spec"
    assert filter_dp.range_evidence is not None and filter_dp.range_evidence.value_role == "spec"
    rated = contract.channel("rated_flow")  # over-extracted by one sample: kept, with low agreement
    assert rated.sampling == "unknown" and rated.confidence == pytest.approx(1 / 3, abs=1e-6)

    assert [f.id for f in machine.fault_modes] == ["impeller_cavitation", "impeller_wear",
                                                   "inlet_cartridge_filter_f1_clogging"]
    cavitation = machine.fault_modes[0]
    assert cavitation.observable_by == ("vibration_rms",)
    assert "fault mode impeller_cavitation: observable_by 'cavitation_index' is not a channel" in report.dropped
    assert all(e.verification == "verbatim" for f in machine.fault_modes for e in f.evidence)
    assert report.context.mode == "whole" and report.n_samples == 3 and report.unit_problems == ()
    assert machine.index_hash == ctx.manifest.index_hash and machine.inputs[0].kind == "document_index"


def test_rerun_is_free_and_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = _run(tmp_path / "a", monkeypatch, ScriptedTransport(ScriptedUnderstanding()))
    transport = ScriptedTransport(ScriptedUnderstanding())
    make = tmp_path / "a"
    monkeypatch.setattr(understand_phase, "read_lock", locked_prompts)
    monkeypatch.setattr(ingest_phase, "read_lock", locked_prompts)
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("ingest", "understand"), budget_usd=1.0)
    ctx, base = pipeline.start_run(understand_settings(make), request, transport=transport,
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, ("ingest", "understand"), base)
    assert transport.sent == [] and ctx.manifest.phase("understand").cost_usd == 0
    a, b = _artifacts(first), _artifacts(ctx)
    for x, y in zip(a, b, strict=True):
        assert x.model_copy(update={"run_id": "r", "inputs": ()}) == y.model_copy(update={"run_id": "r", "inputs": ()})


def test_declared_instrumentation_joins_the_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(ScriptedUnderstanding()), instrumentation=ACCEL)
    _, contract, report = _artifacts(ctx)
    accel = contract.channel("housing_accel_rms")
    assert (accel.source, accel.ref, accel.confidence, accel.sample_period_s) == (
        "deployment", "machine.toml [[instrumentation]] housing_accel_rms", 1.0, 1.0)
    assert next(i for i in report.items if i.id == "housing_accel_rms").source == "deployment"


def test_unlocked_prompts_stop_understand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = locked_prompts()
    lock = lock.model_copy(update={"prompts": {k: v for k, v in lock.prompts.items() if k != "understand.fault_modes"}})
    transport = ScriptedTransport(ScriptedUnderstanding())
    ctx = _run(tmp_path, monkeypatch, transport, lock=lock)
    record = ctx.manifest.phase("understand")
    assert record.status == "failed" and "understand.fault_modes" in record.failures[0] and transport.sent == []


TINY_GOLDEN = """schema_version = 1
machine_id = "tiny_pump"
status = "reviewed"

[[legacy_fields]]
name = "flow_rate"
verdict = "channel"
role = "operating_input"
secondary_role = ""
note = ""

[[candidate_channels]]
name = "rated_flow"
verdict = "not_channel"
role = ""
secondary_role = ""
note = ""

[[components]]
id = "impeller"
verdict = "child"
parent = "booster_pump_p1"
note = ""

[[fault_modes]]
component = "impeller"
what_degrades = "impeller wear"
verdict = "yes"
doc_id = ""
page = ""
page_label = ""
note = ""
"""


def test_understand_score_scores_the_latest_passed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`kaful understand score` (offline): loads the run's artifacts, saves the score, prints a summary."""
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(ScriptedUnderstanding()))
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    (golden_dir / "tiny_pump.toml").write_text(TINY_GOLDEN)
    monkeypatch.setattr(understand_tools, "GOLDEN_DIR", golden_dir)
    args = argparse.Namespace(machine=MACHINE_ID, run=None)
    assert understand_tools.understand_score(args, understand_settings(tmp_path)) == 0
    saved = GoldenScore.model_validate_json(
        (ctx.machine_dir / "outputs" / "understand_scores" / f"{ctx.manifest.run_id}.json").read_text())
    assert saved.run_id == ctx.manifest.run_id and saved.channels.wrong == ("rated_flow -> rated_flow (not_channel)",)
    assert saved.components.parent_accuracy == 1.0 and saved.fault_modes.recall == 1.0
    table = (ctx.machine_dir / "outputs" / "understand_scores" / f"{ctx.manifest.run_id}.md").read_text()
    assert "| channel | rated_flow | a channel (operating_input) | not_channel |" in table


def test_a_truncated_reply_fails_the_phase_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seen live on eversys (Haiku, components hit max_tokens): the run must stop with a failed phase and keep
    the unusable reply for inspection, not crash and leave the manifest 'running'."""
    scripted = ScriptedUnderstanding()

    def responder(request: dict[str, Any]) -> Any:
        text = scripted(request)
        if "list the machine's physical components" in request["system"][0]["text"]:
            return Completion(text=text[:40], model=request["model"], stop_reason="max_tokens", usage=DEFAULT_USAGE)
        return text
    ctx = _run(tmp_path, monkeypatch, ScriptedTransport(responder))
    record = ctx.manifest.phase("understand")
    assert record.status == "failed" and record.failures == ("understand.components: stop_reason 'max_tokens'",)
    assert ctx.manifest.status == "failed"
    saved = ctx.run_dir / "understand" / "unusable_reply.txt"
    assert saved.read_text(encoding="utf-8").startswith("understand.components: stop_reason 'max_tokens'\n")


class _CountingTransport(ScriptedTransport):
    @property
    def can_count_tokens(self) -> bool:
        return True

    def count_tokens(self, request: dict[str, Any]) -> int:
        return 1234


def test_the_whole_index_size_is_counted_when_the_transport_can(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _run(tmp_path, monkeypatch, _CountingTransport(ScriptedUnderstanding()))
    _, _, report = _artifacts(ctx)
    assert report.context.mode == "whole"
    assert "context: the whole index is 1234 tokens by count_tokens" in report.notes
