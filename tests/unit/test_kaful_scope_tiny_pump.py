"""Scope on tiny_pump end to end (offline), the M2 over-decomposition eval case (PLAN M8): the manual says the
skid has two main components; legacy kept 11 full models. Scope must keep exactly 2 (the inlet filter F1, and
the impeller under the booster pump P1: the wear lives in the impeller, so P1 is connection_only as its
parent; Natalie 2026-11-09), with the sensors as channels, not components, and no paid call."""

from pathlib import Path

import pytest

from kaful import pipeline
from kaful.artifacts.scope import Scope
from kaful.phases import PhaseName
from kaful.phases.p1_ingest import phase as ingest_phase
from kaful.phases.p2_understand import phase as understand_phase
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
