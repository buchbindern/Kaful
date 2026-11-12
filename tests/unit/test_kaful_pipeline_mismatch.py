"""A reused artifact whose bytes changed since it was saved stops the run with a recorded failure (M10b).

Regression: runs.load_artifact raised a plain ValueError that pipeline._run_phase did not catch, so the process
crashed and the manifest stayed `running`, with the phase `pending` and no record of why.
"""

from pathlib import Path

import pytest

from kaful import pipeline
from kaful.phases import PhaseName
from kaful.phases.p1_ingest import phase as ingest_phase
from kaful.phases.p2_understand import phase as understand_phase
from kaful.runs import read_manifest
from tests.harness.kaful_ingest import FakeEmbeddings
from tests.harness.kaful_llm import ScriptedTransport
from tests.harness.kaful_understand import (
    MACHINE_ID,
    ScriptedUnderstanding,
    locked_prompts,
    make_tiny_pump,
    understand_settings,
)

UPSTREAM: tuple[PhaseName, ...] = ("ingest", "understand")


def _run(tmp_path: Path, only: tuple[PhaseName, ...], base_run: str | None = None) -> pipeline.RunContext:
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=only, budget_usd=1.0, base_run=base_run)
    ctx, base = pipeline.start_run(understand_settings(tmp_path), request,
                                   transport=ScriptedTransport(ScriptedUnderstanding()),
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, only, base)
    return ctx


def test_a_changed_reused_artifact_fails_the_phase_and_the_run(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    make_tiny_pump(tmp_path)
    for module in (ingest_phase, understand_phase):
        monkeypatch.setattr(module, "read_lock", locked_prompts)
    first = _run(tmp_path, UPSTREAM)
    assert first.manifest.status == "passed"
    [ref] = [r for r in first.manifest.phase("understand").artifacts if r.kind == "machine_model"]
    path = first.machine_dir / ref.path
    path.write_text(path.read_text() + " ")  # still valid JSON, no longer the recorded bytes

    second = _run(tmp_path, ("scope",), base_run=first.manifest.run_id)
    manifest = read_manifest(second.run_dir)
    assert manifest.status == "failed"
    record = manifest.phase("scope")
    assert record.status == "failed"
    assert any(ref.path in f and "sha256" in f for f in record.failures), record.failures


def test_an_index_hash_mismatch_under_strict_fails_the_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """IndexMismatch (--strict) crashed the same way."""
    make_tiny_pump(tmp_path)
    for module in (ingest_phase, understand_phase):
        monkeypatch.setattr(module, "read_lock", locked_prompts)
    first = _run(tmp_path, ("ingest",))
    manifest_file = first.run_dir / "manifest.json"
    manifest_file.write_text(first.manifest.model_copy(update={"index_hash": "0" * 64}).model_dump_json())
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run",), only=("understand",), budget_usd=1.0,
                                  base_run=first.manifest.run_id, strict=True)
    ctx, base = pipeline.start_run(understand_settings(tmp_path), request,
                                   transport=ScriptedTransport(ScriptedUnderstanding()),
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, ("understand",), base)
    manifest = read_manifest(ctx.run_dir)
    assert manifest.status == "failed"
    assert manifest.phase("understand").status == "failed"
    assert any("index hash" in f for f in manifest.phase("understand").failures)
