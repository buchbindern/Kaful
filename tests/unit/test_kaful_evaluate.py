"""The evaluate phase (M14), checkpoint 1 sections: lineage, grounding, prompt quality, twin quality and cost, built
from saved artifacts and ledgers only; a lineage that stopped early lists every section it never reached; the M1
baseline data matches its document; output identical under two hash seeds."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from kaful.artifacts.base import ArtifactBase, ArtifactRef, Evidence, Provenanced
from kaful.artifacts.evaluation import EvaluationReport
from kaful.artifacts.manifest import PhaseRecord, RunManifest
from kaful.artifacts.verification import CheckResult, VerificationReport
from kaful.config import REPO_ROOT, EvaluateSettings, KafulSettings
from kaful.llm.ledger import LedgerRow
from kaful.phases import PHASE_ORDER
from kaful.phases.p9_evaluate.a_resolve_lineage import resolve
from kaful.phases.p9_evaluate.b_grounding import count
from kaful.phases.p9_evaluate.c_prompt_quality import uses
from kaful.phases.p9_evaluate.d_twin_quality import rates
from kaful.phases.p9_evaluate.e_cost import m1
from kaful.phases.p9_evaluate.helpers.render import report_md
from kaful.phases.p9_evaluate.phase import evaluate
from kaful.prompts.registry import LockEntry, PromptLock, RunScores
from kaful.runs import dump_json, sha256_bytes, write_text_atomic
from tests.harness import kaful_factories as f

PRODUCER = f.RUN_ID
EVALUATED = "20261103T120000Z-def456"
SETTINGS = KafulSettings()


def _save(machine_dir: Path, phase: str, artifact: ArtifactBase) -> ArtifactRef:
    kind = str(getattr(artifact, "kind"))
    rel = f"outputs/runs/{PRODUCER}/{phase}/{kind}.json"
    text = dump_json(artifact)
    write_text_atomic(machine_dir / rel, text)
    return ArtifactRef(kind=kind, run_id=PRODUCER, path=rel, sha256=sha256_bytes(text.encode()))


def _row(phase: str, **kw: object) -> str:
    base = {"schema_version": 2, "ts": "2026-11-02T12:00:00+00:00", "run_id": PRODUCER, "event": "llm_call",
            "phase": phase, "step": "s", "status": "ok"}
    return LedgerRow.model_validate({**base, **kw}).model_dump_json(exclude_none=True)


def _ledger(machine_dir: Path) -> None:
    rows = [
        _row("ingest", event="pdf_extract", cost_usd=0.2),
        _row("understand", prompt_id="understand.components", prompt_version=5, response_cache="hit",
             model="claude-haiku-5-5", cost_usd=0.0, unbilled_cost_usd=0.01),
        _row("understand", prompt_id="understand.components", prompt_version=5, response_cache="hit",
             model="claude-haiku-5-5", cost_usd=0.0),
        _row("parameterize", prompt_id="parameterize.life_judgement", prompt_version=1, response_cache="miss",
             model="claude-haiku-5-5", cost_usd=0.003),
    ]
    write_text_atomic(machine_dir / f"outputs/runs/{PRODUCER}/ledger.jsonl", "\n".join(rows) + "\n")
    (machine_dir / f"outputs/runs/{EVALUATED}").mkdir(parents=True, exist_ok=True)


def _manifest(machine_dir: Path, stop_at: str | None = None) -> RunManifest:
    """A lineage from the factories; with `stop_at`, that phase failed and nothing after it ran (CDV-like)."""
    saved = {"ingest": (f.document_index(),), "understand": (f.machine_model(), f.sensor_contract()), "scope": (f.scope(),),
             "parameterize": (f.param_set(),), "build": (f.twin_definition(),),
             "verify": (f.verification_report(),)}
    _ledger(machine_dir)
    records, stopped = [], False
    for p in PHASE_ORDER:
        status = "reused"
        if stopped or p in ("synthesize", "estimate", "evaluate"):
            status = "pending" if stopped else "skipped"
        elif p == stop_at:
            status, stopped = "failed", True
        refs = tuple(sorted((_save(machine_dir, p, a) for a in saved.get(p, ())), key=lambda r: (r.kind, r.path)))
        if status in ("pending", "skipped"):
            refs = ()
        records.append(PhaseRecord(phase=p, status=status, artifacts=refs,
                                   failures=("unit gate: 2 units do not parse",) if p == stop_at else ()))
    return f.run_manifest().model_copy(update={"run_id": EVALUATED, "phases": tuple(records)})


def _report(tmp_path: Path, stop_at: str | None = None) -> EvaluationReport:
    return evaluate(_manifest(tmp_path, stop_at), EVALUATED, EVALUATED, tmp_path, SETTINGS, root=tmp_path)


def test_lineage_names_the_producing_run_and_why_a_phase_was_not_reached(tmp_path: Path) -> None:
    lineage = resolve(_manifest(tmp_path, stop_at="understand"))
    assert lineage.get("understand").produced_by == PRODUCER and not lineage.reached("understand")
    assert lineage.get("scope").produced_by is None
    assert lineage.why_not("verify") == f"understand failed in run {EVALUATED}: unit gate: 2 units do not parse"
    assert "evaluate" not in {p.phase for p in lineage.phases}


def test_grounding_counts_every_quote_class_and_each_downgrade_by_code() -> None:
    ev = (Evidence(doc_id="m", page=3, quote="q", verification="verbatim"),
          Evidence(doc_id="m", page=4, quote="q", verification="in_layer"))
    item = Provenanced(source="default", ref="r", confidence=0.1, evidence=ev, downgraded_from="manual")
    c = count("x", item)
    assert c.quotes == {"unverified": 0, "verbatim": 1, "near": 0, "in_layer": 1, "unsupported": 0}
    assert c.downgraded == {"manual -> default": 1}


def test_a_full_lineage_reports_each_section_from_its_own_files(tmp_path: Path) -> None:
    r = _report(tmp_path)
    assert r.evaluated_run == EVALUATED and {n.section for n in r.not_reached} == {"calibration"}
    assert r.sensitivity is not None and r.sensitivity.notes[0].startswith("no forecasts in the lineage")
    assert r.grounding is not None and r.grounding.params_by_source["llm_judgement"] == 3
    assert r.grounding.life_statements_by_source == {"manual": 1}
    assert r.grounding.page_coverage is None and "page coverage not reported" in r.grounding.notes[0]
    assert r.twin_quality is not None and r.twin_quality.components == 2
    assert (r.twin_quality.dynamic_pass_rate, r.twin_quality.quarantined) == (0.5, ("impeller",))
    by = {p.phase: p for p in r.cost.by_phase}
    assert by["ingest"].billed_usd == pytest.approx(0.2) and by["understand"].unbilled_usd == pytest.approx(0.01)
    assert by["understand"].cold_usd == pytest.approx(0.01) and r.cost.development_runs == 1
    assert r.cost.runs_total == 2 and any("lower bound" in n and "understand" in n for n in r.cost.notes)
    assert r.cost.m1 is None  # no baseline for the fixture machine
    assert all(u.calls >= 1 for u in r.prompts.uses)
    assert "## Cost" in report_md(r) and r.disclaimers[0].startswith("Absolute rates are checked for consistency")


def test_a_lineage_that_stopped_early_lists_every_section_it_never_reached(tmp_path: Path) -> None:
    r = _report(tmp_path, stop_at="understand")
    missing = {n.section: n.reason for n in r.not_reached}
    assert set(missing) == {"grounding", "twin_quality", "calibration", "sensitivity"} and not r.complete
    assert missing["grounding"].startswith("understand failed") and r.grounding is None and r.twin_quality is None
    assert {p.phase for p in r.cost.by_phase} == set(PHASE_ORDER) - {"evaluate"}
    assert any("not reached" in n for n in r.prompts.notes)


def test_a_failed_legacy_free_twin_reason_carries_the_legacy_baseline(tmp_path: Path) -> None:
    rel = SETTINGS.evaluate.legacy_baseline
    write_text_atomic(tmp_path / rel, json.dumps({"components": [{}] * 7, "dynamic_pass_rate": 1 / 7,
                                                  "full_pass_rate": 0.0}))
    r = _report(tmp_path, stop_at="understand")
    reason = next(n.reason for n in r.not_reached if n.section == "twin_quality")
    assert "dynamic pass rate 14% of 7 components" in reason


def test_rates_use_m11_definitions_life_and_provenance_are_not_dynamic() -> None:
    report = VerificationReport(**f.BASE, results=(
        CheckResult(check="loads", target="composite", status="fail"),
        CheckResult(check="monotone", target="impeller", status="pass"),
        CheckResult(check="life_consistency", target="seal", status="fail"),
    ))
    assert rates(report, ("impeller", "seal"), EvaluateSettings()) == (1.0, 0.5)
    assert rates(report, (), EvaluateSettings()) == (None, None)


def test_prompt_uses_are_locked_older_than_the_lock_or_not_locked(tmp_path: Path) -> None:
    lock = PromptLock(prompts={"a.p": LockEntry(version=2, content_sha256="c" * 64, run_scores=(
        RunScores(result="evals/results/a.p/v2.json", sample=0, metrics={"recall": 0.9}),))})
    old = tmp_path / "evals/results/a.p/v1__claude-haiku-5-5.json"
    write_text_atomic(old, json.dumps({"sample": 1, "metrics": [{"name": "recall", "value": 0.7}]}))
    rows = {"understand": [LedgerRow.model_validate_json(_row("understand", prompt_id=p, prompt_version=v,
                                                              response_cache="miss", model="m"))
                           for p, v in (("a.p", 2), ("a.p", 1), ("b.q", 1))]}
    got = {(u.prompt_id, u.version): u for u in uses(rows, lock, tmp_path)}
    assert got[("a.p", 2)].lock == "locked" and got[("a.p", 2)].run_scores[0].metrics == {"recall": 0.9}
    assert got[("a.p", 1)].lock == "older_than_lock" and got[("a.p", 1)].run_scores[0].sample == 1
    assert got[("b.q", 1)].lock == "not_locked" and got[("b.q", 1)].run_scores == ()


def test_m1_baseline_figures_are_the_documents_and_map_onto_new_phases() -> None:
    doc = (REPO_ROOT / "docs/m1_cost_baseline.md").read_text(encoding="utf-8")
    data = tomllib.loads((REPO_ROOT / EvaluateSettings().m1_baseline).read_text(encoding="utf-8"))
    for machine, entry in data["machines"].items():
        section = doc[doc.index(f"## {machine}"):]
        section = section[: section.index("### Per component")]
        for step, usd in entry["steps"].items():
            row = next(line for line in section.splitlines() if line.startswith(f"| {step} "))
            assert row.split("|")[6].strip() == f"${usd:.2f}" or (usd == 0 and "$0.0000" in row), (machine, step)
        assert f"**${entry['cold_total_usd']:.2f}**" in section
    eversys = m1("eversys_coffee_machine", EvaluateSettings())
    assert eversys is not None and eversys.by_phase_usd["ingest"] == 2.18
    assert sum(eversys.by_phase_usd.values()) == pytest.approx(5.95)  # rounded steps; the document's total is $5.96
    assert m1("air_compressor", EvaluateSettings()) is None


def test_an_unmapped_legacy_step_is_an_error() -> None:
    s = EvaluateSettings(legacy_step_phases={"1": "ingest"})
    with pytest.raises(ValueError, match="no phase"):
        m1("eversys_coffee_machine", s)


def _dump_for(path: str) -> str:
    """The report's JSON for a fresh lineage under `path` (run in a subprocess per hash seed)."""
    return dump_json(_report(Path(path)))


def test_the_report_is_identical_under_two_hash_seeds(tmp_path: Path) -> None:
    out = []
    for seed in ("0", "1"):
        target = tmp_path / seed
        code = f"from tests.unit.test_kaful_evaluate import _dump_for; print(_dump_for({str(target)!r}))"
        done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
                              env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
        out.append(done.stdout.replace(str(tmp_path / seed), "<dir>"))
    assert out[0] == out[1] and '"evaluation_report"' in out[0]


def test_the_report_tool_evaluates_a_run_that_cannot_be_a_base_run(tmp_path: Path) -> None:
    from kaful.cli import main

    machine_dir = tmp_path / f.MACHINE
    manifest = _manifest(machine_dir, stop_at="understand")
    write_text_atomic(machine_dir / f"outputs/runs/{EVALUATED}/manifest.json", dump_json(manifest))
    assert main(["evaluate", "report", f.MACHINE, "--run", EVALUATED], KafulSettings(machines_dir=tmp_path)) == 0
    saved = EvaluationReport.model_validate_json((machine_dir / f"outputs/evaluation/{EVALUATED}.json").read_bytes())
    assert not saved.complete and saved.grounding is None
    assert (machine_dir / f"outputs/evaluation/{EVALUATED}.md").read_text().startswith(f"# {f.MACHINE}: evaluation")


def test_overrides_of_the_producing_runs_are_flagged_by_what_differed(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from kaful.artifacts.manifest import Override

    manifest = _manifest(tmp_path)
    producer = f.run_manifest().model_copy(update={"overrides": (Override(
        kind="base_run", target="20261101T000000Z-aaaaaa", value="inputs differ",
        reason="config estimate.workers: 1 -> 6; machine.toml: changed", recorded_at=datetime(2026, 11, 2, tzinfo=UTC)),)})
    write_text_atomic(tmp_path / f"outputs/runs/{PRODUCER}/manifest.json", dump_json(producer))
    r = evaluate(manifest, EVALUATED, EVALUATED, tmp_path, SETTINGS, root=tmp_path)
    assert (f"run {PRODUCER} reused run 20261101T000000Z-aaaaaa although these differed: "
            "config estimate.workers, machine.toml") in r.flags
