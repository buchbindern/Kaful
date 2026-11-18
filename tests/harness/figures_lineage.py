"""A complete evaluate lineage of the small synthesize/estimate test machine, saved on disk the way the pipeline
saves it (M16 figures tests): factory artifacts for ingest..scope, the synthesize fixture's twin, contract and
ParamSet, real synthesize and estimate outputs, and an evaluate run whose report is the real phase's output."""

from pathlib import Path

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ParamSet
from kaful.config import KafulSettings
from kaful.paths import to_rel
from kaful.phases import PHASE_ORDER
from kaful.phases.p8_estimate.phase import estimate
from kaful.phases.p9_evaluate.phase import evaluate
from kaful.runs import dump_json, sha256_bytes, write_text_atomic
from tests.harness import kaful_factories as f
from tests.harness.kaful_factories import BASE

MACHINE = "tiny_figures"
EVALUATE_RUN = "20261103T130000Z-f16a00"


def _save(machine_dir: Path, phase: str, artifact: ArtifactBase, name: str | None = None) -> ArtifactRef:
    kind = str(getattr(artifact, "kind"))
    target = machine_dir / "outputs" / "runs" / BASE["run_id"] / phase / (name or f"{kind}.json")
    text = dump_json(artifact)
    write_text_atomic(target, text)
    return ArtifactRef(kind=kind, run_id=BASE["run_id"], path=to_rel(machine_dir, target),
                       sha256=sha256_bytes(text.encode()))


def build_lineage(machines_dir: Path) -> KafulSettings:
    """Writes machines_dir/MACHINE with the lineage; returns settings pointing at machines_dir."""
    from tests.unit.test_kaful_estimate_phase import _settings
    from tests.unit.test_kaful_synthesize import _synthesize

    machine_dir = machines_dir / MACHINE
    settings = _settings().model_copy(update={"machines_dir": machines_dir})
    done, twin, contract = _synthesize(machine_dir, settings)
    ps = _twin_param_set()
    est_dir = machine_dir / "outputs" / "runs" / BASE["run_id"] / "estimate"

    def save_est(artifact: ArtifactBase, name: str) -> ArtifactRef:
        return _save(machine_dir, "estimate", artifact, name)

    est = estimate({**BASE}, twin, ps, contract, done.dataset, settings, machine_dir, est_dir, save_est)
    saved = {"ingest": (f.document_index(),), "understand": (f.machine_model(), contract), "scope": (f.scope(),),
             "parameterize": (ps,), "build": (twin,), "verify": (twin, f.verification_report()),
             "synthesize": (done.dataset,), "estimate": (est.forecast_set,)}
    records = []
    for p in PHASE_ORDER:
        if p == "evaluate":
            continue
        refs = tuple(sorted((_save(machine_dir, p, a) for a in saved[p]), key=lambda r: (r.kind, r.path)))
        records.append(PhaseRecord(phase=p, status="reused", artifacts=refs))
    manifest = f.run_manifest().model_copy(update={"run_id": EVALUATE_RUN, "machine_id": MACHINE,
                                                   "phases": tuple(records)})
    write_text_atomic(machine_dir / "outputs" / "runs" / BASE["run_id"] / "ledger.jsonl", "")
    report = evaluate(manifest, EVALUATE_RUN, EVALUATE_RUN, machine_dir, settings, root=machines_dir)
    text = dump_json(report)
    rel = f"outputs/runs/{EVALUATE_RUN}/evaluate/evaluation_report.json"
    write_text_atomic(machine_dir / rel, text)
    ref = ArtifactRef(kind="evaluation_report", run_id=EVALUATE_RUN, path=rel, sha256=sha256_bytes(text.encode()))
    final = manifest.model_copy(update={"phases": (*records, PhaseRecord(phase="evaluate", status="passed",
                                                                         artifacts=(ref,)))})
    write_text_atomic(machine_dir / "outputs" / "runs" / EVALUATE_RUN / "manifest.json", dump_json(final))
    return settings


def _twin_param_set() -> ParamSet:
    from tests.unit.test_kaful_synthesize import _twin

    ps: ParamSet = _twin()[1]
    return ps
