"""The evaluate phase: one evaluation report of a run's lineage, from saved artifacts and files only (M14). No API call.

  a_resolve_lineage  per phase, its status and the run that produced its artifacts
  b_grounding        evidence quotes by verification class, code downgrades, parameters by source and role, ingest page
                     coverage (helpers/files)
  c_prompt_quality   per prompt version called, its lock and per-run eval scores; the machine's understand score
                     (helpers/ledgers, helpers/files)
  d_twin_quality     verification per component against the M11 legacy baseline (helpers/files)
  e_cost             billed, unbilled and cold cost per phase; development spend; the M1 baseline (helpers/ledgers,
                     helpers/files)
The report and its markdown rendering (helpers/render) are written by the phase; a section whose phase the lineage
never reached is listed with the reason. `kaful evaluate report <machine> --run RUN` (tools/commands) builds the same
report for a run that cannot be a base run (CDV: understand failed its gate), through `evaluate` below.
"""

import logging
from pathlib import Path

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.evaluation import EvaluationReport, NotReached, Section
from kaful.artifacts.manifest import PhaseRecord, RunManifest
from kaful.artifacts.param_set import ParamSet
from kaful.config import REPO_ROOT, KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage, resolve
from kaful.phases.p9_evaluate.b_grounding import grounding
from kaful.phases.p9_evaluate.c_prompt_quality import prompt_quality
from kaful.phases.p9_evaluate.d_twin_quality import not_reached_reason, twin_quality
from kaful.phases.p9_evaluate.e_cost import cost
from kaful.phases.p9_evaluate.helpers.render import report_md
from kaful.runs import MANIFEST_FILE, RunContext, load_artifact, now_utc, read_manifest, runs_dir, write_text_atomic

log = logging.getLogger("kaful.evaluate")

PENDING: dict[Section, str] = {"calibration": "not built yet (M14 checkpoint 2)", "sensitivity": "not built yet (M14 checkpoint 3)"}
FLAG_LOCK = "{pid} v{v} was called but is {state} (locked: v{locked})"
FLAG_OVERRIDE = "run {run} reused run {base} although these differed: {keys}"
MD_NAME = "evaluation_report.md"


def _inputs(lineage: Lineage, kinds: tuple[str, ...]) -> tuple[ArtifactRef, ...]:
    refs = {(r.kind, r.path): r for p in lineage.phases if lineage.reached(p.phase) for r in p.artifacts
            if r.kind in kinds}
    return tuple(v for _, v in sorted(refs.items()))


def _time_unit(lineage: Lineage, machine_dir: Path) -> str | None:
    ref = lineage.ref("parameterize", "param_set") if lineage.reached("parameterize") else None
    return load_artifact(machine_dir, ref, ParamSet).time_unit if ref else None


def _keys(reason: str) -> str:
    """'config estimate.workers: 1 -> 6; machine.toml: changed' -> 'config estimate.workers, machine.toml'."""
    return ", ".join(sorted({part.split(":")[0].strip() for part in reason.split("; ") if part.strip()}))


def lineage_overrides(lineage: Lineage, machine_dir: Path) -> set[str]:
    """Base-run overrides recorded by the runs that produced the lineage (what was reused across a difference)."""
    out = set()
    for run in sorted({p.produced_by for p in lineage.phases if p.produced_by}):
        if not (runs_dir(machine_dir) / run / MANIFEST_FILE).is_file():
            continue
        for o in read_manifest(runs_dir(machine_dir) / run).overrides:
            if o.kind == "base_run":
                out.add(FLAG_OVERRIDE.format(run=run, base=o.target, keys=_keys(o.reason)))
    return out


def evaluate(manifest: RunManifest, evaluated_run: str, report_run_id: str, machine_dir: Path,
             settings: KafulSettings, root: Path = REPO_ROOT) -> EvaluationReport:
    """Steps a to e on the manifest's lineage."""
    s = settings.evaluate
    lineage = resolve(manifest)
    missing: list[NotReached] = []
    ground = grounding(lineage, machine_dir, s, root) if lineage.reached("understand") else None
    if ground is None:
        missing.append(NotReached(section="grounding", reason=lineage.why_not("understand")))
    prompts = prompt_quality(lineage, machine_dir, s, root)
    twin = twin_quality(lineage, machine_dir, s) if lineage.reached("verify") else None
    if twin is None:
        missing.append(NotReached(section="twin_quality", reason=not_reached_reason(lineage, machine_dir, s)))
    spend = cost(lineage, machine_dir, s, root)
    missing += [NotReached(section=k, reason=v) for k, v in sorted(PENDING.items())]
    flags = {FLAG_LOCK.format(pid=u.prompt_id, v=u.version, state=u.lock.replace("_", " "), locked=u.locked_version)
             for u in prompts.uses if u.lock != "locked"}
    flags |= lineage_overrides(lineage, machine_dir)
    return EvaluationReport(
        run_id=report_run_id, machine_id=manifest.machine_id, evaluated_run=evaluated_run,
        complete=not missing, time_unit=_time_unit(lineage, machine_dir), lineage=lineage.phases, grounding=ground,
        prompts=prompts, twin_quality=twin, cost=spend, not_reached=tuple(sorted(missing, key=lambda n: n.section)),
        flags=tuple(sorted(flags)),
        inputs=_inputs(lineage, (*s.evidence_kinds, "verification_report")))


class EvaluatePhase:
    name: PhaseName = "evaluate"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        evaluated = ctx.manifest.base_run.run_id if ctx.manifest.base_run else ctx.manifest.run_id
        report = evaluate(ctx.manifest, evaluated, ctx.manifest.run_id, ctx.machine_dir, ctx.settings)
        ref = ctx.save_artifact(self.name, report)
        write_text_atomic(ctx.run_dir / self.name / MD_NAME, report_md(report))
        notes = tuple(f"not reached: {n.section}: {n.reason}" for n in report.not_reached)
        log.info("evaluate finished", extra={"fields": {"complete": report.complete}})
        return PhaseRecord(phase=self.name, status="passed", started_at=started, finished_at=now_utc(),
                           artifacts=(ref,), notes=notes)


__all__ = ["EvaluatePhase", "MD_NAME", "evaluate"]
