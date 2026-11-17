"""Step e: cost. Per phase, the ledger rows of the run that produced it:
- billed: what the API charged then;
- unbilled: what its cache hits would have cost (the ledger's unbilled_cost_usd), so billed + unbilled is the cost of a
  cold run of that phase today. A cached row with no unbilled figure is counted and flagged, never guessed.
The machine's development spend is every ledger of every run, billed (failed and repeated runs included). The M1
legacy baseline (docs/m1_baseline/summary.toml) exists for eversys and CNC only; its steps are mapped onto the new
phases by settings.evaluate.legacy_step_phases.
"""

import tomllib
from pathlib import Path

from kaful.artifacts.evaluation import Cost, FileRef, M1Comparison, PhaseCost
from kaful.config import REPO_ROOT, EvaluateSettings
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.files import machine_ref, repo_ref
from kaful.phases.p9_evaluate.helpers.ledgers import all_runs, count_runs, ledger_rel, phase_rows, run_rows

CACHED = ("hit", "replay")
EVAL_SPEND = "prompt eval runs are billed to evals/results, not to a machine: not in the development spend"
NO_UNBILLED = "{n} cached ledger rows carry no unbilled cost: the cold-run cost of {phases} is a lower bound"


def m1(machine_id: str, s: EvaluateSettings, root: Path = REPO_ROOT) -> M1Comparison | None:
    path = resolve_in(root, s.m1_baseline)
    if not path.is_file():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entry = data.get("machines", {}).get(machine_id)
    if entry is None:
        return None
    unmapped = sorted(set(entry["steps"]) - set(s.legacy_step_phases))
    if unmapped:
        raise ValueError(f"legacy steps {unmapped} have no phase in settings.evaluate.legacy_step_phases")
    by_phase: dict[str, float] = {}
    for step, usd in sorted(entry["steps"].items()):
        phase = s.legacy_step_phases[step]
        by_phase[phase] = round(by_phase.get(phase, 0.0) + float(usd), 6)
    return M1Comparison(source=repo_ref(s.m1_baseline, root), cold_total_usd=entry["cold_total_usd"],
                        rerun_total_usd=entry["rerun_total_usd"], by_phase_usd=dict(sorted(by_phase.items())),
                        mapping=dict(sorted(s.legacy_step_phases.items())))


def cost(lineage: Lineage, machine_dir: Path, s: EvaluateSettings, root: Path = REPO_ROOT) -> Cost:
    by_phase, ledgers, missing, short = [], set(), 0, []
    for rec in lineage.phases:
        rows = phase_rows(machine_dir, rec.produced_by, rec.phase)
        if rec.produced_by is not None and resolve_in(machine_dir, ledger_rel(rec.produced_by)).is_file():
            ledgers.add(rec.produced_by)
        no_unbilled = sum(r.response_cache in CACHED and r.unbilled_cost_usd is None and r.event == "llm_call"
                          for r in rows)
        if no_unbilled:
            missing += no_unbilled
            short.append(rec.phase)
        by_phase.append(PhaseCost(phase=rec.phase, produced_by=rec.produced_by, calls=len(rows),
                                  billed_usd=sum(r.cost_usd or 0.0 for r in rows),
                                  unbilled_usd=sum(r.unbilled_cost_usd or 0.0 for r in rows)))
    runs = all_runs(machine_dir)
    development = sum(r.cost_usd or 0.0 for run in runs for r in run_rows(machine_dir, run))
    notes = [EVAL_SPEND] + ([NO_UNBILLED.format(n=missing, phases=", ".join(short))] if missing else [])
    refs: list[FileRef] = [machine_ref(machine_dir, ledger_rel(run)) for run in sorted(ledgers)]
    return Cost(by_phase=tuple(by_phase), ledgers=tuple(refs), development_billed_usd=development,
                development_runs=len(runs), runs_total=count_runs(machine_dir), m1=m1(lineage.machine_id, s, root), notes=tuple(notes))
