"""Step c: prompt quality. Per prompt version the lineage called (read from the producing runs' ledgers, so a reused
or cached phase still names its prompts), its lock and the per-run eval scores behind it; plus the machine's own
understand score against its golden labels, when one exists for the understand run used.

Eval sets mix machines, so prompt scores are per version, not per machine. A hold-out machine's scores are labelled
report only (prompts are never tuned on it).
"""

import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from kaful.artifacts.evaluation import MachineScore, Prompts, PromptUse, RunScore
from kaful.config import REPO_ROOT, EvaluateSettings, ReuseRule
from kaful.llm.ledger import LedgerRow
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.files import machine_ref
from kaful.phases.p9_evaluate.helpers.ledgers import phase_rows
from kaful.prompts.registry import LOCK_FILE, PromptLock, read_lock

EVAL_RESULTS = "evals/results"
HOLDOUT = "hold-out machine: report only, prompts are never tuned on it"


def _older_results(prompt_id: str, version: int, root: Path) -> tuple[RunScore, ...]:
    folder = resolve_in(root, f"{EVAL_RESULTS}/{prompt_id}")
    out = []
    for path in sorted(folder.glob(f"v{version}__*.json")) if folder.is_dir() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = {m["name"]: float(m["value"]) for m in data.get("metrics", []) if m.get("value") is not None}
        out.append(RunScore(result=f"{EVAL_RESULTS}/{prompt_id}/{path.name}", sample=int(data.get("sample", 0)),
                            metrics=dict(sorted(metrics.items()))))
    return tuple(out)


def uses(rows_by_phase: dict[str, list[LedgerRow]], lock: PromptLock, root: Path,
         reuse: Mapping[str, ReuseRule] | None = None) -> tuple[PromptUse, ...]:
    """Each prompt version called: locked; equivalent to the lock (its reuse rule serves this version's replies
    under the lock, which differs only in max_tokens; the lock's scores apply); older than the lock; or not locked."""
    calls: dict[tuple[str, int], list[LedgerRow]] = defaultdict(list)
    phase_of: dict[tuple[str, int], str] = {}
    for phase, rows in sorted(rows_by_phase.items()):
        for r in rows:
            if r.event == "llm_call" and r.prompt_id and r.prompt_version is not None:
                calls[(r.prompt_id, r.prompt_version)].append(r)
                phase_of.setdefault((r.prompt_id, r.prompt_version), phase)
    out = []
    for (pid, version), rows in sorted(calls.items()):
        entry = lock.prompts.get(pid)
        rule = (reuse or {}).get(pid)
        if entry is not None and (entry.version == version or (rule is not None and rule.version == version
                                                                and entry.version > version)):
            state = "locked" if entry.version == version else "equivalent_to_lock"
            scores = tuple(RunScore(result=s.result, sample=s.sample, metrics=dict(sorted(
                s.metrics.items()))) for s in entry.run_scores)
        else:
            state, scores = ("older_than_lock" if entry is not None else "not_locked"), _older_results(pid, version, root)
        out.append(PromptUse(prompt_id=pid, version=version, phase=phase_of[(pid, version)],
                             models=tuple(sorted({r.model for r in rows if r.model})), calls=len(rows), lock=state,
                             locked_version=entry.version if entry else None, run_scores=scores))
    return tuple(out)


def _numbers(node: object, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a score file, as dotted keys (lists are skipped: they are per-item detail)."""
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for k in sorted(node):
            v = node[k]
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                out[f"{prefix}{k}"] = float(v)
            elif isinstance(v, dict):
                out.update(_numbers(v, f"{prefix}{k}."))
    return out


def machine_scores(lineage: Lineage, machine_dir: Path, s: EvaluateSettings) -> tuple[MachineScore, ...]:
    run = lineage.get("understand").produced_by
    if run is None or not lineage.reached("understand"):
        return ()
    rel = f"{s.understand_scores}/{run}.json"
    if not resolve_in(machine_dir, rel).is_file():
        return ()
    data = json.loads(resolve_in(machine_dir, rel).read_text(encoding="utf-8"))
    sections = {k: v for k, v in data.items() if k in ("components", "channels", "fault_modes", "values", "grounding")}
    note = HOLDOUT if lineage.machine_id in s.holdout_machines else ""
    return (MachineScore(tool="understand score (golden labels)", file=machine_ref(machine_dir, rel),
                         metrics=_numbers(sections), note=note),)


def prompt_quality(lineage: Lineage, machine_dir: Path, s: EvaluateSettings, root: Path = REPO_ROOT,
                   lock_path: Path = LOCK_FILE, reuse: Mapping[str, ReuseRule] | None = None) -> Prompts:
    rows: dict[str, list[LedgerRow]] = {rec.phase: phase_rows(machine_dir, rec.produced_by, rec.phase) for rec in lineage.phases
            if rec.produced_by is not None}
    notes = []
    if lineage.machine_id in s.holdout_machines:
        notes.append(f"{lineage.machine_id} is a {HOLDOUT}")
    if not lineage.reached("understand"):
        notes.append(f"understand not reached ({lineage.why_not('understand')}): its prompts' calls are listed, its "
                     "outputs are not scored")
    return Prompts(uses=uses(rows, read_lock(lock_path), root, reuse), machine_scores=machine_scores(lineage, machine_dir, s),
                   notes=tuple(notes))
