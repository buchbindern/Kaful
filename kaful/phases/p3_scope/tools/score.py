"""Score a Scope against the understand golden labels (tests/fixtures/understand/golden/<machine>.toml), M8.

Deterministic, no API calls, no new labels. Run components are matched to golden component rows by
normalized id or a row's `aliases` (the understand scorer's mapping). A golden component is:
- expected (strict) when it has a `yes` fault mode with `documented` not false;
- expected (lenient) when it has any `yes` fault mode, also `documented = false` (Natalie, 2026-11-09).

- recall (exit criterion 1, given what understand extracted): of the strict-expected golden components that a
  run component with a counted fault mode matches, the share with a full model. Expected components that no
  run component matches, or whose run components have no counted fault mode, are reported, not counted.
- criterion 1 (restated 2026-11-09, Natalie): every such run component is full_model, or connection_only with
  the flag "nothing to calibrate or observe" (overridable by declaring a channel or a life); `unflagged` lists
  the ones that are neither. Recall is still reported.
- precision, strict and lenient: of the run's full models, the share whose golden component is expected
  under that definition. A full model that matches no golden row counts against precision (unreviewed).
Not used during a run (a tool). The score is saved data: outputs/scope_scores/<run_id>.json.
"""

import hashlib
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from kaful.artifacts.base import StrictModel
from kaful.artifacts.scope import Scope
from kaful.phases.p2_understand.tools.golden_score import norm
from kaful.phases.p3_scope.c_decide import FLAG_NOTHING_TO_CALIBRATE

Row = dict[str, Any]


class Precision(StrictModel):
    value: float | None
    correct: tuple[str, ...] = Field(description="Full models whose golden component is expected: 'run -> golden'.")
    wrong: tuple[str, ...] = Field(description="Full models whose golden component is not expected.")


class ScopeScore(StrictModel):
    machine_id: str
    run_id: str
    golden_sha256: str
    full_models: tuple[str, ...]
    recall: float | None
    found: tuple[str, ...] = Field(description="Expected golden components with a full model: 'golden <- run'.")
    missed: tuple[str, ...] = Field(description="Expected and extracted with a fault mode, but no full model.")
    unflagged: tuple[str, ...] = Field(description=(
        "Run components of missed golden components that are neither full_model nor connection_only with the "
        "'nothing to calibrate or observe' flag (criterion 1 as restated); empty passes."))
    not_extracted: tuple[str, ...] = Field(description="Expected golden components no run component matches.")
    no_fault_mode: tuple[str, ...] = Field(description="Expected and matched, but understand gave no counted mode.")
    precision_strict: Precision
    precision_lenient: Precision
    unmatched_full_models: tuple[str, ...] = Field(description="Full models with no golden component row.")


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 6) if d else None


def _lookup(rows: Sequence[Row]) -> dict[str, str]:
    return {norm(name): row["id"] for row in rows for name in (row["id"], *row.get("aliases", ()))}


def expected(golden: Row) -> tuple[set[str], set[str]]:
    """(strict, lenient) golden component ids with gradual fault modes."""
    lookup = _lookup(golden["components"])
    rows = golden["fault_modes"] + golden.get("candidate_fault_modes", [])
    yes = [(lookup.get(norm(r["component"]), str(r["component"])), r) for r in rows if r["verdict"] == "yes"]
    return {g for g, r in yes if r.get("documented", True)}, {g for g, _ in yes}


def _precision(full: dict[str, str | None], want: set[str]) -> Precision:
    matched = {run: g for run, g in full.items() if g is not None}
    correct = sorted(f"{run} -> {g}" for run, g in matched.items() if g in want)
    wrong = sorted(f"{run} -> {g}" for run, g in matched.items() if g not in want)
    return Precision(value=_ratio(len(correct), len(full)), correct=tuple(correct), wrong=tuple(wrong))


def score(golden_path: Path, scope: Scope) -> ScopeScore:
    text = golden_path.read_bytes()
    golden = tomllib.loads(text.decode("utf-8"))
    if golden["machine_id"] != scope.machine_id:
        raise ValueError(f"golden labels are for {golden['machine_id']!r}, the scope for {scope.machine_id!r}")
    lookup = _lookup(golden["components"])
    to_golden = {d.component: lookup.get(norm(d.component)) for d in scope.decisions}
    full = {d.component: to_golden[d.component] for d in scope.decisions if d.decision == "full_model"}
    strict, lenient = expected(golden)
    runs: dict[str, list[str]] = {}
    for d in scope.decisions:
        g = to_golden[d.component]
        if g is not None:
            runs.setdefault(g, []).append(d.component)
    has_mode = {d.component for d in scope.decisions if d.signals.fault_modes}
    counted = sorted(g for g in strict if any(r in has_mode for r in runs.get(g, ())))
    found = [f"{g} <- {', '.join(sorted(r for r in runs[g] if r in full))}" for g in counted
             if any(r in full for r in runs[g])]
    flagged = {d.component for d in scope.decisions
               if d.decision == "connection_only" and FLAG_NOTHING_TO_CALIBRATE in d.flags}
    unflagged = sorted(r for g in counted if not any(r in full for r in runs[g])
                       for r in runs[g] if r in has_mode and r not in flagged)
    return ScopeScore(
        machine_id=scope.machine_id, run_id=scope.run_id, golden_sha256=hashlib.sha256(text).hexdigest(),
        full_models=tuple(sorted(full)), recall=_ratio(len(found), len(counted)), found=tuple(found),
        missed=tuple(g for g in counted if not any(r in full for r in runs[g])), unflagged=tuple(unflagged),
        not_extracted=tuple(sorted(g for g in strict if g not in runs)),
        no_fault_mode=tuple(sorted(g for g in strict if g in runs and g not in counted)),
        precision_strict=_precision(full, strict), precision_lenient=_precision(full, lenient),
        unmatched_full_models=tuple(sorted(run for run, g in full.items() if g is None)))
