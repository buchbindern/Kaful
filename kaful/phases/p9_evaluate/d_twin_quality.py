"""Step d: twin quality. The lineage's verification report per component, with M11's definitions, so the rates
compare with the legacy baseline (outputs/legacy_verification/legacy_baseline.json):
- dynamic pass: no failing check other than life and provenance (settings.evaluate.non_dynamic_checks);
- full pass: no failing check.
Components are the bindings' components of the twin verify checked (build's), counted before quarantine; twin-level
targets (connections, 'composite') are reported among the failing checks only.
"""

import json
from collections import Counter
from pathlib import Path

from kaful.artifacts.evaluation import LegacyBaselineRef, TwinQuality
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import VerificationReport
from kaful.config import EvaluateSettings
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.files import machine_ref
from kaful.runs import load_artifact


def legacy(machine_dir: Path, s: EvaluateSettings) -> LegacyBaselineRef | None:
    if not resolve_in(machine_dir, s.legacy_baseline).is_file():
        return None
    data = json.loads(resolve_in(machine_dir, s.legacy_baseline).read_text(encoding="utf-8"))
    return LegacyBaselineRef(file=machine_ref(machine_dir, s.legacy_baseline), components=len(data["components"]),
                             dynamic_pass_rate=data["dynamic_pass_rate"], full_pass_rate=data["full_pass_rate"])


def rates(report: VerificationReport, components: tuple[str, ...],
          s: EvaluateSettings) -> tuple[float | None, float | None]:
    if not components:
        return None, None
    failing = {(r.target, r.check) for r in report.results if r.status == "fail"}
    dynamic = sum(not any(t == c and k not in s.non_dynamic_checks for t, k in failing) for c in components)
    full = sum(not any(t == c for t, _ in failing) for c in components)
    return dynamic / len(components), full / len(components)


def twin_quality(lineage: Lineage, machine_dir: Path, s: EvaluateSettings) -> TwinQuality:
    """Called only when verify was reached."""
    ref, twin_ref = lineage.ref("verify", "verification_report"), lineage.ref("build", "twin_definition")
    if ref is None or twin_ref is None:
        raise ValueError(f"verify reached but its report or build's twin is missing in run {lineage.run_id}")
    report = load_artifact(machine_dir, ref, VerificationReport)
    twin = load_artifact(machine_dir, twin_ref, TwinDefinition)
    components = tuple(sorted({b.component for b in twin.bindings}))
    dynamic, full = rates(report, components, s)
    failing = Counter(r.check for r in report.results if r.status == "fail")
    return TwinQuality(report=ref, components=len(components), dynamic_pass_rate=dynamic, full_pass_rate=full,
                       passed=report.passed, quarantined=report.quarantined_components,
                       failing_checks=dict(sorted(failing.items())), legacy=legacy(machine_dir, s))


def not_reached_reason(lineage: Lineage, machine_dir: Path, s: EvaluateSettings) -> str:
    """Why there is no twin to judge, with the legacy baseline when one exists (CDV)."""
    base = lineage.why_not("verify")
    old = legacy(machine_dir, s)
    if old is None:
        return base
    return (f"{base}; legacy baseline ({old.file.path}): dynamic pass rate {old.dynamic_pass_rate:.0%} of "
            f"{old.components} components")
