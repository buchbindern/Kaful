"""Step c: the dynamic checks (kaful.dynamics.trace_checks) on every binding in every simulation of step b, merged into
one result per (component, check): it fails when any run fails it, and its message names each failing run.

- starts_healthy, finite_bounded, monotone, event_agreement, maintenance_resets: every run (maintenance_resets
  is `skipped` for a binding with no scheduled reset and no renewal rule);
- life_consistency: the nominal run at prior medians only, against the component's median `life_nominal`.
A failing component is quarantined in step d.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult
from kaful.config import VerifySettings
from kaful.phases.p6_verify.b_simulate import MEDIAN, NOMINAL, Simulated
from kaful.dynamics.trace_checks import (
    Finding,
    event_agreement,
    finite_bounded,
    life_consistency,
    maintenance_resets,
    monotone,
    starts_healthy,
)
from kaful.dynamics.traces import BindingTrace, binding_trace

EVERY_RUN: tuple[tuple[str, Callable[[BindingTrace, float], Finding | None]], ...] = (
    ("event_agreement", event_agreement), ("finite_bounded", finite_bounded),
    ("maintenance_resets", maintenance_resets), ("monotone", monotone), ("starts_healthy", starts_healthy))
LIFE = "life_consistency"
DYNAMIC_CHECKS = frozenset({*(c for c, _ in EVERY_RUN), LIFE})  # the checks that quarantine a component (step d)
NOT_APPLICABLE = {"maintenance_resets": "no scheduled reset and no renewal rule"}


def _finite(values: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in sorted(values.items()) if np.isfinite(v)}


def _merge(component: str, check: str, findings: Sequence[tuple[str, Finding]]) -> CheckResult:
    if not findings:
        return CheckResult(check=check, target=component, status="skipped", message=NOT_APPLICABLE.get(check, ""))
    failing = [(run, f) for run, f in findings if not f.ok]
    run, shown = failing[0] if failing else findings[0]
    return CheckResult(check=check, target=component, status="fail" if failing else "pass",
                       measured=_finite(shown.measured), limits=_finite(shown.limits),
                       message="; ".join(f"{r}: {f.message}" for r, f in failing))


def dynamic_results(twin: TwinDefinition, ps: ParamSet, sims: Sequence[Simulated],
                    v: VerifySettings) -> tuple[CheckResult, ...]:
    out = []
    for b in sorted(twin.bindings, key=lambda b: b.component):
        findings: dict[str, list[tuple[str, Finding]]] = defaultdict(list)
        for s in sims:
            trace = binding_trace(twin, b, s.result, s.profile, s.values)
            for check, run_check in EVERY_RUN:
                found = run_check(trace, v.damage_tolerance)
                if found is not None:
                    findings[check].append((s.run.name, found))
            if (s.run.profile, s.run.quantile) == (NOMINAL, MEDIAN):
                median = ps.get(f"{b.component}.life_nominal").prior.quantile(MEDIAN)
                findings[LIFE].append((s.run.name, life_consistency(trace, median, v.life_band)))
        out += [_merge(b.component, check, findings[check]) for check in sorted(DYNAMIC_CHECKS)]
    return tuple(out)
