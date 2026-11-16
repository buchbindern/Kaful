"""Step c: the dynamic checks (kaful.dynamics.trace_checks) on every binding in every simulation of step b, merged into
one result per (component, check): it fails when any run fails it, and its message names each failing run.

- starts_healthy, finite_bounded, monotone, event_agreement, maintenance_resets: every run (maintenance_resets
  is `skipped` for a binding with no scheduled reset and no renewal rule);
- life_consistency: the nominal run at prior medians only, against the component's median `life_nominal`; for a twin
  with connections, the uncoupled nominal run (every coupling neutral), so the calibration is checked as before;
- coupled_life_consistency (M13b): a connection's to-binding, on the coupled nominal run at medians, against the
  coupling form's closed-form crossing time (`skipped` for a binding no connection feeds).
A failing component is quarantined in step d.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult
from kaful.config import VerifySettings
from kaful.phases.p6_verify.b_simulate import MEDIAN, NOMINAL, UNCOUPLED, Simulated
from kaful.dynamics.trace_checks import (
    Finding,
    coupled_life_consistency,
    event_agreement,
    finite_bounded,
    life_consistency,
    maintenance_resets,
    monotone,
    starts_healthy,
)
from kaful.dynamics.traces import BindingTrace, binding_trace
from kaful.artifacts.twin_definition import TemplateBinding
from kaful.templates.couplings import COUPLINGS

EVERY_RUN: tuple[tuple[str, Callable[[BindingTrace, float], Finding | None]], ...] = (
    ("event_agreement", event_agreement), ("finite_bounded", finite_bounded),
    ("maintenance_resets", maintenance_resets), ("monotone", monotone), ("starts_healthy", starts_healthy))
LIFE, COUPLED_LIFE = "life_consistency", "coupled_life_consistency"
DYNAMIC_CHECKS = frozenset({*(c for c, _ in EVERY_RUN), LIFE, COUPLED_LIFE})  # they quarantine a component (step d)
NOT_APPLICABLE = {"maintenance_resets": "no scheduled reset and no renewal rule",
                  COUPLED_LIFE: "no single connection feeds it (none; several have no closed form)"}
COUPLED_RUN = f"{NOMINAL}@q{MEDIAN:g}"


def coupled_crossing(twin: TwinDefinition, b: TemplateBinding, values: dict[str, float]) -> tuple[float, float] | None:
    """(closed-form coupled crossing time, uncoupled life) of a binding one connection feeds, at `values`, with every
    binding at nominal conditions (f = 1); None when no connection, or several, feed it (no closed form)."""
    links = [c for c in twin.connections if c.to_binding == b.id]
    if len(links) != 1:
        return None
    c, src = links[0], next(x for x in twin.bindings if x.id == links[0].from_binding)
    renews = any(r.binding == src.id and r.trigger == "threshold" for r in twin.maintenance)
    source_life = values[src.threshold_param] / values[src.params["k"]]
    target_life = values[b.threshold_param] / values[b.params["k"]]
    slots = {slot: values[pid] for slot, pid in c.params.items()}
    return COUPLINGS[c.form]().crossing_time(target_life, source_life, slots, renews), target_life


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
    life_run = UNCOUPLED if twin.connections else COUPLED_RUN
    for b in sorted(twin.bindings, key=lambda b: b.component):
        findings: dict[str, list[tuple[str, Finding]]] = defaultdict(list)
        for s in sims:
            trace = binding_trace(twin, b, s.result, s.profile, s.values)
            for check, run_check in EVERY_RUN:
                found = run_check(trace, v.damage_tolerance)
                if found is not None:
                    findings[check].append((s.run.name, found))
            if s.run.name == life_run:
                median = ps.get(f"{b.component}.life_nominal").prior.quantile(MEDIAN)
                findings[LIFE].append((s.run.name, life_consistency(trace, median, v.life_band)))
            crossing = coupled_crossing(twin, b, s.values) if s.run.name == COUPLED_RUN else None
            if crossing is not None:
                findings[COUPLED_LIFE].append((s.run.name, coupled_life_consistency(trace, *crossing, v.life_band)))
        out += [_merge(b.component, check, findings[check]) for check in sorted(DYNAMIC_CHECKS)]
    return tuple(out)
