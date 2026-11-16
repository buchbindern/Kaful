"""Step c: verify's dynamic checks (kaful.dynamics.trace_checks) on one unit's truth, every binding.

starts_healthy, finite_bounded, monotone, event_agreement and maintenance_resets, as verify runs them on its corner
runs; life_consistency does not apply (a unit's life is drawn, not the median). Any failure fails the phase (hard
rule 2): a synthetic dataset is only written from truth the twin's own checks accept.
"""

from collections.abc import Mapping

from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import SimulationResult
from kaful.dynamics.trace_checks import (
    event_agreement,
    finite_bounded,
    maintenance_resets,
    monotone,
    starts_healthy,
)
from kaful.dynamics.traces import binding_trace

CHECKS = (("event_agreement", event_agreement), ("finite_bounded", finite_bounded),
          ("maintenance_resets", maintenance_resets), ("monotone", monotone), ("starts_healthy", starts_healthy))


def check_unit(unit: str, twin: TwinDefinition, result: SimulationResult, profile: OperatingProfile,
               values: Mapping[str, float], tolerance: float) -> list[str]:
    """One line per failing (binding, check), sorted."""
    out = []
    for b in sorted(twin.bindings, key=lambda b: b.component):
        trace = binding_trace(twin, b, result, profile, values)
        for name, check in CHECKS:
            found = check(trace, tolerance)
            if found is not None and not found.ok:
                out.append(f"{unit} {b.component} {name}: {found.message}")
    return out
