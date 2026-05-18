"""
phases/phase4_twin/__init__.py
--------------------------------
Phase 4: Build the digital twin machine model.

Steps:
    a — machine comprehension (components, flow paths, operation sequence)
    b — component physics (operating ranges, degradation, faults, maintenance)
    c — merge physics, resolve missing units, flag no-physics components
    d — validate machine model structure

Each step saves its output to disk and skips if already done.
To re-run a step, delete its output file and re-run.

Call from run.py:
    from phases.phase4_twin import run
    model, issues, warnings = run(cfg, rag)
"""

from phases.phase4_twin.step_a_comprehend import run as step_a
from phases.phase4_twin.step_b_physics import run as step_b
from phases.phase4_twin.step_c_resolve import run as step_c
from phases.phase4_twin.step_d_validate import run as step_d


def run(cfg: dict, rag) -> tuple[dict, list, list]:
    """
    Run all phase 4 steps in order.

    Args:
        cfg: result of get_machine_config()
        rag: ManualRAG instance (already indexed from phase 1)

    Returns:
        tuple of (resolved_model, issues, warnings)
    """
    print(f"\n{'='*52}")
    print(f"Phase 4 — Digital Twin: {cfg['machine_id']}")
    print(f"{'='*52}")

    model            = step_a(cfg, rag)
    component_physics = step_b(cfg, rag, model)
    resolved         = step_c(cfg, rag, model, component_physics)
    issues, warnings = step_d(cfg, resolved)

    print(f"\n✓ Phase 4 complete — {len(resolved['components'])} components, "
          f"{len(issues)} issues, {len(warnings)} warnings.")

    return resolved, issues, warnings