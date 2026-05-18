"""
phases/phase5_triage/__init__.py
----------------------------------
Phase 5: Triage components for digital twin modeling.

Steps:
    a — assign schema fields to components
    b — triage each component (full_component / simple_state / exclude)
    c — filter by priority and manual exclusions

Each step saves its output to disk and skips if already done.

Call from run.py:
    from phases.phase5_triage import run
    full_components = run(cfg, schema, machine_model)
"""

from phases.phase5_triage.step_a_assign import run as step_a
from phases.phase5_triage.step_b_triage import run as step_b
from phases.phase5_triage.step_c_filter import run as step_c


def run(cfg: dict, schema: list[dict], machine_model: dict) -> list[dict]:
    """
    Run all phase 5 steps in order.

    Args:
        cfg:           result of get_machine_config()
        schema:        final schema from phase 2
        machine_model: resolved machine model from phase 4

    Returns:
        filtered list of full_component dicts
    """
    print(f"\n{'='*52}")
    print(f"Phase 5 — Triage: {cfg['machine_id']}")
    print(f"{'='*52}")

    assignments, comp_fields = step_a(cfg, schema, machine_model)
    triage                   = step_b(cfg, machine_model, comp_fields)
    full_components          = step_c(cfg, triage)

    print(f"\n✓ Phase 5 complete — {len(full_components)} full components.")
    return full_components