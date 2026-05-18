"""
phases/phase3_simulate/__init__.py
------------------------------------
Phase 3: Generate synthetic simulator data.

Steps:
    a — retrieve manual context using simulation queries
    b — generate simulator plan (JSON spec)
    c — generate simulator code (Python)
    d — validate simulator output against schema

Each step saves its output to disk and skips if already done.

Call from run.py:
    from phases.phase3_simulate import run
    run(cfg, rag, schema)
"""

from phases.phase3_simulate.step_a_retrieve import run as step_a
from phases.phase3_simulate.step_b_plan import run as step_b
from phases.phase3_simulate.step_c_codegen import run as step_c
from phases.phase3_simulate.step_d_validate import run as step_d


def run(cfg: dict, rag, schema: list[dict]) -> dict:
    """
    Run all phase 3 steps in order.

    Args:
        cfg:    result of get_machine_config()
        rag:    ManualRAG instance
        schema: final schema from phase 2

    Returns:
        validation report dict
    """
    print(f"\n{'='*52}")
    print(f"Phase 3 — Simulate: {cfg['machine_id']}")
    print(f"{'='*52}")

    grouped, context = step_a(cfg, rag)
    plan             = step_b(cfg, schema, context)
    code             = step_c(cfg, schema, context, plan)
    report           = step_d(cfg, schema)

    print(f"\n✓ Phase 3 complete.")
    return report