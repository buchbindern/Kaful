"""
phases/phase6_codegen/__init__.py
-----------------------------------
Phase 6: Generate ProgPy component code and composite model.

Steps:
    a — generate ProgPy spec per component
    b — generate ProgPy component code per component
    c — runtime validate each component
    d — build composite model (connections, external inputs, codegen, validate)

Each step saves its output to disk and skips if already done.
Components are saved individually — rerun one by deleting its file.

Call from run.py:
    from phases.phase6_codegen import run
    result = run(cfg, full_components, specs, machine_model)
"""

from phases.phase6_codegen.step_a_spec import run as step_a
from phases.phase6_codegen.step_b_generate import run as step_b
from phases.phase6_codegen.step_c_validate import run as step_c
from phases.phase6_codegen.step_d_composite import run as step_d


def run(cfg: dict, full_components: list[dict], machine_model: dict) -> dict:
    """
    Run all phase 6 steps in order.

    Args:
        cfg:             result of get_machine_config()
        full_components: filtered list from phase 5
        machine_model:   resolved machine model from phase 4

    Returns:
        composite model result dict
    """
    print(f"\n{'='*52}")
    print(f"Phase 6 — Codegen: {cfg['machine_id']}")
    print(f"{'='*52}")

    specs   = step_a(cfg, full_components)
    code    = step_b(cfg, specs)
    reports = step_c(cfg, specs)

    # Check validation results
    failed = [name for name, r in reports.items() if r["status"] == "fail"]
    if failed:
        print(f"\n⚠ {len(failed)} components failed validation: {failed}")
        print(f"  Review and fix before running composite.")

    result = step_d(cfg, full_components, specs, machine_model)

    print(f"\n✓ Phase 6 complete — composite model ready.")
    return result