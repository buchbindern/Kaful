"""
phases/phase2_component_map/__init__.py
-----------------------------------------
Phase 2: Map sensor signals to physical components and assign ProgPy roles.

Steps:
    a — build signal schema from phase 1 outputs (no LLM)
    b — LLM maps signal channels to physical components
    c — LLM (or queries.py short-circuit) identifies degradation pathway
    d — LLM assigns ProgPy variable roles (states, inputs, outputs)

Each step saves its output to disk and skips if already done.

Call from run_data.py:
    from phases.phase2_component_map import run
    progpy_roles = run(cfg, force=False)
"""

from phases.phase2_component_map.step_a_signal_schema      import run as step_a
from phases.phase2_component_map.step_b_component_map      import run as step_b
from phases.phase2_component_map.step_c_degradation_pathway import run as step_c
from phases.phase2_component_map.step_d_progpy_roles        import run as step_d


def run(cfg: dict, force: bool = False) -> dict:
    """
    Run all phase 2 steps in order.

    Args:
        cfg:   result of get_record_config()
        force: if True, delete existing outputs and re-run all steps

    Returns:
        progpy_roles dict (step d output) — consumed by phases 3 and 4
    """
    dataset_id = cfg["dataset_id"]
    record_id  = cfg["record_id"]

    print(f"\n{'='*52}")
    print(f"Phase 2 — Component Map: {dataset_id} / {record_id}")
    print(f"{'='*52}")

    if force:
        print("  --force: clearing phase 2 outputs")
        for key in [
            "step_a_schema_path",
            "step_b_map_path",
            "step_c_pathway_path",
            "step_d_progpy_path",
        ]:
            p = cfg[key]
            if p.exists():
                p.unlink()
                print(f"    deleted {p.name}")

    signals       = step_a(cfg)
    component_map = step_b(cfg, signals)
    pathway       = step_c(cfg, component_map)
    progpy_roles  = step_d(cfg, signals, component_map, pathway)

    states  = progpy_roles.get("state_variables", [])
    outputs = progpy_roles.get("output_variables", [])
    filter_features = [o["name"] for o in outputs if o.get("use_for_filter")]

    print(f"\n✓ Phase 2 complete — {dataset_id} / {record_id}")
    print(f"    {len(states)} state variable(s): {[s['name'] for s in states]}")
    print(f"    {len(filter_features)} filter feature(s): {filter_features}")

    return progpy_roles