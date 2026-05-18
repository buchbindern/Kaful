"""
phases/phase5_triage/step_c_filter.py
---------------------------------------
Step C: Filter full components by priority and manual exclusions.

Drops low-priority components and any manually excluded components
before passing to codegen.

Output saved to: outputs/triage/step_c_full_components.json
"""

import json
import importlib.util


def run(cfg: dict, triage_result: dict) -> list[dict]:
    """
    Filter full components by priority and manual exclusions.

    Args:
        cfg:           result of get_machine_config()
        triage_result: result from step_b

    Returns:
        filtered list of full_component dicts
    """
    output_path = cfg["triage_step_c_full_components"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_c — filtering components...")

    triaged         = triage_result.get("triaged_components", [])
    full_components = [c for c in triaged if c["decision"] == "full_component"]
    pre_filter      = len(full_components)

    # Priority filter — drop low priority
    dropped = [c for c in full_components if c.get("priority") == "low"]
    kept    = [c for c in full_components if c.get("priority") != "low"]
    full_components = kept

    if dropped:
        print(f"    Dropped {len(dropped)} low-priority components:")
        for c in dropped:
            print(f"      ✗ {c['name']} — {c['reasoning']}")

    # Manual exclusions from machine's queries.py
    manual_exclusions = _load_manual_exclusions(cfg)
    if manual_exclusions:
        before = len(full_components)
        full_components = [c for c in full_components if c["name"] not in manual_exclusions]
        print(f"    Manually excluded: {manual_exclusions}")
        print(f"    Components: {before} → {len(full_components)}")

    print(f"\n    Full components: {pre_filter} → {len(full_components)} after filter")
    print(f"    Proceeding with:")
    for c in full_components:
        print(f"      [{c['priority']:6}] {c['name']}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_components, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return full_components


def _load_manual_exclusions(cfg: dict) -> list[str]:
    """Load manual_exclusions from the machine's queries.py if defined."""
    queries_path = cfg["machine_dir"] / "queries.py"

    if not queries_path.exists():
        return []

    spec   = importlib.util.spec_from_file_location("queries", queries_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, "manual_exclusions", [])