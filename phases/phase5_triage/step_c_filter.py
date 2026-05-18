"""
phases/phase5_triage/step_c_filter.py
---------------------------------------
Step C: Filter full components by priority and manual overrides.

- Drops low-priority components unless manually included
- Applies manual_exclusions from queries.py
- Applies manual_inclusions from queries.py — forces components to
  full_component regardless of what triage decided

Output saved to: outputs/triage/step_c_full_components.json
"""

import json
import importlib.util


def run(cfg: dict, triage_result: dict) -> list[dict]:
    """
    Filter full components by priority and manual overrides.

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

    manual_exclusions, manual_inclusions = _load_manual_overrides(cfg)

    triaged         = triage_result.get("triaged_components", [])
    full_components = [c for c in triaged if c["decision"] == "full_component"]
    pre_filter      = len(full_components)

    # Apply manual inclusions — force non-full components into full_component
    if manual_inclusions:
        all_components = {c["name"]: c for c in triaged}
        for name in manual_inclusions:
            if name not in {c["name"] for c in full_components}:
                if name in all_components:
                    comp = all_components[name].copy()
                    comp["decision"]  = "full_component"
                    comp["priority"]  = "high"
                    comp["reasoning"] = comp.get("reasoning", "") + " [manually included]"
                    full_components.append(comp)
                    print(f"    ✓ Manually included: {name}")
                else:
                    print(f"    ⚠ Manual inclusion '{name}' not found in triage — skipping")

    # Priority filter — drop low priority unless manually included
    dropped = [c for c in full_components
               if c.get("priority") == "low" and c["name"] not in manual_inclusions]
    kept    = [c for c in full_components
               if c.get("priority") != "low" or c["name"] in manual_inclusions]
    full_components = kept

    if dropped:
        print(f"    Dropped {len(dropped)} low-priority components:")
        for c in dropped:
            print(f"      ✗ {c['name']} — {c['reasoning'][:60]}")

    # Apply manual exclusions
    if manual_exclusions:
        before = len(full_components)
        full_components = [c for c in full_components
                           if c["name"] not in manual_exclusions]
        removed = before - len(full_components)
        if removed:
            print(f"    Manually excluded: {manual_exclusions}")

    print(f"\n    Full components: {pre_filter} → {len(full_components)} after filter")
    print(f"    Proceeding with:")
    for c in full_components:
        tag = " [manual]" if c["name"] in manual_inclusions else ""
        print(f"      [{c['priority']:6}] {c['name']}{tag}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_components, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return full_components


def _load_manual_overrides(cfg: dict) -> tuple[list[str], list[str]]:
    """Load manual_exclusions and manual_inclusions from the machine's queries.py."""
    queries_path = cfg["machine_dir"] / "queries.py"

    if not queries_path.exists():
        return [], []

    spec   = importlib.util.spec_from_file_location("queries", queries_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    exclusions = getattr(module, "manual_exclusions", [])
    inclusions = getattr(module, "manual_inclusions", [])

    return exclusions, inclusions