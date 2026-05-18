"""
phases/phase4_twin/step_d_validate.py
---------------------------------------
Step D: Validate the resolved machine model before triage.

Checks:
  ISSUES (must fix before continuing):
    - Components in pass1 missing from pass2
    - Orphan components in pass2 not in pass1
    - Signals missing units
    - Fault conditions missing descriptions
    - Flow path signals referencing unknown components

  WARNINGS (review before continuing):
    - Components with no operating ranges
    - Components with no degradation mechanisms
    - Components with no fault conditions
    - Signals with no typical_value or min/max
    - Degradation mechanisms with no driver
    - Flow path endpoints not in component list

Output saved to: outputs/twin/step_d_validation.json

Note: Runtime component testing (instantiate, run N steps, check bounds)
happens after codegen in phase 5.
"""

import json


def run(cfg: dict, model: dict) -> tuple[list, list]:
    """
    Validate the resolved machine model.

    Args:
        cfg:   result of get_machine_config()
        model: resolved machine model from step_c

    Returns:
        tuple of (issues, warnings)
        issues   — must fix before continuing
        warnings — review before continuing
    """
    output_path = cfg["twin_step_d_validation"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_d already done — loading from disk")
        with open(output_path) as f:
            saved = json.load(f)
        return saved["issues"], saved["warnings"]

    print("  Running step_d — validating machine model...")

    issues   = []
    warnings = []

    components       = model.get("components", [])
    component_physics = model.get("component_physics", {})
    flow_paths        = model.get("flow_paths", [])

    pass1_names = {c["name"] for c in components}
    pass2_names = set(component_physics.keys())

    # ── Cross-pass checks ─────────────────────────────────────────────────────

    for name in pass1_names - pass2_names:
        issues.append(f"Component '{name}' in pass1 but missing from pass2 physics")

    for name in pass2_names - pass1_names:
        issues.append(f"Component '{name}' in pass2 physics but not in pass1 (orphan)")

    # ── Per-component checks ──────────────────────────────────────────────────

    for comp in components:
        name    = comp["name"]
        physics = component_physics.get(name, {})
        pclass  = comp.get("physics_class", "full_component")

        # Skip simple_state components — they intentionally have no physics
        if pclass == "simple_state":
            continue

        # Operating ranges
        ranges = comp.get("operating_ranges", [])
        if not ranges:
            warnings.append(f"{name}: no operating ranges — nominal values will need LLM fallback")
        else:
            for r in ranges:
                if not r.get("unit"):
                    issues.append(f"{name}.{r['signal_name']}: missing unit")
                if r.get("typical_value") is None and r.get("min_value") is None:
                    warnings.append(f"{name}.{r['signal_name']}: no typical_value or min/max stated")

        # Degradation mechanisms
        degradation = comp.get("degradation_mechanisms", [])
        if not degradation:
            warnings.append(f"{name}: no degradation mechanisms — dx() will have no physics basis")
        else:
            for d in degradation:
                if not d.get("driven_by"):
                    warnings.append(f"{name}: degradation '{d.get('what_degrades', '?')}' has no driver")

        # Fault conditions
        faults = comp.get("fault_conditions", [])
        if not faults:
            warnings.append(f"{name}: no fault conditions — no failure events will be generated")
        else:
            for f in faults:
                if not f.get("description"):
                    issues.append(f"{name}: fault '{f.get('name', '?')}' has no description")

        # Maintenance actions
        maintenance = comp.get("maintenance_actions", [])
        if not maintenance:
            warnings.append(f"{name}: no maintenance actions — degradation will never be restored")

    # ── Flow path checks ──────────────────────────────────────────────────────

    for fp in flow_paths:
        substance = fp.get("substance", "?")
        for sig in fp.get("boundary_signals", []):
            signal_name = sig.get("signal_name", "?")

            # Missing unit
            if not sig.get("unit"):
                issues.append(
                    f"Flow path '{substance}' signal '{signal_name}': missing unit"
                )

            # Unknown endpoints
            for endpoint_key in ["from", "to"]:
                endpoint = sig.get(endpoint_key, "")
                if endpoint and endpoint != "external" and endpoint not in pass1_names:
                    warnings.append(
                        f"Flow path '{substance}': endpoint '{endpoint}' "
                        f"not in component list"
                    )

    # ── Coverage summary ──────────────────────────────────────────────────────

    full_components = [c for c in components if c.get("physics_class") != "simple_state"]
    with_ranges     = sum(1 for c in full_components if c.get("operating_ranges"))
    with_degrad     = sum(1 for c in full_components if c.get("degradation_mechanisms"))
    with_faults     = sum(1 for c in full_components if c.get("fault_conditions"))
    with_maint      = sum(1 for c in full_components if c.get("maintenance_actions"))
    n               = len(full_components)

    # ── Print report ──────────────────────────────────────────────────────────

    print(f"\n    {'='*52}")
    print(f"    Validation — {len(components)} components ({len(full_components)} full, "
          f"{len(components) - len(full_components)} simple_state)")
    print(f"    {'='*52}")

    print(f"\n    Coverage (full components only):")
    print(f"      Operating ranges:       {with_ranges}/{n}")
    print(f"      Degradation mechanisms: {with_degrad}/{n}")
    print(f"      Fault conditions:       {with_faults}/{n}")
    print(f"      Maintenance actions:    {with_maint}/{n}")

    if not issues and not warnings:
        print(f"\n    ✓ All checks passed")
    else:
        if issues:
            print(f"\n    Issues ({len(issues)}) — must fix before continuing:")
            for i in issues:
                print(f"      ✗ {i}")
        if warnings:
            print(f"\n    Warnings ({len(warnings)}) — review before continuing:")
            for w in warnings:
                print(f"      ⚠ {w}")

    print(f"\n    {len(issues)} issue(s)   {len(warnings)} warning(s)")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"issues": issues, "warnings": warnings}, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return issues, warnings