"""
phases/phase4_twin/step_c_resolve.py
--------------------------------------
Step C: Merge physics, resolve missing units, flag no-physics components.

1. Merges component_physics back into the machine model components
2. Finds signals with missing units and resolves them via one RAG + LLM call
3. Flags components with no operating_ranges and no degradation_mechanisms
   as 'simple_state' — they won't get a ProgPy model

Output saved to: outputs/twin/step_c_machine_model.json
"""

import json

from utils.llm import call_claude
from utils.parsing import parse_json
from utils.helpers import dedupe_chunks, chunks_to_context


UNIT_RESOLUTION_PROMPT = """
You are resolving missing units for signals extracted from a machine manual.

For each signal below, determine the most physically appropriate unit based on:
- The signal name
- The component it belongs to
- The machine type: {machine_type}
- Any relevant manual context provided

If the unit is genuinely dimensionless (e.g. a 0-100 scale, a boolean, a count),
state "dimensionless" and explain briefly.
If the unit cannot be determined from context, state null.

Return ONLY valid JSON — a dict mapping signal_name to:
{{
  "unit": "string or null",
  "reasoning": "string",
  "confidence": "high | medium | low"
}}

Signals needing units:
{signals}

Relevant manual context:
{context}
"""


def run(cfg: dict, rag, machine_model: dict, component_physics: dict) -> dict:
    """
    Merge physics, resolve units, flag no-physics components.

    Args:
        cfg:               result of get_machine_config()
        rag:               ManualRAG instance
        machine_model:     result from step_a
        component_physics: result from step_b

    Returns:
        resolved machine model dict
    """
    output_path = cfg["twin_step_c_model"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_c — merging physics and resolving units...")

    # 1. Merge component physics into machine model
    model = _merge_physics(machine_model, component_physics)

    # 2. Collect missing units
    missing_units = _collect_missing_units(model)
    print(f"    Signals needing unit resolution: {len(missing_units)}")

    if missing_units:
        # One targeted RAG query + LLM call
        resolution_queries = [
            f"{s['signal_name'].replace('_', ' ')} unit measurement specification"
            for s in missing_units
        ]
        chunks  = rag.retrieve_chunks(resolution_queries, n_results_per_query=3)
        chunks  = dedupe_chunks(chunks)
        context = chunks_to_context(chunks)

        raw = call_claude(
            prompt=UNIT_RESOLUTION_PROMPT.format(
                machine_type=model.get("machine_type", ""),
                signals=json.dumps(missing_units, indent=2),
                context=context,
            ),
            max_tokens=2000,
            temperature=0.2,
        )

        resolved = parse_json(raw) or {}

        # Write resolved units back into model
        model = _apply_resolved_units(model, resolved)

        print(f"    Resolved units:")
        for signal_name, result in resolved.items():
            icon = "✓" if result.get("unit") else "✗"
            print(f"      {icon} {signal_name}: {result.get('unit')} ({result.get('confidence')}) — {result.get('reasoning', '')[:60]}")

    # 3. Flag no-physics components
    no_physics = _flag_no_physics(model)
    if no_physics:
        print(f"    Components flagged as simple_state ({len(no_physics)}):")
        for name in no_physics:
            print(f"      {name}")
    else:
        print(f"    ✓ All components have extractable physics")

    # 4. Save component_physics into model for downstream use
    model["component_physics"] = component_physics

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _merge_physics(machine_model: dict, component_physics: dict) -> dict:
    """Merge component physics back into machine model components."""
    model = machine_model.copy()

    for comp in model["components"]:
        name = comp["name"]
        if name in component_physics:
            comp.update(component_physics[name])

    return model


def _collect_missing_units(model: dict) -> list[dict]:
    """Find all signals with missing units across components and flow paths."""
    missing = []

    for comp in model.get("components", []):
        for r in comp.get("operating_ranges", []):
            if not r.get("unit"):
                missing.append({
                    "component":     comp["name"],
                    "signal_name":   r["signal_name"],
                    "typical_value": r.get("typical_value"),
                })

    for fp in model.get("flow_paths", []):
        for sig in fp.get("boundary_signals", []):
            if not sig.get("unit"):
                missing.append({
                    "component":     f"flow_path ({fp['substance']})",
                    "signal_name":   sig["signal_name"],
                    "typical_value": sig.get("typical_value"),
                })

    return missing


def _apply_resolved_units(model: dict, resolved: dict) -> dict:
    """Write resolved units back into the model."""
    for comp in model.get("components", []):
        for r in comp.get("operating_ranges", []):
            if not r.get("unit") and r["signal_name"] in resolved:
                r["unit"]             = resolved[r["signal_name"]]["unit"]
                r["unit_confidence"]  = resolved[r["signal_name"]]["confidence"]

    for fp in model.get("flow_paths", []):
        for sig in fp.get("boundary_signals", []):
            if not sig.get("unit") and sig["signal_name"] in resolved:
                sig["unit"]            = resolved[sig["signal_name"]]["unit"]
                sig["unit_confidence"] = resolved[sig["signal_name"]]["confidence"]

    return model


def _flag_no_physics(model: dict) -> list[str]:
    """Flag components with no physics as simple_state."""
    no_physics = []

    for comp in model.get("components", []):
        has_ranges      = bool(comp.get("operating_ranges"))
        has_degradation = bool(comp.get("degradation_mechanisms"))

        if not has_ranges and not has_degradation:
            comp["physics_class"] = "simple_state"
            no_physics.append(comp["name"])
        else:
            comp.setdefault("physics_class", "full_component")

    return no_physics