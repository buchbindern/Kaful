"""
phases/phase6_codegen/step_d_composite.py
-------------------------------------------
Step D: Build and validate the composite model.

Sub-steps:
  1. Propose connections from flow paths
  2. Resolve external inputs (typical_value → LLM fallback)
  3. Generate composite model code
  4. Validate composite model at t=0 and over 500 steps

Output saved to:
    outputs/codegen/step_d_connections.json
    outputs/codegen/step_d_external_inputs.json
    outputs/codegen/composite_model.py
    outputs/codegen/step_d_validation.json
"""

import json
import math
import importlib.util
import re
import sys

from utils.llm import call_claude
from utils.parsing import strip_fences, is_valid_python, parse_json
from utils.progpy_rag import get_framework_context


BATCH_NOMINAL_PROMPT = """
You are assigning nominal operating values for input ports of a machine digital twin.
Nominal means normal steady-state operation — not startup, not failure.

Machine type: {machine_type}

Ports needing values:
{ports}

Each entry has a name, unit, and typical_value if known.
Return a physically reasonable nominal float for each port in the given unit,
based on the machine type and what the port name implies physically.

Return ONLY valid JSON — a flat dict mapping port name to float:
{{"port_name": float}}
"""

COMPOSITE_CODE_PROMPT = """
You are generating Python code to assemble a ProgPy CompositeModel.

ProgPy reference:
{progpy_context}

Component files directory: {component_dir}

Actual component interfaces:
{interfaces}

Verified connections:
{connections}

External input nominal values:
{external_inputs}

Requirements:
1. Add the component directory to sys.path at the top.
2. Import each component class by its class_name from its file.
3. Instantiate each component.
4. Build the connections list using EXACTLY the verified connection tuples — do not modify.
5. Instantiate CompositeModel with the models dict and connections list.
   Assign to a module-level variable named composite_model.
6. Define future_loading_eqn(t, x=None) that:
   - Uses the external_inputs values above
   - Checks each key against composite_model.inputs before adding it
   - Returns composite_model.InputContainer with only valid keys
7. Do not perform any unit conversions.
8. Do not modify or re-derive any connection — use verified_connections exactly.

Return ONLY valid Python code. No markdown fences.
"""

COMPOSITE_QUERIES = [
    "CompositeModel instantiation models connections",
    "CompositeModel wiring outputs inputs tuple list",
    "composite model future loading function InputContainer",
    "CompositeModel simulate_to simulate_to_threshold",
]


def run(cfg: dict, full_components: list[dict], specs: dict,
        machine_model: dict) -> dict:
    """
    Build and validate the composite model.

    Args:
        cfg:             result of get_machine_config()
        full_components: filtered list from phase 5 step_c
        specs:           component specs from phase 6 step_a
        machine_model:   resolved machine model from phase 4

    Returns:
        dict with connections, external_inputs, composite_path, validation
    """
    output_path = cfg["codegen_step_d_validation"]

    if output_path.exists():
        print("  ✓ step_d already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_d — building composite model...")

    # Build actual interfaces from specs
    actual_interfaces = _build_interfaces(full_components, specs, cfg)

    # 1. Propose and verify connections
    connections = _propose_connections(cfg, machine_model, actual_interfaces)

    # 2. Resolve external inputs
    external_inputs = _resolve_external_inputs(
        cfg, connections, actual_interfaces, full_components, machine_model
    )

    # 3. Generate composite code
    composite_path = _generate_composite(
        cfg, actual_interfaces, connections, external_inputs
    )

    # 4. Validate
    validation = _validate_composite(cfg, composite_path, external_inputs)

    result = {
        "connections":      connections,
        "external_inputs":  external_inputs,
        "composite_path":   str(composite_path),
        "validation":       validation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved → {output_path.name}")
    return result


# ── Interfaces ────────────────────────────────────────────────────────────────

def _build_interfaces(full_components, specs, cfg):
    """Build actual_interfaces from component specs."""
    interfaces = {}
    for comp in full_components:
        name     = comp["name"]
        comp_key = f"{name}_component"
        spec     = specs.get(name, {})

        interfaces[comp_key] = {
            "class_name": _to_class_name(name),
            "file":       str(cfg["codegen_code_dir"] / f"{name}.py"),
            "inputs":  {i["name"]: {"unit": i["unit"]} for i in spec.get("inputs", [])},
            "outputs": {o["name"]: {"unit": o["unit"]} for o in spec.get("outputs", [])},
        }

    return interfaces


def _to_class_name(name: str) -> str:
    return "".join(w.capitalize() for w in name.split("_"))


# ── Connections ───────────────────────────────────────────────────────────────

def _propose_connections(cfg, machine_model, actual_interfaces):
    """Match flow path boundary signals to component ports."""
    output_path = cfg["codegen_step_d_connections"]

    if output_path.exists():
        print("  ✓ connections already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("    Proposing connections from flow paths...")

    proposed = []
    skipped  = []

    # Build port lookup
    output_ports = {}
    input_ports  = {}
    for comp_key, iface in actual_interfaces.items():
        for port in iface["outputs"]:
            output_ports[port] = comp_key
        for port in iface["inputs"]:
            input_ports[port] = comp_key

    for fp in machine_model.get("flow_paths", []):
        for sig in fp.get("boundary_signals", []):
            signal = sig["signal_name"]
            src    = output_ports.get(signal)
            tgt    = input_ports.get(signal)

            if src and tgt and src != tgt:
                conn = (f"{src}.{signal}", f"{tgt}.{signal}")
                if conn not in proposed:
                    proposed.append(conn)
                    print(f"      ✓ {src}.{signal} → {tgt}.{signal}")
            else:
                reason = "no source" if not src else "no target" if not tgt else "same component"
                skipped.append({"signal": signal, "from": sig.get("from"), "to": sig.get("to"), "reason": reason})

    print(f"    Verified: {len(proposed)} connections, {len(skipped)} skipped")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(proposed, f, indent=2)

    return proposed


# ── External inputs ───────────────────────────────────────────────────────────

def _resolve_external_inputs(cfg, connections, actual_interfaces,
                              full_components, machine_model):
    """Find unconnected input ports and assign nominal values."""
    output_path = cfg["codegen_step_d_external_inputs"]

    if output_path.exists():
        print("  ✓ external inputs already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("    Resolving external inputs...")

    connected_targets = {tgt for _, tgt in connections}

    # Find unconnected ports
    ports_needing_values = []
    for comp in full_components:
        name     = comp["name"]
        comp_key = f"{name}_component"
        for inp in comp.get("inputs", []):
            full_name = f"{comp_key}.{inp['name']}"
            if full_name not in connected_targets:
                ports_needing_values.append({
                    "name":          full_name,
                    "unit":          inp.get("unit", ""),
                    "typical_value": inp.get("typical_value"),
                })

    print(f"    Ports needing values: {len(ports_needing_values)}")

    external_inputs = {}

    # Pass 1: resolve from typical_value
    still_needed = []
    for p in ports_needing_values:
        mid = _parse_range_midpoint(p.get("typical_value"))
        if mid is not None:
            external_inputs[p["name"]] = mid
            print(f"      ✓ {p['name']}: {mid} {p['unit']} (from typical_value)")
        else:
            still_needed.append(p)
            print(f"      ? {p['name']}: needs LLM fallback")

    # Pass 2: LLM fallback
    if still_needed:
        raw = call_claude(
            prompt=BATCH_NOMINAL_PROMPT.format(
                machine_type=machine_model.get("machine_type", ""),
                ports=json.dumps(still_needed, indent=2),
            ),
            max_tokens=500,
            temperature=0.2,
        )
        llm_values = parse_json(raw) or {}
        for p in still_needed:
            name = p["name"]
            try:
                val = float(llm_values.get(name, 1.0))
                external_inputs[name] = val
                print(f"      ✓ {name}: {val} {p['unit']} (from LLM)")
            except (TypeError, ValueError):
                external_inputs[name] = 1.0
                print(f"      ⚠ {name}: 1.0 {p['unit']} (fallback)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(external_inputs, f, indent=2)

    return external_inputs


def _parse_range_midpoint(range_text):
    """Extract midpoint from a typical_range string."""
    if not isinstance(range_text, str) or not range_text.strip():
        return None
    nums = re.findall(r"\d+\.?\d*", range_text)  # only positive numbers
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return None


# ── Composite codegen ─────────────────────────────────────────────────────────

def _generate_composite(cfg, actual_interfaces, connections, external_inputs):
    """Generate the composite model Python file."""
    composite_path = cfg["codegen_composite_path"]

    if composite_path.exists():
        print("  ✓ composite code already done — loading from disk")
        return composite_path

    print("    Generating composite model code...")

    # Get ProgPy context for composite-specific queries
    framework_context = get_framework_context()

    raw = call_claude(
        prompt="Generate the composite model code.",
        system=COMPOSITE_CODE_PROMPT.format(
            progpy_context=framework_context,
            component_dir=str(cfg["codegen_code_dir"]),
            interfaces=json.dumps(actual_interfaces, indent=2),
            connections=json.dumps(connections, indent=2),
            external_inputs=json.dumps(external_inputs, indent=2),
        ),
        model="claude-opus-4-6",
        max_tokens=4000,
        temperature=0.2,
    )

    code = strip_fences(raw)
    if not is_valid_python(code):
        raise RuntimeError("Composite codegen produced invalid Python")

    composite_path.parent.mkdir(parents=True, exist_ok=True)
    composite_path.write_text(code)
    print(f"    ✓ Composite model written → {composite_path.name}")

    return composite_path


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_composite(cfg, composite_path, external_inputs):
    """Load and validate the composite model."""
    print("    Validating composite model...")

    # Clear cached modules
    for key in list(sys.modules.keys()):
        if any(n in key for n in ["composite", "ceramic", "brewing", "coffee",
                                   "steam", "water", "milk", "powder", "cleaning"]):
            del sys.modules[key]

    # Load composite
    spec_obj = importlib.util.spec_from_file_location("composite_model", composite_path)
    mod      = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)

    composite_model    = mod.composite_model
    future_loading_eqn = mod.future_loading_eqn

    print(f"    Components: {len(composite_model.states)} states, "
          f"{len(composite_model.inputs)} inputs, "
          f"{len(composite_model.outputs)} outputs, "
          f"{len(composite_model.events)} events")

    # t=0 check
    x0           = composite_model.initialize()
    event_states = composite_model.event_state(x0)
    triggered    = [k for k, v in event_states.items() if v < 1.0]

    if triggered:
        print(f"    ⚠ Events triggered at t=0: {triggered}")
    else:
        print(f"    ✓ All events healthy at t=0")

    # Step stability — 10 steps
    try:
        x = composite_model.initialize()
        for i in range(10):
            u = future_loading_eqn(i)
            x = composite_model.next_state(x, u, dt=1.0)
            for k in composite_model.states:
                v = x[k]
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    raise ValueError(f"state '{k}' is None/NaN at step {i}")
        print(f"    ✓ 10 steps stable")
    except Exception as e:
        print(f"    ✗ Step stability failed: {e}")

    # Degradation — 500 steps
    try:
        x0  = composite_model.initialize()
        x0d = dict(x0)
        x   = x0
        for i in range(500):
            u = future_loading_eqn(i)
            x = composite_model.next_state(x, u, dt=1.0)
        xfd = dict(x)

        changed = [(k, x0d[k], xfd[k]) for k in composite_model.states
                   if abs(xfd.get(k, 0) - x0d.get(k, 0)) > 1e-9]
        static  = [k for k in composite_model.states
                   if abs(xfd.get(k, 0) - x0d.get(k, 0)) <= 1e-9]

        print(f"    ✓ {len(changed)} states evolved over 500 steps")
        if static:
            print(f"    ⚠ {len(static)} states never changed: {static}")

    except Exception as e:
        print(f"    ✗ Degradation check failed: {e}")

    # Event progression — 5000 steps
    try:
        x = composite_model.initialize()
        for i in range(5000):
            u = future_loading_eqn(i)
            x = composite_model.next_state(x, u, dt=1.0)
        es        = composite_model.event_state(x)
        degrading = {k: v for k, v in es.items() if v < 1.0}

        if degrading:
            print(f"    ✓ {len(degrading)} events progressing after 5000 steps")
        else:
            print(f"    ⚠ No events changed after 5000 steps — rates may be too low")

    except Exception as e:
        print(f"    ✗ Event progression failed: {e}")

    return {
        "n_states":         len(composite_model.states),
        "n_inputs":         len(composite_model.inputs),
        "n_outputs":        len(composite_model.outputs),
        "n_events":         len(composite_model.events),
        "triggered_at_t0":  triggered,
    }