"""
phases/phase6_codegen/step_a_spec.py
--------------------------------------
Step A: Generate a ProgPy ready spec for each full component.

Takes the triage entry for each component and formalizes it into a precise
spec with states, events, parameters, and transition logic.

Each component spec is saved individually so you can rerun one without
redoing all of them.

Output saved to: outputs/codegen/specs/{component_name}.json
"""

import json, re

from utils.llm import call_claude
from utils.parsing import parse_json


COMPONENT_SPEC_PROMPT = """
You are creating a ProgPy component modeling spec for a physics-based digital twin.

You will receive a triage entry for one machine component. It contains:
- The component's role in the machine
- Its input and output ports (names and units already defined — do not change them)
- Candidate states derived from degradation mechanisms
- Candidate events derived from fault conditions

Your job is to formalize this into a precise ProgPy-ready spec.

---

STRICT PORT RULES:
- inputs and outputs are LOCKED. Copy them exactly from the triage entry.
  Do not add, remove, or rename any port. Do not change any unit.
- candidate_states are suggestions — you may refine descriptions and initial values
  but keep the names and units as given.
- candidate_events are suggestions — you may refine descriptions but keep names
  and fault_codes as given.

---

STATES:
Every state must have a state_type:

- "degradation": wear, fouling, loss of efficiency. Starts at 1.0, degrades to 0.0.
  min_value: 0.0, max_value: 1.0. Must have driver and degradation_timescale.

- "accumulation": quantity that builds up or drains. Must have capacity or limit.

- "tracking": physical quantity following a target via first-order lag.
  Must use stability-capped coefficient (max 0.5).

- "static": mode or flag, only changes on discrete user action.
  Do not include in state transition equations.

For every non-static state:
- min_value, max_value, degradation_timescale (if degradation), driver, initial_value

---

PARAMETERS:
- Every rate constant, threshold, and capacity must be a tunable parameter.
- Never hard-code magic numbers in logic.
- Use the LOCKED simulator rate for a state when one is provided below; only
  derive from degradation_timescale if no locked rate matches.

---

STATE TRANSITION LOGIC:
- Plain English equations referencing parameter names.
- Degradation: decrease by rate * driver_input * dt
- Accumulation: increase/decrease by rate * dt, clamped to capacity
- Tracking: approach target using first-order lag, coefficient capped at 0.5
- ALL non-static states must be clamped to [min_value, max_value]

---

OUTPUT LOGIC:
- Derive each output from states and inputs with a physically meaningful formula.
- If degradation reduces an output, state this explicitly.

---

EVENTS:
Every event must have:
- event_severity: "failure" | "warning" | "maintenance"
- threshold_states: list of state names ONLY — never input names.
  If a fault references an input value, add a tracking state for it instead.
- Under default parameters with healthy initial states, ALL event_state values
  must equal 1.0.

---

SIMULATOR GROUND TRUTH (AUTHORITATIVE — treat exactly like ports):
These degradation rates, thresholds, and drivers come from the validated phase-3
simulator that generated this machine's data. For each degradation state you MUST:
- use the matching rate constant as its parameter default (do NOT invent a rate),
- use the matching driver as its `driver` (do NOT infer a different one),
- use the matching critical_* / *_threshold value as its event threshold.
Match by semantic name. If a state has no match, derive from degradation_timescale
and say so in `assumptions`.

{ground_truth}

---

Return ONLY valid JSON:
{{
  "component_name": "string",
  "purpose": "string",
  "inputs": [{{"name": "string", "unit": "string", "description": "string"}}],
  "outputs": [{{"name": "string", "unit": "string", "description": "string"}}],
  "states": [
    {{
      "name": "string",
      "unit": "string",
      "state_type": "degradation | accumulation | tracking | static",
      "initial_value": 0.0,
      "min_value": "number or null",
      "max_value": "number or null",
      "degradation_timescale": "string or null",
      "driver": "string or null",
      "description": "string"
    }}
  ],
  "events": [
    {{
      "name": "string",
      "fault_code": "string or null",
      "event_severity": "failure | warning | maintenance",
      "threshold_states": ["string — state names only"],
      "description": "string"
    }}
  ],
  "parameters": [
    {{"name": "string", "description": "string", "default": 0.0, "unit": "string or null"}}
  ],
  "state_transition_logic": ["string"],
  "output_logic": ["string"],
  "event_logic": ["string"],
  "assumptions": ["string"]
}}

Return ONLY valid JSON.

Component triage entry:
{triage_entry}
"""

def _extract_simulator_constants(cfg) -> dict:
    """Pull rate/threshold/critical constants from the phase-3 simulator (LOCKED ground truth)."""
    sim_path = cfg["sim_step_c_code"]
    consts = {}
    if not sim_path.exists():
        return consts
    text = sim_path.read_text()
    for m in re.finditer(r"^\s*([a-zA-Z_]\w*)\s*(?::\s*[\w\[\], ]+)?\s*=\s*([-+0-9.eE]+)", text, re.M):
        name, val = m.group(1), m.group(2)
        if "rate" in name or name.startswith("critical_") or "threshold" in name:
            try:
                consts[name] = float(val)
            except ValueError:
                pass
    return consts


def _load_degradation_drivers(cfg) -> list:
    """Load the plan's degradation_model: state_variable -> driver (accumulates_from)."""
    plan_path = cfg["sim_step_b_plan"]
    if not plan_path.exists():
        return []
    with open(plan_path) as f:
        return json.load(f).get("degradation_model", [])


def _build_ground_truth_block(consts: dict, drivers: list) -> str:
    lines = []
    if drivers:
        lines.append("DEGRADATION DRIVERS (state <- what drives it):")
        for d in drivers:
            lines.append(f"- {d.get('state_variable')} <- {d.get('accumulates_from')}")
    if consts:
        lines.append("\nDEGRADATION RATES & CRITICAL THRESHOLDS:")
        for k, v in consts.items():
            lines.append(f"- {k} = {v}")
    return "\n".join(lines) if lines else "(no simulator ground truth found)"

def run(cfg: dict, full_components: list[dict]) -> dict:
    """
    Generate a ProgPy spec for each full component.

    Args:
        cfg:             result of get_machine_config()
        full_components: filtered list from phase 5 step_c

    Returns:
        dict of {component_name: spec_dict}
    """
    specs_dir = cfg["codegen_specs_dir"]
    specs_dir.mkdir(parents=True, exist_ok=True)

    all_specs    = {}
    already_done = 0
    ran          = 0

    print(f"  Running step_a — generating specs ({len(full_components)} components)...")

    ground_truth = _build_ground_truth_block(
          _extract_simulator_constants(cfg),
          _load_degradation_drivers(cfg),
      )
      
    for component in full_components:
        name = component["name"]
        path = specs_dir / f"{name}.json"

        # Load from disk if already done
        if path.exists():
            with open(path) as f:
                all_specs[name] = json.load(f)
            already_done += 1
            continue

        print(f"    [{ran + already_done + 1}/{len(full_components)}] {name}...", end=" ", flush=True)

        raw = call_claude(
            prompt=COMPONENT_SPEC_PROMPT.format(
                triage_entry=json.dumps(component, indent=2),
                ground_truth=ground_truth,
            ),
            max_tokens=8000,
            temperature=0.2,
        )

        spec = parse_json(raw)
        if not spec:
            print(f"✗ failed to parse — raw tail: ...{raw[-200:]!r}")
            continue

        # Validate ports match triage entry
        issues = _validate_ports(component, spec)
        if issues:
            print(f"⚠ port mismatch:")
            for issue in issues:
                print(f"      {issue}")

        # Save individually
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)

        all_specs[name] = spec
        ran += 1
        print(f"✓ ({len(spec.get('states', []))} states, {len(spec.get('events', []))} events)")

    if already_done > 0:
        print(f"    ✓ {already_done} specs loaded from disk, {ran} newly generated")
    print(f"    Total: {len(all_specs)} specs")

    return all_specs


def _validate_ports(triage_entry: dict, spec: dict) -> list[str]:
    """Check that spec ports match triage entry exactly."""
    issues = []

    triage_inputs  = {p["name"] for p in triage_entry.get("inputs", [])}
    triage_outputs = {p["name"] for p in triage_entry.get("outputs", [])}
    spec_inputs    = {p["name"] for p in spec.get("inputs", [])}
    spec_outputs   = {p["name"] for p in spec.get("outputs", [])}

    for name in triage_inputs - spec_inputs:
        issues.append(f"Input '{name}' missing from spec")
    for name in spec_inputs - triage_inputs:
        issues.append(f"Extra input '{name}' added in spec")
    for name in triage_outputs - spec_outputs:
        issues.append(f"Output '{name}' missing from spec")
    for name in spec_outputs - triage_outputs:
        issues.append(f"Extra output '{name}' added in spec")

    return issues