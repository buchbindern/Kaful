"""
phases/phase5_triage/step_b_triage.py
---------------------------------------
Step B: Triage components — decide how to model each one.

For each component decides:
  full_component   — has physics, gets a ProgPy model
  simple_state     — inventory/capacity only, no dynamics
  connection_only  — passive transfer, no model needed
  exclude          — UI, cosmetic, too vague

Optionally injects domain_context from the machine's queries.py to guide
prioritization toward the machine's most critical failure modes.

Output saved to: outputs/triage/step_b_triage.json
"""

import json

from utils.llm import call_claude
from utils.parsing import parse_json
from utils.helpers import load_domain_context


TRIAGE_PROMPT = """
You are deciding how to model each component of a machine in a ProgPy digital twin.

You have:
- A machine model document with each component's role, degradation mechanisms,
  fault conditions, and flow paths
- A field assignment document mapping schema fields to components —
  these are observable signals measurable by sensors
{domain_context_section}
For each component you must define TWO types of ports:

OBSERVABLE ports (source: "schema_field"):
- Take from the field assignments for this component
- These are signals that appear in real or synthetic telemetry
- Used for state estimation — the state estimation observes these

INTERNAL ports (source: "flow_path"):
- Take from flow_paths in the machine model doc
- boundary_signals where this component is the "to" endpoint → inputs
- boundary_signals where this component is the "from" endpoint → outputs
- These are physical signals flowing between components
- Used for composite model wiring — not directly observable

A component can have both types. Use signal_name, unit exactly as given in each source.
Do not invent port names not present in either source.

For every full_component:
- inputs:  observable inputs from schema assignments PLUS internal inputs from flow_paths
- outputs: observable outputs from schema assignments PLUS internal outputs from flow_paths
- candidate_states: from degradation_mechanisms in machine model
- candidate_events: from fault_conditions in machine model

IMPORTANT — outputs vs inputs for schema fields:
- Schema fields that are MEASURED or REPORTED by a component → outputs
- Schema fields that are SETTINGS or COMMANDS sent TO a component → inputs
- When in doubt, prefer outputs — the state estimation needs observable outputs

Criteria for full_component:
- Has degradation mechanisms, wear, fouling, or thermal/pressure dynamics
- Has clear inputs and outputs that influence machine performance
- Its failure mode matters for the machine's health

Criteria for simple_state:
- Mainly represents inventory, capacity, fill level, or availability
- No meaningful internal dynamics

Criteria for connection_only:
- Is primarily a pipe, tube, or passive transfer point

Criteria for exclude:
- UI, display, cosmetic, administrative, or too vague to model

Machine model document:
{machine_model}

Field assignments per component:
{component_fields}

Return ONLY valid JSON with EXACTLY this structure and EXACTLY these key names:
{{
  "triaged_components": [
    {{
      "name":     "string — exactly as in machine model doc",
      "decision": "full_component | simple_state | connection_only | exclude",
      "priority": "high | medium | low",
      "reasoning": "string",
      "inputs": [
        {{
          "name":          "string",
          "unit":          "string",
          "typical_value": "string or null",
          "source":        "schema_field | flow_path"
        }}
      ],
      "outputs": [
        {{
          "name":          "string",
          "unit":          "string",
          "typical_value": "string or null",
          "source":        "schema_field | flow_path"
        }}
      ],
      "candidate_states": [
        {{
          "name":          "string",
          "unit":          "string",
          "initial_value": 0.0,
          "description":   "string"
        }}
      ],
      "candidate_events": [
        {{
          "name":        "string",
          "fault_code":  "string or null",
          "description": "string"
        }}
      ]
    }}
  ],
  "simple_states": [
    {{
      "name":        "string",
      "unit":        "string or null",
      "description": "string"
    }}
  ],
  "modeling_notes": ["string"]
}}
Return ONLY valid JSON.
"""


def run(cfg: dict, model: dict, component_fields: dict) -> dict:
    """
    Triage all components.

    Args:
        cfg:              result of get_machine_config()
        model:            resolved machine model from phase 4
        component_fields: lookup from step_a

    Returns:
        triage result dict
    """
    output_path = cfg["triage_step_b_triage"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_b already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_b — triaging components...")

    # Load optional domain context
    domain_context = load_domain_context(cfg)
    if domain_context:
        domain_context_section = (
            f"\nDomain context about this machine:\n{domain_context.strip()}\n"
            f"Use this to prioritize components whose failure has the highest consequence.\n"
        )
        print(f"    Injecting domain context ({len(domain_context)} chars)")
    else:
        domain_context_section = ""

    raw = call_claude(
        prompt=TRIAGE_PROMPT.format(
            domain_context_section=domain_context_section,
            machine_model=json.dumps(model, indent=2),
            component_fields=json.dumps(component_fields, indent=2),
        ),
        max_tokens=8000,
        temperature=0.2,
    )

    result = parse_json(raw)
    if not result:
        raise ValueError(f"step_b: failed to parse triage result.\nRaw: {raw[:500]}")

    # Print summary
    triaged = result.get("triaged_components", [])
    full    = [c for c in triaged if c["decision"] == "full_component"]
    simple  = [c for c in triaged if c["decision"] == "simple_state"]
    conn    = [c for c in triaged if c["decision"] == "connection_only"]
    excl    = [c for c in triaged if c["decision"] == "exclude"]

    print(f"    full_component:  {len(full)}")
    for c in full:
        print(f"      [{c['priority']:6}] {c['name']:<30} "
              f"{len(c['inputs'])} in  {len(c['outputs'])} out  "
              f"{len(c['candidate_states'])} states  {len(c['candidate_events'])} events")
    print(f"    simple_state:    {len(simple)}")
    print(f"    connection_only: {len(conn)}")
    print(f"    exclude:         {len(excl)}")

    if result.get("modeling_notes"):
        print(f"    Notes:")
        for note in result["modeling_notes"]:
            print(f"      • {note}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return result