"""
phases/phase5_triage/step_a_assign.py
---------------------------------------
Step A: Assign schema fields to machine components.

Maps each field from the final schema to the component it belongs to,
and classifies it as an input or output signal.

Output saved to: outputs/triage/step_a_field_assignments.json
"""

import json

from utils.llm import call_claude
from utils.parsing import parse_json


FIELD_ASSIGNMENT_PROMPT = """
You are assigning schema fields to machine components for a digital twin.

You have a list of schema fields extracted from machine telemetry —
each field represents a measurable signal on the machine.

You have a list of components extracted from the machine manual.

For each schema field, identify which component it belongs to.
Base your assignment on the field name, unit, description, and
what you know about each component's role.

A field can only belong to ONE component.
If a field is clearly a machine-level measurement not tied to one component,
assign it to the most relevant component.
If a field cannot be assigned to any component, mark it unassigned.

Machine components:
{components}

Schema fields:
{schema_fields}

Return ONLY valid JSON:
{{
  "assignments": [
    {{
      "field_name":    "string — normalized_name from schema",
      "unit":          "string — from schema",
      "typical_range": "string — from schema",
      "component":     "string — component name or null if unassigned",
      "signal_type":   "input | output",
      "reasoning":     "string"
    }}
  ],
  "unassigned": ["string — field names that could not be assigned"]
}}

Signal type rules:
- output: something the component produces or reports — temperature, flow rate,
  pressure reading, wear indicator, quality metric
- input: something that drives or configures the component — setpoint, command,
  supply condition, environmental factor
"""


def run(cfg: dict, schema: list[dict], model: dict) -> tuple[dict, dict]:
    """
    Assign schema fields to components.

    Args:
        cfg:    result of get_machine_config()
        schema: final schema from phase 2
        model:  resolved machine model from phase 4

    Returns:
        tuple of (field_assignment_result, component_fields lookup)
    """
    output_path = cfg["triage_step_a_assignments"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_a already done — loading from disk")
        with open(output_path) as f:
            result = json.load(f)
        return result, _build_component_fields(result)

    print("  Running step_a — assigning fields to components...")

    # Build compact representations for the prompt
    schema_for_assignment = [
        {
            "name":          f["normalized_name"],
            "unit":          f["unit"],
            "typical_range": f["typical_range"],
            "description":   f["description"],
            "category":      f["category"],
        }
        for f in schema
    ]

    components_for_assignment = [
        {
            "name":     c["name"],
            "role":     c["role"],
            "category": c["category"],
        }
        for c in model["components"]
    ]

    raw = call_claude(
        prompt=FIELD_ASSIGNMENT_PROMPT.format(
            components=json.dumps(components_for_assignment, indent=2),
            schema_fields=json.dumps(schema_for_assignment, indent=2),
        ),
        max_tokens=4000,
        temperature=0.2,
    )

    result = parse_json(raw)
    if not result:
        raise ValueError(f"step_a: failed to parse field assignments.\nRaw: {raw[:500]}")

    assignments = result.get("assignments", [])
    unassigned  = result.get("unassigned", [])

    # Build component_fields lookup
    component_fields = _build_component_fields(result)

    # Print summary
    print(f"    Assigned: {len(assignments) - len(unassigned)}/{len(assignments)} fields")
    for comp, fields in component_fields.items():
        ins  = fields["inputs"]
        outs = fields["outputs"]
        print(f"    {comp}: {len(ins)} inputs, {len(outs)} outputs")

    if unassigned:
        print(f"    Unassigned ({len(unassigned)}): {unassigned}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return result, component_fields


def _build_component_fields(result: dict) -> dict:
    """Build lookup: component_name → {inputs: [...], outputs: [...]}"""
    component_fields = {}

    for a in result.get("assignments", []):
        comp = a.get("component")
        if not comp:
            continue

        if comp not in component_fields:
            component_fields[comp] = {"inputs": [], "outputs": []}

        component_fields[comp][a["signal_type"] + "s"].append({
            "name":          a["field_name"],
            "unit":          a["unit"],
            "typical_range": a["typical_range"],
        })

    return component_fields