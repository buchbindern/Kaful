"""
phases/phase3_simulate/step_b_plan.py
---------------------------------------
Step B: Generate the simulator plan.

Sends the final schema + manual context to Claude to produce a structured
JSON specification describing how the simulator should work — subsystems,
state variables, degradation model, usage profiles, etc.

This plan is the source of truth for step_c codegen.

Output saved to: outputs/simulate/step_b_plan.json
"""

import json

from utils.llm import call_claude
from utils.parsing import parse_json


SIMULATOR_PLAN_PROMPT = """
You are designing a realistic event-level simulator for industrial equipment.

You are given:
1. A telemetry schema
2. Relevant manual context extracted from the equipment documentation

Your job is NOT to write code yet.
Your job is to design the simulator specification.

Return ONLY valid JSON with exactly this structure:

{
  "machine_type": "string",
  "machine_name": "string",
  "operation_types": [
    {
      "name": "string",
      "description": "string",
      "active_subsystems": ["string"]
    }
  ],
  "subsystems": [
    {
      "name": "string",
      "description": "string",
      "state_variables": ["string"]
    }
  ],
  "state_variables": [
    {
      "name": "string",
      "subsystem": "string",
      "type": "permanent_degradation|reversible_buildup|thermal|usage|other",
      "description": "string",
      "affects_fields": ["string"]
    }
  ],
  "usage_profiles": [
    {
      "name": "string",
      "description": "string",
      "operating_hours": "string",
      "relative_intensity": "low|medium|high",
      "operation_mix": {}
    }
  ],
  "field_generation_rules": [
    {
      "field_name": "string",
      "role": "control_input|measured_sensor|duration|outcome|metadata",
      "depends_on": ["string"],
      "generation_logic": "string"
    }
  ],
  "degradation_model": [
    {
      "state_variable": "string",
      "accumulates_from": "string",
      "restored_by": ["string"],
      "effects_on_fields": ["string"]
    }
  ],
  "maintenance_model": [
    {
      "event_type": "string",
      "trigger": "string",
      "effect_on_state": "string"
    }
  ],
  "failure_model": [
    {
      "failure_name": "string",
      "trigger_conditions": "string",
      "observable_effects": ["string"]
    }
  ],
  "event_timing_model": {
    "arrival_process": "string",
    "notes": "string"
  },
  "null_field_rules": [
    {
      "operation_type": "string",
      "fields_set_to_null": ["string"],
      "reason": "string"
    }
  ]
}

Requirements:
- Base everything only on the schema and manual context.
- Do not invent unsupported operation types or subsystems.
- Focus on realism for event-level telemetry simulation.
- If a field applies only to some operation types, capture that in null_field_rules.
- Return JSON only.
"""


def run(cfg: dict, schema: list[dict], manual_context: str) -> dict:
    """
    Generate the simulator plan from the schema and manual context.

    Args:
        cfg:            result of get_machine_config()
        schema:         final schema from phase 2
        manual_context: formatted context string from step_a

    Returns:
        simulator plan as a dict
    """
    output_path = cfg["sim_step_b_plan"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_b already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_b — generating simulator plan...")

    user_message = f"""Schema:
{json.dumps(schema, indent=2)}

Manual context:
{manual_context}"""

    raw = call_claude(
        prompt=user_message,
        system=SIMULATOR_PLAN_PROMPT,
        max_tokens=8000,
        temperature=0.2,
    )

    plan = parse_json(raw)
    if not plan:
        raise ValueError(f"step_b: failed to parse simulator plan.\nRaw: {raw[:500]}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"    Machine: {plan.get('machine_name')} ({plan.get('machine_type')})")
    print(f"    Operation types: {len(plan.get('operation_types', []))}")
    print(f"    Subsystems:      {len(plan.get('subsystems', []))}")
    print(f"    State variables: {len(plan.get('state_variables', []))}")
    print(f"    Usage profiles:  {len(plan.get('usage_profiles', []))}")
    print(f"    Saved → {output_path.name}")

    return plan