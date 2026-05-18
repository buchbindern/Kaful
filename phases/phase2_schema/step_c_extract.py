"""
phases/phase2_schema/step_c_extract.py
---------------------------------------
Step C: Extract candidate schema fields N times.

Runs the candidate fields prompt N_SCHEMA_RUNS times against the manual context.
Multiple runs increase coverage — different runs surface different fields,
which step_d and step_e will normalize and merge.

Output saved to: outputs/schema/step_c_runs.json
"""

import json

#from config import N_SCHEMA_RUNS
from utils.llm import call_claude

N_SCHEMA_RUNS = 5

CANDIDATE_FIELDS_PROMPT_TEMPLATE = """
You are extracting candidate event-level fields for a machine event schema.

Machine type: {machine_type}
Machine name: {machine_name}
Core event type: {event_type}
One event is defined as: {event_description}

Extract a broad but precise list of fields relevant to a single event from trigger to completion.
Include fields only if they are one of:
- an event timing field
- a control or input setting for the event
- a measured quantity during the event
- a duration of an event phase
- an event outcome or result

Prioritize these event-level categories when supported by the text:
{measurement_categories}

Exclude:
{exclusions}

Include only fields explicitly supported by the provided text, or clearly parameterized,
measured, logged, selected, or reported as part of one event.
Do not invent unsupported fields.

For each field output:
- field_name
- category: event_time | control_input | sensor_measurement | phase_duration | event_outcome
- data_type: decimal | integer | boolean | string | timestamp
- unit
- typical_range
- description
- source_pages
- evidence: short quote or paraphrase from the provided text

Requirements:
- Focus on one {event_type}
- Keep names lowercase_with_underscores
- Prefer compact, operationally useful fields over admin/UI metadata
- Use explicit units from the text when available, such as: {unit_examples}
- Output ONLY a valid JSON array
"""


def run(cfg: dict, context: str, context_info: dict) -> list[str]:
    """
    Extract candidate schema fields N times.

    Args:
        cfg:          result of get_machine_config()
        context:      formatted context string from step_a
        context_info: machine understanding dict from step_b

    Returns:
        list of N raw JSON strings, one per run
    """
    output_path = cfg["step_c_runs"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print(f"  Running step_c — extracting fields ({N_SCHEMA_RUNS} runs)...")

    prompt = _build_prompt(context_info)

    all_runs = []
    for i in range(N_SCHEMA_RUNS):
        print(f"    Run {i+1}/{N_SCHEMA_RUNS}...", end=" ", flush=True)

        raw = call_claude(
            prompt=f"{context}\n\n{prompt}",
            max_tokens=8000,
        )

        all_runs.append(raw)
        print("✓")

    # Save all raw runs to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return all_runs


def _build_prompt(context_info: dict) -> str:
    """Build the candidate fields prompt from context_info."""
    measurement_categories = context_info.get("measurement_categories", [])
    exclusions             = context_info.get("exclusions", [])
    unit_examples          = context_info.get("unit_examples", [])

    return CANDIDATE_FIELDS_PROMPT_TEMPLATE.format(
        machine_type=context_info.get("machine_type", ""),
        machine_name=context_info.get("machine_name", ""),
        event_type=context_info.get("event_type", "event"),
        event_description=context_info.get("event_description", ""),
        measurement_categories="\n".join(f"- {c}" for c in measurement_categories),
        exclusions="\n".join(f"- {e}" for e in exclusions),
        unit_examples=", ".join(unit_examples) if unit_examples else "none specified",
    )