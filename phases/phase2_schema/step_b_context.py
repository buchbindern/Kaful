"""
phases/phase2_schema/step_b_context.py
---------------------------------------
Step B: Infer extraction context from retrieved chunks.

Makes two Claude calls:
1. EVENT_CONTEXT  — what is the main operational event this machine produces?
2. EXTRACTION_SCOPE — what categories of fields are worth extracting?

Merges both into a single context_info dict used by step_c to guide
schema field extraction.

Output saved to: outputs/schema/step_b_context.json
"""

import json

from utils.llm import call_claude
from utils.parsing import parse_json


EVENT_CONTEXT_PROMPT = """
You are analyzing technical documentation.
Identify the main operational event described in the text.
Return ONLY a valid JSON object with exactly these keys:
{
  "machine_type": "string",
  "machine_name": "string",
  "event_type": "string",
  "event_description": "string"
}
Guidelines:
- machine_type: generic machine or system type
- machine_name: specific product/model if present, else ""
- event_type: one unit of operation from trigger to completion
- event_description: one concise sentence defining that event

Requirements:
- Focus on the main operational event, not installation, admin, or maintenance tasks
- Prefer event-level context over machine-level metadata
- Do not invent unsupported details
- Return ONLY valid JSON
"""

EXTRACTION_SCOPE_PROMPT = """
You are preparing event-level field extraction instructions from technical documentation.
Given the text, identify:
- the major categories of event-level fields worth extracting
- the types of fields that should be excluded
- common units explicitly mentioned in the text

Return ONLY a valid JSON object with exactly these keys:
{
  "measurement_categories": ["string"],
  "exclusions": ["string"],
  "unit_examples": ["string"]
}

Guidelines:
- measurement_categories: short phrases describing event-level field groups relevant to one event
- exclusions: field types that should not be extracted because they are not part of one event
- unit_examples: units explicitly present in the text, if any

Requirements:
- Focus on one event from trigger to completion
- Include only categories supported by the text
- Exclude installation, admin, firmware, user-role, and unrelated maintenance/setup information
  unless directly part of one event
- Return ONLY valid JSON
"""


def run(cfg: dict, context: str) -> dict:
    """
    Infer extraction context from the retrieved chunk context string.

    Args:
        cfg:     result of get_machine_config()
        context: formatted context string from step_a (chunks_to_context output)

    Returns:
        dict with merged event context and extraction scope
    """
    output_path = cfg["step_b_context"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_b already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_b — inferring extraction context...")

    # Call 1 — event context
    raw_event = call_claude(
        prompt=f"{context}\n\n{EVENT_CONTEXT_PROMPT}",
        max_tokens=1000,
    )
    event_context = parse_json(raw_event)
    if not event_context:
        raise ValueError(f"step_b: failed to parse event context.\nRaw: {raw_event[:500]}")

    # Call 2 — extraction scope
    raw_scope = call_claude(
        prompt=f"{context}\n\n{EXTRACTION_SCOPE_PROMPT}",
        max_tokens=1000,
    )
    scope_context = parse_json(raw_scope)
    if not scope_context:
        raise ValueError(f"step_b: failed to parse extraction scope.\nRaw: {raw_scope[:500]}")

    # Merge both into one context_info dict
    context_info = {**event_context, **scope_context}

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(context_info, f, indent=2)

    print(f"    Machine: {context_info.get('machine_name')} ({context_info.get('machine_type')})")
    print(f"    Event: {context_info.get('event_type')}")
    print(f"    Categories: {context_info.get('measurement_categories')}")
    print(f"    Saved → {output_path.name}")

    return context_info