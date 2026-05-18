"""
phases/phase4_twin/step_b_physics.py
--------------------------------------
Step B: Extract physics details for each component.

Runs COMPONENT_PHYSICS_PROMPT once per component, retrieving relevant manual
chunks for each one. Each component is saved individually so you can rerun
a single component without redoing all of them.

Output saved to: outputs/twin/components/{component_name}.json
"""

import json
import importlib.util

from utils.llm import call_claude
from utils.parsing import parse_json
from utils.helpers import dedupe_chunks, chunks_to_context


COMPONENT_PHYSICS_PROMPT = """
You are extracting physics details for one specific component of a machine,
for use in a physics-based digital twin.
You will receive:
- The component name and role
- Manual excerpts relevant to this component

Extract only what is directly supported by the text.
Do not invent numbers, mechanisms, or fault codes.

Return ONLY valid JSON with exactly this structure:
{{
  "name": "string — exactly as provided",
  "operating_ranges": [
    {{
      "signal_name": "string — descriptive, snake_case",
      "unit": "string — exactly as written in manual",
      "typical_value": "string or null",
      "min_value": "string or null",
      "max_value": "string or null",
      "evidence": "short quote or paraphrase"
    }}
  ],
  "degradation_mechanisms": [
    {{
      "what_degrades": "string",
      "driven_by": "string",
      "rate_description": "string or null",
      "evidence": "short quote or paraphrase"
    }}
  ],
  "fault_conditions": [
    {{
      "name": "string — concise snake_case",
      "fault_code": "string or null",
      "description": "string",
      "symptoms": "string or null",
      "evidence": "short quote or paraphrase"
    }}
  ],
  "maintenance_actions": [
    {{
      "name": "string",
      "interval": "string or null",
      "what_it_resets": "string",
      "evidence": "short quote or paraphrase"
    }}
  ],
  "assumptions": [
    {{
      "statement": "string",
      "reason": "string"
    }}
  ]
}}

Component name: {name}
Component role: {role}
Manual excerpts:
{context}
"""


def run(cfg: dict, rag, machine_model: dict) -> dict:
    """
    Extract physics for each component in the machine model.

    Args:
        cfg:           result of get_machine_config()
        rag:           ManualRAG instance (already indexed)
        machine_model: result from step_a

    Returns:
        dict of {component_name: physics_dict}
    """
    components_dir = cfg["twin_components_dir"]
    components_dir.mkdir(parents=True, exist_ok=True)

    components    = machine_model.get("components", [])
    all_physics   = {}
    already_done  = 0
    ran           = 0

    print(f"  Running step_b — component physics ({len(components)} components)...")

    for component in components:
        name = component["name"]
        role = component["role"]
        path = components_dir / f"{name}.json"

        # Load from disk if already done
        if path.exists():
            with open(path) as f:
                all_physics[name] = json.load(f)
            already_done += 1
            continue

        print(f"    [{ran + already_done + 1}/{len(components)}] {name}...", end=" ", flush=True)

        # Retrieve chunks relevant to this component
        queries = _build_component_queries(name, role)
        chunks  = rag.retrieve_chunks(queries, n_results_per_query=3)
        chunks  = dedupe_chunks(chunks)
        context = chunks_to_context(chunks)

        # Call Claude
        raw = call_claude(
            prompt=COMPONENT_PHYSICS_PROMPT.format(
                name=name,
                role=role,
                context=context,
            ),
            max_tokens=4000,
            temperature=0.2,
        )

        physics = parse_json(raw)
        if not physics:
            print(f"✗ failed to parse — skipping")
            continue

        # Save individually
        with open(path, "w") as f:
            json.dump(physics, f, indent=2)

        all_physics[name] = physics
        ran += 1
        print("✓")

    if already_done > 0:
        print(f"    ✓ {already_done} components loaded from disk, {ran} newly generated")
    print(f"    Total: {len(all_physics)} components with physics")

    return all_physics


def _build_component_queries(name: str, role: str) -> list[str]:
    """Build RAG queries for a specific component."""
    # Use the component name (with spaces) and role as queries
    name_readable = name.replace("_", " ")
    return [
        name_readable,
        f"{name_readable} specifications operating range",
        f"{name_readable} maintenance cleaning fault error",
    ]