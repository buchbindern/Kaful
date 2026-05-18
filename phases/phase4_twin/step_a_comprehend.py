"""
phases/phase4_twin/step_a_comprehend.py
-----------------------------------------
Step A: Machine comprehension pass.

Retrieves manual chunks using comprehension queries, then sends them to Claude
to build a structured model of the machine — components, flow paths, operation
sequence, and operating envelope.

Output saved to: outputs/twin/step_a_comprehension.json
"""

import json
import importlib.util

from utils.llm import call_claude
from utils.parsing import parse_json
from utils.helpers import dedupe_chunks, chunks_to_context


MACHINE_COMPREHENSION_PROMPT = """
You are analyzing technical documentation to build a structured model of a machine
for use in a physics-based digital twin pipeline.
Your task is to extract the machine's identity, physical structure, process flow,
and operating envelope from the provided text.
This is a comprehension pass — focus on what the machine IS and how it WORKS.
Do not invent physics, thresholds, or degradation details — those come later.
Only include information directly supported by the provided text.
---
Return ONLY valid JSON with exactly this structure:
{{
  "machine_name": "string — from manual, or best inference if not explicit",
  "machine_type": "string — e.g. beverage dispenser, CNC lathe, infusion pump",
  "description": "string — one or two sentences on what the machine does",
  "components": [
    {{
      "name": "string — concise snake_case, e.g. brew_group, spindle_motor",
      "role": "string — what this component does in the machine",
      "category": "mechanical | thermal | hydraulic | electrical | pneumatic | control | structural | other",
      "evidence": "short quote or paraphrase from the text",
      "confidence": "high | medium | low"
    }}
  ],
  "flow_paths": [
    {{
      "substance": "string — what flows",
      "boundary_signals": [
        {{
          "from": "component_a",
          "to": "component_b",
          "signal_name": "string — descriptive name",
          "unit": "string — physical unit from manual, e.g. degC, bar, ml/s",
          "typical_value": "string — if stated in manual, else null"
        }}
      ],
      "evidence": "short quote or paraphrase from the text",
      "confidence": "high | medium | low"
    }}
  ],
  "operation_sequence": [
    {{
      "step": 1,
      "description": "string",
      "components_involved": ["string"],
      "evidence": "short quote or paraphrase from the text"
    }}
  ],
  "assumptions": [
    {{
      "statement": "string",
      "reason": "string — why this had to be assumed rather than read directly"
    }}
  ]
}}
---
Rules:
- component names must be snake_case and refer to physical or functional subsystems,
  not vague labels like system or module.
- flow_paths must follow the actual direction of flow described in the manual.
- boundary_signals should capture what crosses between two components.
- If a value is not stated, set typical_value to null.
- Do not add components, flows, or signals not supported by the text.
- Confidence: high = explicitly stated, medium = strongly implied, low = inferred.
- Return ONLY valid JSON. No markdown, no explanation.

Text:
{context}
"""


def run(cfg: dict, rag) -> dict:
    """
    Run the machine comprehension pass.

    Args:
        cfg: result of get_machine_config()
        rag: ManualRAG instance (already indexed)

    Returns:
        machine model dict with components, flow paths, operation sequence
    """
    output_path = cfg["twin_step_a_comprehension"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_a already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_a — machine comprehension...")

    # Load comprehension queries
    queries = _load_comprehension_queries(cfg)

    # Retrieve and dedupe chunks
    chunks  = rag.retrieve_chunks(queries, n_results_per_query=4)
    chunks  = dedupe_chunks(chunks)
    context = chunks_to_context(chunks)

    print(f"    Retrieved {len(chunks)} chunks")

    # Call Claude
    raw = call_claude(
        prompt=MACHINE_COMPREHENSION_PROMPT.format(context=context),
        max_tokens=8000,
        temperature=0.2,
    )

    model = parse_json(raw)
    if not model:
        raise ValueError(f"step_a: failed to parse machine model.\nRaw: {raw[:500]}")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"    Machine: {model.get('machine_name')} ({model.get('machine_type')})")
    print(f"    Components:  {len(model.get('components', []))}")
    print(f"    Flow paths:  {len(model.get('flow_paths', []))}")
    print(f"    Op sequence: {len(model.get('operation_sequence', []))} steps")
    print(f"    Saved → {output_path.name}")

    return model


def _load_comprehension_queries(cfg: dict) -> list[str]:
    """Load twin_comprehension_queries from the machine's queries.py."""
    queries_path = cfg["machine_dir"] / "queries.py"

    if not queries_path.exists():
        raise FileNotFoundError(f"No queries.py found at {queries_path}.")

    spec   = importlib.util.spec_from_file_location("queries", queries_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "twin_comprehension_queries"):
        raise AttributeError(
            f"{queries_path} must define a 'twin_comprehension_queries' list."
        )

    return module.twin_comprehension_queries