"""
phases/phase2_component_map/step_b_component_map.py
-----------------------------------------------------
Step b: LLM maps each signal channel to a physical component.

We pass the LLM the real channel list (force_x, vibration_z, ae_rms)
derived from cuts.parquet — not the feature list. After the LLM responds,
reconcile.py snaps all channel references back to the ground-truth set:
phantom names are dropped with fuzzy-match suggestions, unassigned real
channels get a low-relevance placeholder.

This means the output is always self-consistent regardless of what the
LLM invented — no phantom channel names leak to phase 3 or 4.

Output: step_b_component_map.json
{
  "components": [
    {
      "name":     "cutting_tool",
      "role":     "primary degrading component",
      "channels": ["force_z", "ae_rms", "vibration_x", ...],  ← only real channels
      ...
    }
  ],
  "channel_assignments": {
    "force_z": { "degradation_relevance": "high", ... },      ← every real channel
    ...
  }
}
"""

import json
from utils.llm import call_claude
from utils.parsing import parse_json
from phases.phase2_component_map.reconcile import reconcile_channels


SYSTEM_PROMPT = """You are a mechanical engineering expert specialising in
predictive maintenance and digital twin construction for industrial machines.
You map sensor signals to physical machine components and identify which
signals are most relevant to degradation monitoring."""


COMPONENT_MAP_PROMPT = """
You are building a physics-based digital twin for condition monitoring.

Given the sensor channels below, identify:
1. The physical components of this machine
2. Which channels each component exposes
3. The degradation relevance of each channel

{dataset_context}

Signal channels available (use ONLY these exact names — do not invent others):
{channels_json}

Return ONLY valid JSON in exactly this structure:
{{
  "components": [
    {{
      "name":        "snake_case_component_name",
      "description": "one sentence description of this component",
      "role":        "primary degrading component | support component | environmental",
      "channels":    ["channel_name", ...],
      "reasoning":   "why these channels belong to this component"
    }}
  ],
  "channel_assignments": {{
    "channel_name": {{
      "component":             "component name from above",
      "observable":            "what physical quantity this signal measures",
      "degradation_relevance": "high | medium | low",
      "reasoning":             "why this signal is or isn't diagnostic of degradation"
    }}
  }}
}}

Rules:
- Use ONLY the channel names provided above — no variations, no derived names
- Every channel must appear in exactly one component's channels list
- Every channel must appear in channel_assignments
- Component names must be snake_case
- degradation_relevance=high: signal directly tracks wear/fatigue/damage
- degradation_relevance=medium: indirect indicator
- degradation_relevance=low: process input or environmental condition
"""


def run(cfg: dict, signals: list[dict]) -> dict:
    output_path = cfg["step_b_map_path"]

    if output_path.exists():
        print("  ✓ step_b already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_b — mapping channels to components (LLM)...")

    # ── Build ground-truth channel set ────────────────────────────────────────
    # Derive unique channels from the signals list — this is the only
    # source of truth. The LLM must use these names exactly.
    seen_channels: dict[str, dict] = {}
    for sig in signals:
        ch = sig["channel"]
        if ch not in seen_channels:
            seen_channels[ch] = {
                "name":         ch,
                "unit":         sig["unit"],
                "channel_type": sig["channel_type"],
                "axis":         sig["axis"],
            }

    channels_list  = list(seen_channels.values())
    valid_channels = set(seen_channels.keys())

    # ── Build context block ───────────────────────────────────────────────────
    queries     = cfg.get("queries", {})
    dataset_ctx = queries.get("DATASET_CONTEXT", "")
    meta        = cfg["meta"]

    if meta is not None and not dataset_ctx:
        parts = []
        for attr in ("MACHINE_TYPE", "WORKPIECE_MATERIAL", "TOOL_MATERIAL", "TOOL_GEOMETRY"):
            val = getattr(meta, attr, None)
            if val:
                parts.append(f"{attr.replace('_', ' ').title()}: {val}")
        dataset_ctx = "\n".join(parts)

    context_block = f"Dataset context:\n{dataset_ctx}\n" if dataset_ctx else ""

    # ── LLM call ──────────────────────────────────────────────────────────────
    prompt = COMPONENT_MAP_PROMPT.format(
        dataset_context=context_block,
        channels_json=json.dumps(channels_list, indent=2),
    )

    raw    = call_claude(prompt, system=SYSTEM_PROMPT, max_tokens=4000, temperature=0.2)
    result = parse_json(raw)

    if not result:
        raise ValueError(f"step_b: failed to parse component map.\nRaw:\n{raw[:600]}")

    # ── Reconcile: snap all channel names back to ground truth ────────────────
    result, warnings = reconcile_channels(result, valid_channels)

    for w in warnings:
        print(f"    ⚠ {w}")

    # ── Print summary ─────────────────────────────────────────────────────────
    components = result.get("components", [])
    assignments = result.get("channel_assignments", {})
    print(f"    {len(components)} components identified:")
    for comp in components:
        high_deg = [
            ch for ch in comp.get("channels", [])
            if assignments.get(ch, {}).get("degradation_relevance") == "high"
        ]
        print(f"      {comp['name']} ({comp['role']}) — "
              f"{len(comp['channels'])} channels, {len(high_deg)} high-degradation")

    # Confirm all real channels are accounted for
    assigned = set(assignments.keys())
    missing  = valid_channels - assigned
    if missing:
        print(f"    ⚠ still unassigned after reconcile: {missing}")
    else:
        print(f"    ✓ all {len(valid_channels)} channels assigned")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved → {output_path.name}")

    return result