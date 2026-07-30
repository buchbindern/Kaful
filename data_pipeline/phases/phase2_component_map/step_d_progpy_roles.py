"""
phases/phase2_component_map/step_d_progpy_roles.py
----------------------------------------------------
Step d: Assign ProgPy-specific roles to each signal.

ProgPy models have:
  - states:    latent health variable(s) propagated by the state equation
  - inputs:    known process conditions at each timestep
  - outputs:   sensor features the model predicts (compared to observations)

After the LLM responds, reconcile.py validates every output_variable name
against the actual signals from cuts.parquet. Invalid names are fuzzy-matched
to real ones or dropped. channel and stat fields are always re-derived from
ground truth — never trusted from the LLM.

This guarantees that every feature name in output_variables exists in
cuts.parquet and has correct channel/stat metadata, regardless of dataset.

Output: step_d_progpy_roles.json
{
  "state_variables":  [ { "name": "wear_mm", "unit": "mm", ... } ],
  "input_variables":  [ { "name": "spindle_speed_rpm", ... } ],
  "output_variables": [
    {
      "name":                  "force_z_rms",   ← always a real cuts.parquet column
      "channel":               "force_z",        ← always re-derived from signals
      "stat":                  "rms",            ← always re-derived from signals
      "unit":                  "N",
      "degradation_relevance": "high",
      "use_for_filter":        true,
      "reasoning":             "..."
    },
    ...
  ],
  "cutting_params":      { ... },
  "failure_threshold":   0.2,
  "model_form_hint":     "power_law",
  "primary_component":   "cutting_tool"
}
"""

import json
from utils.llm import call_claude
from utils.parsing import parse_json
from phases.phase2_component_map.reconcile import reconcile_output_variables


SYSTEM_PROMPT = """You are an expert in ProgPy prognostics models and
particle filter state estimation for remaining useful life prediction."""


PROGPY_ROLES_PROMPT = """
You are assigning ProgPy variable roles for a digital twin model.

ProgPy requires:
  states   — latent variables propagated forward by the state equation
  inputs   — known external conditions at each timestep (constants are fine)
  outputs  — sensor features the model predicts (for particle filter updates)

Degradation pathway:
{pathway_json}

Available signal features — use ONLY these exact names in output_variables:
{features_json}

Process conditions (candidates for input_variables):
{cutting_params_json}

Rules:
- states: the health state variable only (e.g. wear_mm, damage fraction)
- inputs: constant process conditions that drive wear rate (speed, feed, depth)
- outputs: 3–6 HIGH-degradation features from the list above
  — use the exact "feature_name" value from the list, nothing else
  — more than 6 outputs hurts particle filter performance
- set use_for_filter=true for features that are monotonically correlated
  with the state variable and have low process-input confounding

Return ONLY valid JSON:
{{
  "state_variables": [
    {{
      "name":          "snake_case_name",
      "unit":          "unit string",
      "initial_value": 0.0,
      "source":        "wear.parquet | synthetic",
      "description":   "one sentence"
    }}
  ],
  "input_variables": [
    {{
      "name":        "snake_case_name",
      "unit":        "unit string",
      "value":       0.0,
      "description": "one sentence"
    }}
  ],
  "output_variables": [
    {{
      "name":                  "exact feature_name from the list above",
      "degradation_relevance": "high | medium",
      "use_for_filter":        true,
      "reasoning":             "why this feature tracks degradation state"
    }}
  ]
}}
"""


def run(cfg: dict, signals: list[dict], component_map: dict, pathway: dict) -> dict:
    output_path = cfg["step_d_progpy_path"]

    if output_path.exists():
        print("  ✓ step_d already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_d — assigning ProgPy roles (LLM)...")

    # ── Build feature list: only high/medium relevance channels ───────────────
    # This is what the LLM chooses from — ground truth names only.
    assignments = component_map.get("channel_assignments", {})
    relevant_features = [
        {
            "feature_name":          sig["name"],      # exact cuts.parquet column name
            "channel":               sig["channel"],
            "stat":                  sig["stat"],
            "unit":                  sig["unit"],
            "degradation_relevance": assignments.get(
                sig["channel"], {}
            ).get("degradation_relevance", "unknown"),
        }
        for sig in signals
        if assignments.get(sig["channel"], {}).get("degradation_relevance", "") in ("high", "medium")
    ]

    # ── Cutting params ────────────────────────────────────────────────────────
    meta           = cfg["meta"]
    cutting_params = {}
    if meta is not None and hasattr(meta, "CUTTING_PARAMS"):
        cutting_params = dict(vars(meta).get("CUTTING_PARAMS", {}))
    if not cutting_params:
        with open(cfg["ingest_meta_path"]) as f:
            im = json.load(f)
        cutting_params = im.get("cutting_params") or {}

    # ── LLM call ──────────────────────────────────────────────────────────────
    prompt = PROGPY_ROLES_PROMPT.format(
        pathway_json=json.dumps(pathway, indent=2),
        features_json=json.dumps(relevant_features, indent=2),
        cutting_params_json=json.dumps(cutting_params, indent=2),
    )

    raw    = call_claude(prompt, system=SYSTEM_PROMPT, max_tokens=3000, temperature=0.2)
    result = parse_json(raw)

    if not result:
        raise ValueError(f"step_d: failed to parse ProgPy roles.\nRaw:\n{raw[:600]}")

    # ── Reconcile: validate + fix all output_variable names ──────────────────
    # This is the critical step — snaps every output name to a real
    # cuts.parquet column and re-derives channel/stat from ground truth.
    result, warnings = reconcile_output_variables(result, signals)

    for w in warnings:
        print(f"    ⚠ {w}")

    # ── Attach metadata for phases 3 and 4 ───────────────────────────────────
    result["cutting_params"]      = cutting_params
    result["failure_threshold"]   = pathway.get("failure_threshold")
    result["state_variable_unit"] = pathway.get("state_variable_unit", "mm")
    result["model_form_hint"]     = pathway.get("model_form_hint", "power_law")
    result["primary_component"]   = pathway.get("primary_component")

    # ── Print summary ─────────────────────────────────────────────────────────
    states         = result.get("state_variables", [])
    inputs         = result.get("input_variables", [])
    outputs        = result.get("output_variables", [])
    filter_features = [o["name"] for o in outputs if o.get("use_for_filter")]

    print(f"    states  ({len(states)}): {[s['name'] for s in states]}")
    print(f"    inputs  ({len(inputs)}): {[i['name'] for i in inputs]}")
    print(f"    outputs ({len(outputs)}): {[o['name'] for o in outputs]}")
    print(f"    filter features ({len(filter_features)}): {filter_features}")

    # Sanity check: all filter features must exist in signals
    sig_names = {s["name"] for s in signals}
    bad = [f for f in filter_features if f not in sig_names]
    if bad:
        raise ValueError(
            f"step_d: filter features not in cuts.parquet after reconcile: {bad}\n"
            f"This is a bug in reconcile.py — please report."
        )
    print(f"    ✓ all filter features verified in cuts.parquet")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved → {output_path.name}")

    return result