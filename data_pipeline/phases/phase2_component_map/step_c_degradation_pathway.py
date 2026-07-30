"""
phases/phase2_component_map/step_c_degradation_pathway.py
-----------------------------------------------------------
Step c: Identify the primary degradation pathway and state variable.

This step answers: what is degrading, how does it degrade, and what
single quantity represents the health state for ProgPy modelling?

For PHM 2010 this is obvious (tool flank wear VB in mm) but the step
exists so the pipeline works generically — for a bearing dataset the
state variable might be RMS vibration amplitude, for a battery it might
be capacity fade.

When queries.py provides PRIMARY_STATE_VARIABLE it short-circuits the
LLM call entirely and records it directly. This is the right behaviour
for well-understood datasets where we know the answer.

Output: step_c_degradation_pathway.json
{
  "primary_component":    "cutting_tool",
  "state_variable":       "wear_mm",
  "state_variable_unit":  "mm",
  "failure_threshold":    0.2,
  "threshold_source":     "ISO 8688 / dataset_meta.py",
  "degradation_mechanism":"abrasive flank wear from workpiece contact",
  "model_form_hint":      "power_law",
  "monotonic":            true,
  "reasoning":            "..."
}
"""

import json
from utils.llm import call_claude
from utils.parsing import parse_json


SYSTEM_PROMPT = """You are a mechanical engineering expert specialising in
predictive maintenance, remaining useful life estimation, and physics-based
degradation modelling."""


DEGRADATION_PATHWAY_PROMPT = """
You are identifying the primary degradation pathway for a digital twin.

Dataset context:
{dataset_context}

Components identified:
{components_json}

Channel assignments (with degradation relevance):
{assignments_json}

Wear labels available in data: {has_wear_labels}
Wear label column: {wear_column}
Wear range observed: {wear_range}

Identify:
1. The primary degrading component
2. The state variable that best represents health (must be a quantity we can
   track over time — either a measured wear label or an indirect proxy)
3. The degradation mechanism in physical terms
4. Whether degradation is expected to be monotonically increasing
5. A hint for the mathematical model form that best fits this degradation

Return ONLY valid JSON:
{{
  "primary_component":    "component name",
  "state_variable":       "column name in wear.parquet or feature name",
  "state_variable_unit":  "unit string",
  "failure_threshold":    0.0,
  "threshold_source":     "where this threshold comes from",
  "degradation_mechanism":"physical description of what is wearing/degrading",
  "model_form_hint":      "power_law | linear | exponential | paris_law",
  "monotonic":            true,
  "reasoning":            "explanation"
}}

Model form hints:
- power_law:   VB(t) = C * t^n   — common for abrasive tool wear (Taylor-type)
- linear:      x(t) = x0 + r*t   — constant-rate degradation
- exponential: x(t) = x0 * e^(r*t) — accelerating degradation, fatigue crack growth
- paris_law:   da/dN = C*(dK)^m  — fracture mechanics crack growth
"""


def run(cfg: dict, component_map: dict) -> dict:
    output_path = cfg["step_c_pathway_path"]

    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_c — identifying degradation pathway...")

    # ── Short-circuit if queries.py declares the state variable ───────────────
    queries = cfg.get("queries", {})
    meta    = cfg["meta"]

    primary_sv   = queries.get("PRIMARY_STATE_VARIABLE")
    primary_unit = queries.get("PRIMARY_STATE_UNIT", "mm")
    model_hint   = queries.get("PREFERRED_MODEL_FORM")
    threshold    = getattr(meta, "WEAR_THRESHOLD_MM", None) if meta else None

    if primary_sv:
        print(f"    queries.py declares state variable: {primary_sv} — skipping LLM")
        result = {
            "primary_component":    _find_primary_component(component_map),
            "state_variable":       primary_sv,
            "state_variable_unit":  primary_unit,
            "failure_threshold":    threshold,
            "threshold_source":     "dataset_meta.py" if threshold else "not set",
            "degradation_mechanism": _infer_mechanism(component_map),
            "model_form_hint":      model_hint or "power_law",
            "monotonic":            True,
            "reasoning":            "State variable and model form set directly from queries.py / dataset_meta.py",
        }
    else:
        result = _llm_identify_pathway(cfg, component_map, threshold, model_hint)

    print(f"    state variable : {result['state_variable']} ({result['state_variable_unit']})")
    print(f"    failure threshold: {result['failure_threshold']} {result['state_variable_unit']}")
    print(f"    model form hint: {result['model_form_hint']}")
    print(f"    monotonic: {result['monotonic']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved → {output_path.name}")

    return result


def _find_primary_component(component_map: dict) -> str:
    """Return the name of the component with role 'primary degrading component'."""
    for comp in component_map.get("components", []):
        if "primary" in comp.get("role", "").lower():
            return comp["name"]
    # Fall back to first component
    comps = component_map.get("components", [])
    return comps[0]["name"] if comps else "unknown"


def _infer_mechanism(component_map: dict) -> str:
    """Pull degradation mechanism hint from component description if available."""
    for comp in component_map.get("components", []):
        if "primary" in comp.get("role", "").lower():
            return comp.get("description", "unknown degradation mechanism")
    return "unknown degradation mechanism"


def _llm_identify_pathway(
    cfg: dict,
    component_map: dict,
    threshold: float | None,
    model_hint: str | None,
) -> dict:
    """Fall back to LLM when queries.py doesn't declare the state variable."""
    import pandas as pd

    queries     = cfg.get("queries", {})
    dataset_ctx = queries.get("DATASET_CONTEXT", "")

    # Check whether wear labels exist and get their range
    wear_df = pd.read_parquet(cfg["wear_path"])
    has_wear = "wear_mm" in wear_df.columns and wear_df["wear_mm"].notna().any()
    if has_wear:
        wear_range = (
            f"{wear_df['wear_mm'].min():.4f} – {wear_df['wear_mm'].max():.4f} mm"
        )
    else:
        wear_range = "not available"

    prompt = DEGRADATION_PATHWAY_PROMPT.format(
        dataset_context=dataset_ctx,
        components_json=json.dumps(component_map.get("components", []), indent=2),
        assignments_json=json.dumps(component_map.get("channel_assignments", {}), indent=2),
        has_wear_labels=str(has_wear),
        wear_column="wear_mm" if has_wear else "none",
        wear_range=wear_range,
    )

    raw    = call_claude(prompt, system=SYSTEM_PROMPT, max_tokens=2000, temperature=0.2)
    result = parse_json(raw)

    if not result:
        raise ValueError(f"step_c: failed to parse degradation pathway.\nRaw:\n{raw[:600]}")

    # Patch in threshold from meta if LLM didn't set it or got it wrong
    if threshold is not None:
        result["failure_threshold"] = threshold
        result["threshold_source"]  = "dataset_meta.py (overrides LLM)"

    return result