"""
phases/phase2_schema/step_f_finalize.py
-----------------------------------------
Step F: Build and save the final schema.

Takes the merged groups from step_e and builds one clean field object
per canonical field, picking the best representative based on observation count.

Output saved to: outputs/schema/final_schema.json
"""

import json


def run(cfg: dict, merged: dict) -> list[dict]:
    """
    Build the final schema from merged groups.

    Args:
        cfg:    result of get_machine_config()
        merged: result dict from step_e containing groups, representatives, field_counts

    Returns:
        list of final schema field dicts
    """
    output_path = cfg["final_schema_path"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_f already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_f — building final schema...")

    groups          = merged["groups"]
    representatives = merged["representatives"]
    field_counts    = merged["field_counts"]

    final_schema = _build_final_schema(groups, representatives, field_counts)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_schema, f, indent=2)

    print(f"    Final schema: {len(final_schema)} fields")
    for field in final_schema:
        count = field_counts.get(field["normalized_name"], 0)
        print(f"      {field['normalized_name']} ({field.get('unit', '-')}) — seen in {count}/5 runs")
    print(f"    Saved → {output_path.name}")

    return final_schema


def _build_final_schema(groups: dict, representatives: dict, field_counts: dict) -> list[dict]:
    """
    Build one clean field object per canonical field.
    Picks the best representative based on observation count.
    """
    final_schema = []

    for canonical, members in groups.items():
        # Pick member with highest count as the representative
        best = max(
            (m for m in members if m in representatives),
            key=lambda m: field_counts.get(m, 0),
            default=None,
        )

        if best is None:
            continue

        field_obj = representatives[best].copy()
        field_obj["normalized_name"] = canonical
        field_obj["field_name"]      = canonical
        final_schema.append(field_obj)

    return final_schema