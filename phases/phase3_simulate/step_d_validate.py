"""
phases/phase3_simulate/step_d_validate.py
------------------------------------------
Step D: Validate simulator output against the schema.

Runs the generated simulator, then checks:
1. All schema fields are present
2. No extra fields
3. Null fractions per field
4. Numeric fields within typical_range

Output saved to:
    outputs/simulate/events.csv
    outputs/simulate/maintenance_log.csv
    outputs/simulate/step_d_validation.json
"""

import json
import re
import subprocess
import sys

import pandas as pd


def run(cfg: dict, schema: list[dict]) -> dict:
    """
    Run the simulator and validate its output.

    Args:
        cfg:    result of get_machine_config()
        schema: final schema from phase 2

    Returns:
        validation report dict
    """
    output_path = cfg["sim_step_d_validation"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_d already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_d — running simulator and validating output...")

    # Run the simulator as a subprocess
    simulator_path = cfg["sim_step_c_code"]
    print(f"  Running {simulator_path.name}...")

    result = subprocess.run(
        [sys.executable, str(simulator_path)],
        cwd=str(cfg["simulate_dir"]),  # simulator saves relative to this dir
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Simulator failed with return code {result.returncode}.\n"
            f"stderr:\n{result.stderr[:1000]}"
        )

    print("  Simulator ran successfully.")
    if result.stdout:
        print(f"  Output:\n{result.stdout[:500]}")

    # Load events CSV
    # Load and concatenate all profile CSVs
    event_files = list(cfg["simulate_dir"].glob("*events_*.csv"))
    #event_files = list(cfg["simulate_dir"].glob("coffee_machine_events_*.csv"))
    if not event_files:
        raise FileNotFoundError(f"No events CSVs found in {cfg['simulate_dir']}")

    events_df = pd.concat([pd.read_csv(f) for f in event_files], ignore_index=True)
    print(f"  Loaded {len(events_df)} events from {len(event_files)} profiles")

    # Validate
    report       = _validate_fields(events_df, schema)
    range_report = _evaluate_ranges(events_df, schema)

    report["range_report"] = range_report

    # Print summary
    _print_report(report)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return report


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_fields(events_df: pd.DataFrame, schema: list[dict]) -> dict:
    """Check field presence, extras, and null fractions."""
    schema_fields = [row["field_name"] for row in schema]
    actual_fields = list(events_df.columns)

    missing_fields = [f for f in schema_fields if f not in actual_fields]
    extra_fields   = [f for f in actual_fields if f not in schema_fields]

    return {
        "row_count":     len(events_df),
        "column_count":  len(actual_fields),
        "missing_fields": missing_fields,
        "extra_fields":   extra_fields,
        "null_fraction":  events_df.isna().mean().sort_values(ascending=False).to_dict(),
    }


def _evaluate_ranges(events_df: pd.DataFrame, schema: list[dict]) -> list[dict]:
    """Check numeric fields against typical_range."""
    rows = []

    for row in schema:
        field = row["field_name"]

        if field not in events_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(events_df[field]):
            continue

        low, high = _parse_range(row.get("typical_range", ""))
        if low is None or high is None:
            continue

        s = events_df[field].dropna()
        if s.empty:
            continue

        out_of_range_pct = ((s < low) | (s > high)).mean()

        rows.append({
            "field":            field,
            "typical_range":    row.get("typical_range"),
            "observed_min":     round(float(s.min()), 4),
            "observed_max":     round(float(s.max()), 4),
            "out_of_range_pct": round(float(out_of_range_pct), 4),
        })

    return sorted(rows, key=lambda x: x["out_of_range_pct"], reverse=True)


def _parse_range(range_text: str):
    """Extract (low, high) from a typical_range string."""
    if not isinstance(range_text, str) or not range_text.strip():
        return None, None

    nums = re.findall(r"-?\d+\.?\d*", range_text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])

    return None, None


def _print_report(report: dict) -> None:
    """Print a readable validation summary."""
    print(f"\n    Validation Report:")
    print(f"      Rows:            {report['row_count']}")
    print(f"      Columns:         {report['column_count']}")

    if report["missing_fields"]:
        print(f"      ⚠ Missing fields: {report['missing_fields']}")
    else:
        print(f"      ✓ No missing fields")

    if report["extra_fields"]:
        print(f"      ⚠ Extra fields:   {report['extra_fields']}")
    else:
        print(f"      ✓ No extra fields")

    # Show high null fraction fields
    high_null = {k: v for k, v in report["null_fraction"].items() if v > 0.5}
    if high_null:
        print(f"      ⚠ High null fields (>50%):")
        for field, frac in list(high_null.items())[:5]:
            print(f"        {field:<35} {frac:.1%}")

    # Show out of range fields
    out_of_range = [r for r in report["range_report"] if r["out_of_range_pct"] > 0.05]
    if out_of_range:
        print(f"      ⚠ Out of range fields (>5%):")
        for r in out_of_range[:5]:
            print(f"        {r['field']:<35} {r['out_of_range_pct']:.1%} out of range")
    else:
        print(f"      ✓ All numeric fields within typical ranges")