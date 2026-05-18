"""
pipeline/test_preprocessor.py

Tests the preprocessor with synthetic coffee machine data
that mirrors the real CSV structure you showed earlier.
"""

import sys
import os
import json
import tempfile
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.preprocessor import build_data_summary, save_summary, load_summary

# ------------------------------------------------------------------
# Generate synthetic CSVs that mirror your real data
# ------------------------------------------------------------------

def make_synthetic_events(n=100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "event_timestamp":        pd.date_range("2024-01-01", periods=n, freq="10min"),
        "machine_id":             ["CM_001"] * n,
        "product_type":           np.random.choice(["latte","espresso","americano","cappuccino"], n),
        "grinder_position":       np.random.randint(1, 10, n),
        "grind_size_setting":     np.random.randint(1, 40, n),
        "coffee_dose_weight":     np.random.normal(18, 0.5, n).round(2),
        "water_temperature_brewing": np.random.normal(93, 1.5, n).round(2),
        "water_volume_dispensed": np.random.normal(36, 2, n).round(2),
        "extraction_time":        np.random.normal(28, 3, n).round(2),
        "extraction_pressure":    np.random.normal(9, 0.5, n).round(2),
        "tamping_pressure":       np.random.normal(15, 1, n).round(2),
        "milk_volume":            np.random.choice([0, 120, 150, 180], n).astype(float),
        "milk_temperature":       np.random.normal(65, 3, n).round(2),
        "steam_temperature":      np.random.normal(130, 5, n).round(2),
        "steam_pressure":         np.random.normal(1.2, 0.1, n).round(3),
        "grounds_drawer_weight":  np.random.normal(200, 50, n).round(1),
        "bean_hopper_level":      np.random.uniform(0, 100, n).round(1),
        "power_consumption":      np.random.normal(1400, 100, n).round(1),
        "event_success":          np.random.choice([True, False], n, p=[0.97, 0.03]),
        "error_code":             np.random.choice([None, "E01", "E02"], n, p=[0.97, 0.02, 0.01]),
        "cleaning_cycle_count":   np.arange(n),
        "ambient_temperature":    np.random.normal(22, 2, n).round(1),
        "water_pressure_inlet":   np.random.normal(3.5, 0.2, n).round(2),
    })


def make_synthetic_maintenance(n=30) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "timestamp":           pd.date_range("2024-01-01", periods=n, freq="1D"),
        "machine_id":          ["CM_001"] * n,
        "maintenance_type":    np.random.choice(["cleaning","descaling","refill"], n),
        "performed_by":        np.random.choice(["technician","auto"], n),
        "duration_minutes":    np.random.randint(5, 60, n),
        "grinder_wear_before": np.random.uniform(0.1, 0.9, n).round(3),
        "brewing_wear_before": np.random.uniform(0.05, 0.7, n).round(3),
        "coffee_buildup_before": np.random.uniform(0, 0.5, n).round(3),
        "scale_buildup_before":  np.random.uniform(0, 0.3, n).round(3),
    })


# ------------------------------------------------------------------
# Run the test
# ------------------------------------------------------------------

print("=== Preprocessor Test ===\n")

with tempfile.TemporaryDirectory() as tmpdir:
    # Write synthetic CSVs
    events_path = f"{tmpdir}/events.csv"
    maint_path  = f"{tmpdir}/maintenance.csv"
    out_path    = f"{tmpdir}/summary.json"

    make_synthetic_events().to_csv(events_path, index=False)
    make_synthetic_maintenance().to_csv(maint_path, index=False)

    # Run preprocessor
    summary = build_data_summary(
        machine_name="coffee_machine",
        events_csv=events_path,
        maintenance_csv=maint_path,
    )

    # Save and reload (round-trip test)
    save_summary(summary, out_path)
    reloaded = load_summary(out_path)

    # Print key parts of the summary
    print("\n--- Identity ---")
    print(f"  machine_name : {reloaded.machine_name}")
    print(f"  machine_type : {reloaded.machine_type}")
    print(f"  date_range   : {reloaded.date_range}")
    print(f"  product_types: {reloaded.product_types}")

    print("\n--- Signals (sample) ---")
    for s in reloaded.signals[:5]:
        if s.mean is not None:
            print(f"  {s.name}: min={s.min}, max={s.max}, mean={s.mean}, unit={s.unit_hint}")
        else:
            print(f"  {s.name}: categories={s.categories[:3]}...")

    print("\n--- Wear Signals ---")
    for w in reloaded.wear_signals:
        print(f"  {w.name}: min={w.min}, max={w.max}, mean={w.mean}")
        print(f"    associated maintenance: {w.maintenance_types}")

    print("\n--- Column Classification ---")
    print(f"  temporal    : {reloaded.temporal_columns}")
    print(f"  ids         : {reloaded.id_columns}")
    print(f"  targets     : {reloaded.target_columns}")
    print(f"  degradation : {reloaded.degradation_indicators}")

    # Validate round-trip
    assert reloaded.machine_name == "coffee_machine"
    assert reloaded.machine_type == "coffee"
    assert len(reloaded.signals) > 0
    assert len(reloaded.wear_signals) == 4
    assert reloaded.has_maintenance is True

    print("\n✅ All assertions passed — preprocessor working correctly")

    # Show what the JSON looks like (first 500 chars)
    with open(out_path) as f:
        raw = f.read()
    print(f"\n--- Summary JSON preview (first 500 chars) ---")
    print(raw[:500])