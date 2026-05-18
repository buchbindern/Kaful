"""
phases/phase7_estimate/step_a_load.py
---------------------------------------
Step A: Load simulator events and build observations for the particle filter.

Loads each usage profile CSV from phase 3, maps CSV columns to composite
model output ports, and builds the observation sequence.

Output saved to: outputs/estimate/{profile}/step_a_observations.json
"""

import json
import importlib.util
import sys
from pathlib import Path

import pandas as pd


def run(cfg: dict) -> dict:
    """
    Load simulator events and build observations per usage profile.

    Args:
        cfg: result of get_machine_config()

    Returns:
        dict of {profile_name: {times, observations, output_map}}
    """
    print("  Running step_a — loading simulator data...")

    # Load composite model to get output port names
    composite_model = _load_composite(cfg)

    # Find all profile CSVs
    simulate_dir = cfg["simulate_dir"]
    csv_files    = list(simulate_dir.glob("coffee_machine_events_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No events CSVs found in {simulate_dir}. "
            f"Run phase 3 first."
        )

    print(f"    Found {len(csv_files)} usage profiles")

    all_profiles = {}

    for csv_path in sorted(csv_files):
        profile_name = csv_path.stem.replace("coffee_machine_events_", "")
        output_path  = cfg["estimate_dir"] / profile_name / "step_a_observations.json"

        if output_path.exists():
            print(f"    ✓ {profile_name} already done — loading from disk")
            with open(output_path) as f:
                all_profiles[profile_name] = json.load(f)
            continue

        print(f"    Processing {profile_name}...")

        df = pd.read_csv(csv_path)

        # Map CSV columns to composite model output ports
        output_map = _build_output_map(df, composite_model)
        print(f"      Observable ports: {len(output_map)}/{len(composite_model.outputs)}")

        # Build observation sequence
        times, observations = _build_observations(df, output_map)
        print(f"      Timesteps: {len(times)}")

        result = {
            "profile_name": profile_name,
            "times":        times,
            "observations": observations,
            "output_map":   output_map,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        all_profiles[profile_name] = result
        print(f"      Saved → {output_path.name}")

    return all_profiles


def _load_composite(cfg: dict):
    """Load composite model from disk."""
    composite_path = cfg["codegen_composite_path"]

    if not composite_path.exists():
        raise FileNotFoundError(
            f"No composite model found at {composite_path}. "
            f"Run phase 6 first."
        )

    # Clear cached modules
    for key in list(sys.modules.keys()):
        if any(n in key for n in ["composite", "ceramic", "brewing", "coffee",
                                   "steam", "water", "milk", "powder", "cleaning"]):
            del sys.modules[key]

    spec_obj = importlib.util.spec_from_file_location("composite_model", composite_path)
    mod      = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)

    return mod.composite_model


def _build_output_map(df: pd.DataFrame, composite_model) -> dict:
    """Map CSV column names to composite model output port names."""
    output_map = {}
    for col in df.columns:
        for port in composite_model.outputs:
            if port.split(".")[-1] == col:
                output_map[col] = port
                break
    return output_map


def _build_observations(df: pd.DataFrame, output_map: dict) -> tuple:
    """Build observation sequence from dataframe."""
    times        = list(range(len(df)))
    observations = []

    for _, row in df.iterrows():
        obs = {}
        for col, port in output_map.items():
            val = row.get(col)
            if pd.notna(val):
                obs[port] = float(val)
        observations.append(obs)

    return times, observations