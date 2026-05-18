"""
phases/phase7_estimate/step_c_rul.py
--------------------------------------
Step C: Monte Carlo RUL prediction per usage profile.

Uses the final particle filter state as the starting point for
Monte Carlo prediction of Remaining Useful Life for each event.

Output saved to: outputs/estimate/{profile}/step_c_rul.json
"""

import json
import sys
import importlib.util

import numpy as np
from progpy.predictors import MonteCarlo

from config import MC_SAMPLES, PREDICTION_HORIZON


def run(cfg: dict, estimation_results: dict) -> dict:
    """
    Run Monte Carlo RUL prediction for each usage profile.

    Args:
        cfg:                result of get_machine_config()
        estimation_results: result from step_b

    Returns:
        dict of {profile_name: rul_results}
    """
    print(f"  Running step_c — Monte Carlo RUL ({MC_SAMPLES} samples, horizon={PREDICTION_HORIZON})...")

    composite_model, _ = _load_composite(cfg)
    external_inputs    = _load_external_inputs(cfg)

    all_rul = {}

    for profile_name, result in estimation_results.items():
        output_path = cfg["estimate_dir"] / profile_name / "step_c_rul.json"

        if output_path.exists():
            print(f"    ✓ {profile_name} already done — loading from disk")
            with open(output_path) as f:
                all_rul[profile_name] = json.load(f)
            continue

        print(f"    Running RUL prediction for {profile_name}...")

        rul = _run_rul(
            composite_model=composite_model,
            external_inputs=external_inputs,
            final_particles=result["final_particles"],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rul, f, indent=2)

        all_rul[profile_name] = rul

        # Print summary
        print(f"      RUL predictions:")
        for event, stats in rul["predictions"].items():
            if stats.get("already_triggered"):
                print(f"        ⚠ {event}: already triggered")
            elif stats.get("rul_mean") is not None:
                print(f"        {event}: {stats['rul_mean']:.0f} ± {stats['rul_std']:.0f} steps")
            else:
                print(f"        {event}: beyond horizon ({PREDICTION_HORIZON} steps)")

        print(f"      Saved → {output_path.name}")

    return all_rul


def _run_rul(composite_model, external_inputs, final_particles) -> dict:
    """Run Monte Carlo RUL prediction from final particle state."""

    def prediction_loading(t, x=None):
        return composite_model.InputContainer({
            k: v for k, v in external_inputs.items()
            if k in composite_model.inputs
        })

    # Check which events already triggered
    x_start = composite_model.StateContainer(final_particles)
    es      = composite_model.event_state(x_start)

    triggered_at_start = [k for k, v in es.items() if v <= 0.0]
    healthy_at_start   = [k for k, v in es.items() if v > 0.0]

    mc = MonteCarlo(composite_model)

    try:
        mc_results = mc.predict(
            final_particles,
            prediction_loading,
            n_samples=MC_SAMPLES,
            horizon=PREDICTION_HORIZON,
            dt=10.0,
            save_freq=1000,
        )
    except Exception as e:
        return {"error": str(e), "predictions": {}}

    predictions = {}

    # Already triggered
    for event in triggered_at_start:
        predictions[event] = {"already_triggered": True, "rul_mean": 0, "rul_std": 0}

    # Healthy events — extract RUL from Monte Carlo results
    for event in healthy_at_start:
        try:
            times = mc_results.time_of_event.get(event)
            if times is None or len(times) == 0:
                predictions[event] = {
                    "already_triggered": False,
                    "rul_mean":          None,
                    "rul_std":           None,
                    "beyond_horizon":    True,
                }
            else:
                valid = [t for t in times if t is not None and np.isfinite(t)]
                if valid:
                    predictions[event] = {
                        "already_triggered": False,
                        "rul_mean":          float(np.mean(valid)),
                        "rul_std":           float(np.std(valid)),
                        "rul_min":           float(np.min(valid)),
                        "rul_max":           float(np.max(valid)),
                        "beyond_horizon":    False,
                    }
                else:
                    predictions[event] = {
                        "already_triggered": False,
                        "rul_mean":          None,
                        "rul_std":           None,
                        "beyond_horizon":    True,
                    }
        except Exception as e:
            predictions[event] = {"error": str(e)}

    return {"predictions": predictions}


def _load_composite(cfg: dict):
    """Load composite model from disk."""
    composite_path = cfg["codegen_composite_path"]

    for key in list(sys.modules.keys()):
        if any(n in key for n in ["composite", "ceramic", "brewing", "coffee",
                                   "steam", "water", "milk", "powder", "cleaning"]):
            del sys.modules[key]

    spec_obj = importlib.util.spec_from_file_location("composite_model", composite_path)
    mod      = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)

    return mod.composite_model, mod.future_loading_eqn


def _load_external_inputs(cfg: dict) -> dict:
    """Load external inputs from phase 6."""
    path = cfg["codegen_step_d_external_inputs"]
    with open(path) as f:
        return json.load(f)