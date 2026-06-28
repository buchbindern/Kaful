"""
phases/phase7_estimate/step_b_estimate.py
-------------------------------------------
Step B: Open-loop ensemble state estimation per usage profile.

Initializes an ensemble by perturbing the composite model's initial state,
then propagates every member forward under the loading profile, recording
ensemble-mean state, spread (uncertainty), and event states at each sample.

This is open-loop forward propagation: it does NOT condition on observations.
Bayesian updating (likelihood + resampling) is deliberate future work — see
the project roadmap.

Output saved to: outputs/estimate/{profile}/step_b_estimated_states.json
"""

import json
import sys
import importlib.util

import numpy as np

from config import ENSEMBLE_SIZE, ESTIMATE_FREQ


def run(cfg: dict, all_profiles: dict) -> dict:
    """
    Run ensemble propagation for each usage profile.

    Args:
        cfg:          result of get_machine_config()
        all_profiles: result from step_a

    Returns:
        dict of {profile_name: estimation_results}
    """
    print(f"  Running step_b — ensemble propagation ({ENSEMBLE_SIZE} members)...")

    composite_model, future_loading_eqn = _load_composite(cfg)

    all_results = {}

    for profile_name, profile_data in all_profiles.items():
        output_path = cfg["estimate_dir"] / profile_name / "step_b_estimated_states.json"

        if output_path.exists():
            print(f"    ✓ {profile_name} already done — loading from disk")
            with open(output_path) as f:
                all_results[profile_name] = json.load(f)
            continue

        print(f"    Propagating ensemble for {profile_name}...")

        result = _run_ensemble_propagation(
            composite_model=composite_model,
            future_loading_eqn=future_loading_eqn,
            profile_data=profile_data,
            cfg=cfg,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        all_results[profile_name] = result

        print(f"      ✓ {len(result['estimated_states'])} estimates saved → {output_path.name}")

    return all_results


def _run_ensemble_propagation(composite_model, future_loading_eqn, profile_data, cfg) -> dict:
    times = profile_data["times"]

    machine_process_noise = cfg.get("process_noise_default", 1e-4)

    # Build a diverse ensemble by perturbing the initial state (~1% per state).
    x0  = composite_model.initialize()
    rng = np.random.default_rng(42)

    ensemble = []
    for _ in range(ENSEMBLE_SIZE):
        noisy = {
            k: float(x0[k]) + rng.normal(0, max(abs(float(x0[k])) * 0.01, 1e-3))
            for k in composite_model.states
        }
        ensemble.append(noisy)

    # Propagate every member forward under the loading profile (open-loop).
    estimated_states  = []
    state_uncertainty = []
    estimated_events  = []

    process_noise = composite_model.StateContainer(
        {k: machine_process_noise for k in composite_model.states}
    )
    composite_model.parameters["process_noise"] = process_noise

    for i, t in enumerate(times[1:]):
        u = future_loading_eqn(t)
        new_ensemble = []
        for p in ensemble:
            x = composite_model.StateContainer(p)
            x = composite_model.next_state(x, u, 1.0)
            x = composite_model.apply_process_noise(x, 1.0)
            new_ensemble.append({k: float(x[k]) for k in composite_model.states})
        ensemble = new_ensemble

        if i % ESTIMATE_FREQ == 0:
            state_mean = {}
            state_std  = {}
            for k in composite_model.states:
                vals = [p[k] for p in ensemble]
                state_mean[k] = float(np.mean(vals))
                state_std[k]  = float(np.std(vals))

            event_mean = {}
            for k in composite_model.events:
                event_vals = []
                for p in ensemble:
                    try:
                        x  = composite_model.StateContainer(p)
                        es = composite_model.event_state(x)
                        event_vals.append(es[k])
                    except Exception:
                        pass
                event_mean[k] = float(np.mean(event_vals)) if event_vals else None

            estimated_states.append({"t": t, **state_mean})
            state_uncertainty.append({"t": t, **state_std})
            estimated_events.append({"t": t, **event_mean})

        if i % 1000 == 0 and i > 0:
            print(f"      step {i}/{len(times)}...")

    return {
        "estimated_states":  estimated_states,
        "state_uncertainty": state_uncertainty,
        "estimated_events":  estimated_events,
        "final_ensemble":    ensemble,
    }


def _load_composite(cfg: dict):
    """Load composite model and future_loading_eqn from disk."""
    composite_path = cfg["codegen_composite_path"]

    for key in list(sys.modules.keys()):
        if any(n in key for n in ["composite", "ceramic", "brewing", "coffee",
                                   "steam", "water", "milk", "powder", "cleaning"]):
            del sys.modules[key]

    spec_obj = importlib.util.spec_from_file_location("composite_model", composite_path)
    mod      = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(mod)

    return mod.composite_model, mod.future_loading_eqn