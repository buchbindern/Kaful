"""
phases/phase7_estimate/step_b_particle_filter.py
--------------------------------------------------
Step B: Run particle filter for state estimation per usage profile.

Runs the particle filter over the observation sequence for each profile,
saving estimated states, uncertainty, and event states at each timestep.

Output saved to: outputs/estimate/{profile}/step_b_estimated_states.json
"""

import json
import sys
import importlib.util

import numpy as np
from progpy.state_estimators import ParticleFilter
from progpy import PrognosticsModel

from config import NUM_PARTICLES, ESTIMATE_FREQ


class ObservableCompositeModel(PrognosticsModel):
    """Wraps composite model exposing only observable outputs to particle filter."""

    def __init__(self, composite_model, output_map):
        self._composite     = composite_model
        self.inputs         = composite_model.inputs
        self.states         = composite_model.states
        self.events         = composite_model.events
        self.outputs        = list(output_map.values())
        self.default_parameters = composite_model.parameters
        super().__init__()

    def initialize(self, u=None, z=None):
        return self._composite.initialize(u, z)

    def next_state(self, x, u, dt):
        return self._composite.next_state(x, u, dt)

    def output(self, x):
        full_out = self._composite.output(x)
        return self.OutputContainer({
            port: full_out[port] for port in self.outputs
        })

    def event_state(self, x):
        return self._composite.event_state(x)

    def threshold_met(self, x):
        return self._composite.threshold_met(x)


def run(cfg: dict, all_profiles: dict) -> dict:
    """
    Run particle filter for each usage profile.

    Args:
        cfg:          result of get_machine_config()
        all_profiles: result from step_a

    Returns:
        dict of {profile_name: estimation_results}
    """
    print(f"  Running step_b — particle filter ({NUM_PARTICLES} particles)...")

    # Load composite model and future_loading_eqn
    composite_model, future_loading_eqn = _load_composite(cfg)

    all_results = {}

    for profile_name, profile_data in all_profiles.items():
        output_path = cfg["estimate_dir"] / profile_name / "step_b_estimated_states.json"

        if output_path.exists():
            print(f"    ✓ {profile_name} already done — loading from disk")
            with open(output_path) as f:
                all_results[profile_name] = json.load(f)
            continue

        print(f"    Running particle filter for {profile_name}...")

        result = _run_particle_filter(
            composite_model=composite_model,
            future_loading_eqn=future_loading_eqn,
            profile_data=profile_data,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        all_results[profile_name] = result

        print(f"      ✓ {len(result['estimated_states'])} estimates saved → {output_path.name}")

    return all_results


def _run_particle_filter(composite_model, future_loading_eqn, profile_data) -> dict:
    """Run particle filter over one profile's observations."""
    times        = profile_data["times"]
    observations = profile_data["observations"]
    output_map   = profile_data["output_map"]

    obs_model = ObservableCompositeModel(composite_model, output_map)

    pf = ParticleFilter(
        obs_model,
        composite_model.initialize(),
        num_particles=NUM_PARTICLES,
    )

    estimated_states  = []
    state_uncertainty = []
    estimated_events  = []

    last_t = times[0] if times else 0

    for i, (t, obs) in enumerate(zip(times[1:], observations[1:])):
        if t <= last_t:
            continue
        last_t = t

        u         = future_loading_eqn(t)
        clean_obs = {k: v for k, v in obs.items() if v is not None and v == v}

        if not clean_obs:
            continue

        try:
            pf.estimate(t, u, obs_model.OutputContainer(clean_obs))
        except Exception as e:
            print(f"      ✗ estimate failed at t={t}: {e}")
            break

        if i % ESTIMATE_FREQ == 0:
            matrix = pf.particles._matrix
            keys   = list(pf.particles.keys())

            state_mean = {}
            state_std  = {}
            for row_idx, k in enumerate(keys):
                vals = matrix[row_idx]
                vals = vals[np.isfinite(vals)]
                state_mean[k] = float(np.mean(vals)) if len(vals) > 0 else None
                state_std[k]  = float(np.std(vals))  if len(vals) > 0 else None

            # Event states across all particles
            event_mean = {k: [] for k in composite_model.events}
            for col_idx in range(matrix.shape[1]):
                particle_dict = {
                    k: float(matrix[row_idx, col_idx])
                    for row_idx, k in enumerate(keys)
                }
                try:
                    x_p = composite_model.StateContainer(particle_dict)
                    es  = composite_model.event_state(x_p)
                    for k in composite_model.events:
                        event_mean[k].append(es[k])
                except Exception:
                    pass

            event_mean = {
                k: float(np.mean(v)) if v else None
                for k, v in event_mean.items()
            }

            estimated_states.append({"t": t, **state_mean})
            state_uncertainty.append({"t": t, **state_std})
            estimated_events.append({"t": t, **event_mean})

        if i % 1000 == 0 and i > 0:
            print(f"      step {i}/{len(times)}...")

    return {
        "estimated_states":  estimated_states,
        "state_uncertainty": state_uncertainty,
        "estimated_events":  estimated_events,
        "final_particles":   {
            k: float(np.mean(pf.particles._matrix[row_idx]))
            for row_idx, k in enumerate(pf.particles.keys())
        },
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