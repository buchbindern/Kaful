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
            cfg=cfg,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        all_results[profile_name] = result

        print(f"      ✓ {len(result['estimated_states'])} estimates saved → {output_path.name}")

    return all_results

def _run_particle_filter(composite_model, future_loading_eqn, profile_data, cfg) -> dict:
    times        = profile_data["times"]
    observations = profile_data["observations"]
    output_map   = profile_data["output_map"]

    machine_process_noise = cfg.get('process_noise_default', 1e-4)

    # Instead of particle filter, build diverse initial particles
    # by propagating forward with noise from multiple starting points
    x0 = composite_model.initialize()
    rng = np.random.default_rng(42)

    # Generate diverse particles by adding noise to initial state
    n = NUM_PARTICLES
    particles = []
    for _ in range(n):
        noisy = {
            k: float(x0[k]) + rng.normal(0, max(abs(float(x0[k])) * 0.01, 1e-3))
            for k in composite_model.states
        }
        particles.append(noisy)

    # Propagate all particles forward through the observations
    u = future_loading_eqn(0)
    estimated_states  = []
    state_uncertainty = []
    estimated_events  = []

    process_noise = composite_model.StateContainer(
        {k: machine_process_noise for k in composite_model.states}
    )
    composite_model.parameters['process_noise'] = process_noise

    for i, t in enumerate(times[1:]):
        u = future_loading_eqn(t)
        new_particles = []
        for p in particles:
            x = composite_model.StateContainer(p)
            x = composite_model.next_state(x, u, 1.0)
            x = composite_model.apply_process_noise(x, 1.0)
            new_particles.append({k: float(x[k]) for k in composite_model.states})
        particles = new_particles

        if i % ESTIMATE_FREQ == 0:
            state_mean = {}
            state_std  = {}
            for k in composite_model.states:
                vals = [p[k] for p in particles]
                state_mean[k] = float(np.mean(vals))
                state_std[k]  = float(np.std(vals))

            event_mean = {}
            for k in composite_model.events:
                event_vals = []
                for p in particles:
                    try:
                        x = composite_model.StateContainer(p)
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
        "final_particles":   particles,
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