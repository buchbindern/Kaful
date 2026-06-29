from progpy import PrognosticsModel
import numpy as np


class CylindricalRollerBearings(PrognosticsModel):
    """
    Prognostics model for cylindrical roller bearings in a spindle assembly.

    Tracks fatigue life, grease condition, and cage wear as primary degradation
    mechanisms. Raises fault events for raceway flaking, scoring, rib damage,
    excessive preload, and cage noise.

    Events:
        flaking_one_side_raceway: Sub-surface fatigue leading to raceway flaking.
        scoring_raceway_rolling_surface: Scoring due to lubricant film breakdown.
        rib_scoring: Scoring between roller end faces and guide rib.
        excessive_preload: Excessive preload leading toward seizure.
        cage_noise: Audible cage noise from cage instability.

    Inputs:
        lubrication_film (cc/brg): Lubricant volume delivered to each bearing.

    States:
        bearing_fatigue_life (dimensionless): Remaining fatigue life [1.0 -> 0.0].
        grease_condition_degradation (dimensionless): Grease health [1.0 -> 0.0].
        cage_material_wear (dimensionless): Cage structural integrity [1.0 -> 0.0].
        lubrication_film_tracked (cc/brg): First-order lag tracking of lubrication_film.

    Outputs:
        (none)
    """

    inputs = ["lubrication_film"]
    states = [
        "bearing_fatigue_life",
        "grease_condition_degradation",
        "cage_material_wear",
        "lubrication_film_tracked",
    ]
    outputs = []
    events = [
        "flaking_one_side_raceway",
        "scoring_raceway_rolling_surface",
        "rib_scoring",
        "excessive_preload",
        "cage_noise",
    ]

    units = {
        "lubrication_film": "cc/brg",
        "bearing_fatigue_life": "dimensionless",
        "grease_condition_degradation": "dimensionless",
        "cage_material_wear": "dimensionless",
        "lubrication_film_tracked": "cc/brg",
    }

    default_parameters = {
        # Degradation rate constants (per 1000 operating hours convention)
        "bearing_wear_rate": 0.02,
        "grease_degradation_rate": 0.03,
        "contamination_buildup_rate": 0.01,
        # Event thresholds (fractional degradation)
        "critical_bearing_wear": 0.8,
        "critical_grease_degradation": 0.7,
        "critical_cage_wear": 0.75,
        # Lubrication thresholds
        "min_adequate_lubrication_film": 0.1,
        "lubrication_film_lag_coefficient": 0.3,
        # Cage noise grease threshold
        "cage_noise_grease_threshold": 0.6,
        # State bounds
        "bearing_fatigue_life_min": 0.0,
        "bearing_fatigue_life_max": 1.0,
        "grease_condition_min": 0.0,
        "grease_condition_max": 1.0,
        "cage_material_wear_min": 0.0,
        "cage_material_wear_max": 1.0,
        # Initial state values
        "x0": {
            "bearing_fatigue_life": 1.0,
            "grease_condition_degradation": 1.0,
            "cage_material_wear": 1.0,
            "lubrication_film_tracked": 0.06,
        },
    }

    def initialize(self, u=None, z=None):
        x0 = {
            "bearing_fatigue_life": 1.0,
            "grease_condition_degradation": 1.0,
            "cage_material_wear": 1.0,
            "lubrication_film_tracked": 0.06,
        }
        return self.StateContainer(x0)

    def next_state(self, x, u, dt):
        p = self.parameters

        # --- Retrieve current states ---
        bfl = x["bearing_fatigue_life"]
        gcd = x["grease_condition_degradation"]
        cmw = x["cage_material_wear"]
        lft = x["lubrication_film_tracked"]

        # --- Retrieve input (safe default if None) ---
        lube_film = 0.0
        if u is not None and u["lubrication_film"] is not None:
            lube_film = float(u["lubrication_film"])

        # --- Degradation rate conversion: rates are per 1000 operating hours ---
        bearing_rate = p["bearing_wear_rate"] / 1000.0
        grease_rate = p["grease_degradation_rate"] / 1000.0
        cage_rate = p["contamination_buildup_rate"] / 1000.0

        # --- Drivers ---
        # load_cycles_and_speed_driver: normalized [0,1]; use 1.0 as nominal
        load_driver = 1.0

        # thermal_stress_and_oxidation_driver: normalized [0,1]; use 1.0 as nominal
        thermal_driver = 1.0

        # metallic_wear_particles_driver: coupled to cage wear severity
        # As cage wears (cmw decreases), particle generation increases
        metallic_driver = 1.0 - cmw  # 0.0 when new, 1.0 when fully worn
        # Clamp to avoid zero driver at start (always some baseline wear)
        metallic_driver = max(metallic_driver, 1e-4)

        # --- State transitions ---

        # bearing_fatigue_life: degradation
        new_bfl = bfl - bearing_rate * load_driver * dt
        new_bfl = max(p["bearing_fatigue_life_min"], min(p["bearing_fatigue_life_max"], new_bfl))

        # grease_condition_degradation: degradation
        new_gcd = gcd - grease_rate * thermal_driver * dt
        new_gcd = max(p["grease_condition_min"], min(p["grease_condition_max"], new_gcd))

        # cage_material_wear: degradation
        new_cmw = cmw - cage_rate * metallic_driver * dt
        new_cmw = max(p["cage_material_wear_min"], min(p["cage_material_wear_max"], new_cmw))

        # lubrication_film_tracked: first-order lag tracking
        lag_coeff = min(p["lubrication_film_lag_coefficient"], 0.5)
        new_lft = lft + lag_coeff * (lube_film - lft) * dt
        new_lft = max(0.0, min(77.0, new_lft))

        return self.StateContainer(
            {
                "bearing_fatigue_life": new_bfl,
                "grease_condition_degradation": new_gcd,
                "cage_material_wear": new_cmw,
                "lubrication_film_tracked": new_lft,
            }
        )

    def output(self, x):
        # No output ports defined for this component
        return self.OutputContainer({})

    def event_state(self, x) -> dict:
        p = self.parameters

        bfl = x["bearing_fatigue_life"]
        gcd = x["grease_condition_degradation"]
        cmw = x["cage_material_wear"]
        lft = x["lubrication_film_tracked"]

        # --- Precompute threshold levels ---
        # critical_bearing_wear = 0.8 => event triggers when bfl <= 0.2
        bearing_threshold = 1.0 - p["critical_bearing_wear"]  # 0.2
        # critical_grease_degradation = 0.7 => event triggers when gcd <= 0.3
        grease_threshold = 1.0 - p["critical_grease_degradation"]  # 0.3
        # critical_cage_wear = 0.75 => event triggers when cmw <= 0.25
        cage_threshold = 1.0 - p["critical_cage_wear"]  # 0.25
        # cage_noise_grease_threshold = 0.6 => event triggers when gcd <= 0.4
        cage_noise_grease_threshold = 1.0 - p["cage_noise_grease_threshold"]  # 0.4

        # --- flaking_one_side_raceway ---
        # event_state = (bfl - bearing_threshold) / (1.0 - bearing_threshold)
        denom_bfl = 1.0 - bearing_threshold
        if denom_bfl <= 0.0:
            es_flaking = 0.0 if bfl <= bearing_threshold else 1.0
        else:
            es_flaking = (bfl - bearing_threshold) / denom_bfl
        es_flaking = float(max(0.0, min(1.0, es_flaking)))

        # --- scoring_raceway_rolling_surface ---
        # min(grease_component, lubrication_film_component)
        denom_gcd = 1.0 - grease_threshold
        if denom_gcd <= 0.0:
            es_grease_scoring = 0.0 if gcd <= grease_threshold else 1.0
        else:
            es_grease_scoring = (gcd - grease_threshold) / denom_gcd

        min_lube = p["min_adequate_lubrication_film"]
        if min_lube <= 0.0:
            es_lube_scoring = 1.0
        else:
            es_lube_scoring = (lft - min_lube) / min_lube

        es_scoring = float(max(0.0, min(1.0, min(es_grease_scoring, es_lube_scoring))))

        # --- rib_scoring ---
        # min(grease_component, cage_component)
        denom_cage = 1.0 - cage_threshold
        if denom_cage <= 0.0:
            es_cage_rib = 0.0 if cmw <= cage_threshold else 1.0
        else:
            es_cage_rib = (cmw - cage_threshold) / denom_cage

        es_rib = float(max(0.0, min(1.0, min(es_grease_scoring, es_cage_rib))))

        # --- excessive_preload ---
        # min(bearing_component, cage_component)
        es_preload = float(max(0.0, min(1.0, min(es_flaking, es_cage_rib))))

        # --- cage_noise ---
        # min(cage_component, grease_noise_component)
        denom_cage_noise_grease = 1.0 - cage_noise_grease_threshold
        if denom_cage_noise_grease <= 0.0:
            es_grease_noise = 0.0 if gcd <= cage_noise_grease_threshold else 1.0
        else:
            es_grease_noise = (gcd - cage_noise_grease_threshold) / denom_cage_noise_grease

        es_cage_noise = float(max(0.0, min(1.0, min(es_cage_rib, es_grease_noise))))

        return {
            "flaking_one_side_raceway": es_flaking,
            "scoring_raceway_rolling_surface": es_scoring,
            "rib_scoring": es_rib,
            "excessive_preload": es_preload,
            "cage_noise": es_cage_noise,
        }

    def threshold_met(self, x) -> dict:
        es = self.event_state(x)
        return {
            "flaking_one_side_raceway": bool(es["flaking_one_side_raceway"] <= 0.0),
            "scoring_raceway_rolling_surface": bool(es["scoring_raceway_rolling_surface"] <= 0.0),
            "rib_scoring": bool(es["rib_scoring"] <= 0.0),
            "excessive_preload": bool(es["excessive_preload"] <= 0.0),
            "cage_noise": bool(es["cage_noise"] <= 0.0),
        }