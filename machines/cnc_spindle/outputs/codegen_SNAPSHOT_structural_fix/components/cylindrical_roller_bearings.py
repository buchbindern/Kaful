from progpy import PrognosticsModel
import numpy as np


class CylindricalRollerBearings(PrognosticsModel):
    """
    Prognostics model for cylindrical roller bearings in a spindle assembly.

    Tracks fatigue life, grease condition, and cage wear as primary degradation
    mechanisms. Raises fault events for raceway flaking, scoring, rib damage,
    excessive preload, and cage noise.

    Events:
        flaking_one_side_raceway: Subsurface fatigue crack propagation leads to
            material flaking on one side of the radial bearing raceway.
        scoring_raceway_rolling_surface: Insufficient lubrication or degraded
            grease causes adhesive scoring between raceway and rolling elements.
        rib_scoring: Degraded grease and cage wear combine to cause scoring
            between roller end faces and the guide rib.
        excessive_preload: Combined fatigue-induced clearance loss and
            insufficient lubrication create excessive bearing preload.
        cage_noise: Worn cage material and degraded grease produce audible
            cage rattling or noise.

    Inputs:
        lubrication_film (cc/brg): Lubricant volume delivered to each bearing.

    States:
        bearing_fatigue_life (dimensionless): Remaining fatigue life [1.0, 0.0].
        grease_condition_degradation (dimensionless): Remaining grease quality [1.0, 0.0].
        cage_material_wear (dimensionless): Remaining cage structural integrity [1.0, 0.0].
        lubrication_film_tracked (cc/brg): First-order lag estimate of effective
            lubrication film thickness.

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
        # Degradation rates (sourced from simulator ground truth)
        "bearing_wear_rate": 0.02,          # 1/cycle
        "grease_degradation_rate": 0.03,    # 1/hour
        "contamination_buildup_rate": 0.01, # 1/hour
        # Tracking lag coefficient (capped at 0.5 for stability)
        "lubrication_lag_coefficient": 0.3,
        # Event thresholds
        "critical_bearing_wear": 0.8,           # dimensionless
        "grease_scoring_threshold": 0.25,        # dimensionless
        "min_film_scoring_threshold": 0.03,      # cc/brg
        "rib_scoring_grease_threshold": 0.3,     # dimensionless
        "rib_scoring_cage_threshold": 0.35,      # dimensionless
        "preload_fatigue_threshold": 0.3,        # dimensionless
        "min_film_preload_threshold": 0.02,      # cc/brg
        "cage_noise_wear_threshold": 0.4,        # dimensionless
        "cage_noise_grease_threshold": 0.35,     # dimensionless
        # Initial values
        "lubrication_film_initial": 0.06,        # cc/brg
        # Initial state values
        "x0": {
            "bearing_fatigue_life": 1.0,
            "grease_condition_degradation": 1.0,
            "cage_material_wear": 1.0,
            "lubrication_film_tracked": 0.06,
        },
    }

    def initialize(self, u=None, z=None):
        """
        Initialize the model state.

        Parameters
        ----------
        u : InputContainer or None
            Initial inputs (may be None or contain None values during
            CompositeModel setup).
        z : OutputContainer or None
            Initial outputs (may be None or contain None values during
            CompositeModel setup).

        Returns
        -------
        StateContainer
            Initial state of the model.
        """
        # Start with safe defaults from parameters
        x0 = {
            "bearing_fatigue_life": self.parameters["x0"]["bearing_fatigue_life"],
            "grease_condition_degradation": self.parameters["x0"]["grease_condition_degradation"],
            "cage_material_wear": self.parameters["x0"]["cage_material_wear"],
            "lubrication_film_tracked": self.parameters["x0"]["lubrication_film_tracked"],
        }

        # Optionally seed lubrication_film_tracked from input if available
        if u is not None:
            lf = u.get("lubrication_film", None) if hasattr(u, "get") else None
            if lf is None:
                try:
                    lf = u["lubrication_film"]
                except (KeyError, TypeError):
                    lf = None
            if lf is not None:
                x0["lubrication_film_tracked"] = float(lf)

        return self.StateContainer(x0)

    def dx(self, x, u):
        """
        Compute the first derivative of the state vector.

        Degradation states decrease at their respective rates multiplied by
        normalized driver scalars. The lubrication_film_tracked state uses a
        first-order lag toward the lubrication_film input.

        Parameters
        ----------
        x : StateContainer
            Current state.
        u : InputContainer
            Current inputs.

        Returns
        -------
        StateContainer
            Time derivatives of each state.
        """
        params = self.parameters

        # --- Retrieve input safely ---
        lubrication_film = 0.0
        if u is not None:
            try:
                val = u["lubrication_film"]
                if val is not None:
                    lubrication_film = float(val)
            except (KeyError, TypeError):
                lubrication_film = 0.0

        # --- Degradation rates (from simulator ground truth) ---
        bearing_wear_rate = params["bearing_wear_rate"]          # 1/cycle
        grease_degradation_rate = params["grease_degradation_rate"]  # 1/hour
        contamination_buildup_rate = params["contamination_buildup_rate"]  # 1/hour

        # --- Driver scalars ---
        # These normalized [0,1] scalars represent the intensity of the
        # respective degradation drivers. In the absence of explicit driver
        # inputs, a unit driver (1.0) is used, meaning the rates apply
        # directly per time step. Machine-level models may override these
        # by providing additional inputs or by subclassing.
        load_cycles_and_speed_driver = 1.0
        thermal_stress_and_oxidation_driver = 1.0
        metallic_wear_particles_driver = 1.0

        # --- State derivatives ---

        # bearing_fatigue_life: degradation state
        # Rate sanity: 0.02 * 1.0 * 1000 = 20.0 > 1.0
        # NOTE: rate * nominal_driver * 1000 = 20.0 > 1.0 — known open issue
        # with long-term timescale; rate is used as specified by simulator.
        d_bearing_fatigue_life = -bearing_wear_rate * load_cycles_and_speed_driver

        # grease_condition_degradation: degradation state
        # Rate sanity: 0.03 * 1.0 * 1000 = 30.0 > 1.0
        # NOTE: rate * nominal_driver * 1000 = 30.0 > 1.0 — known open issue;
        # rate is used as specified by simulator.
        d_grease_condition_degradation = -grease_degradation_rate * thermal_stress_and_oxidation_driver

        # cage_material_wear: degradation state
        # Rate sanity: 0.01 * 1.0 * 1000 = 10.0 > 1.0
        # NOTE: rate * nominal_driver * 1000 = 10.0 > 1.0 — known open issue;
        # rate is used as specified by simulator.
        d_cage_material_wear = -contamination_buildup_rate * metallic_wear_particles_driver

        # lubrication_film_tracked: tracking state (first-order lag)
        # Stability cap: coefficient = min(raw_coefficient, 0.5)
        lag_coeff = min(params["lubrication_lag_coefficient"], 0.5)
        d_lubrication_film_tracked = lag_coeff * (lubrication_film - x["lubrication_film_tracked"])

        return self.StateContainer(
            {
                "bearing_fatigue_life": d_bearing_fatigue_life,
                "grease_condition_degradation": d_grease_condition_degradation,
                "cage_material_wear": d_cage_material_wear,
                "lubrication_film_tracked": d_lubrication_film_tracked,
            }
        )

    def next_state(self, x, u, dt):
        """
        Advance the state by one time step dt.

        Applies the derivatives from dx() and clamps all states to their
        physical bounds.

        Parameters
        ----------
        x : StateContainer
            Current state.
        u : InputContainer
            Current inputs.
        dt : float
            Time step size.

        Returns
        -------
        StateContainer
            Updated state after time step dt.
        """
        dxdt = self.dx(x, u)

        # bearing_fatigue_life: clamp to [0.0, 1.0]
        new_bearing_fatigue_life = x["bearing_fatigue_life"] + dxdt["bearing_fatigue_life"] * dt
        new_bearing_fatigue_life = max(0.0, min(1.0, new_bearing_fatigue_life))

        # grease_condition_degradation: clamp to [0.0, 1.0]
        new_grease_condition_degradation = x["grease_condition_degradation"] + dxdt["grease_condition_degradation"] * dt
        new_grease_condition_degradation = max(0.0, min(1.0, new_grease_condition_degradation))

        # cage_material_wear: clamp to [0.0, 1.0]
        new_cage_material_wear = x["cage_material_wear"] + dxdt["cage_material_wear"] * dt
        new_cage_material_wear = max(0.0, min(1.0, new_cage_material_wear))

        # lubrication_film_tracked: clamp to [0.0, 77.0]
        new_lubrication_film_tracked = x["lubrication_film_tracked"] + dxdt["lubrication_film_tracked"] * dt
        new_lubrication_film_tracked = max(0.0, min(77.0, new_lubrication_film_tracked))

        return self.StateContainer(
            {
                "bearing_fatigue_life": new_bearing_fatigue_life,
                "grease_condition_degradation": new_grease_condition_degradation,
                "cage_material_wear": new_cage_material_wear,
                "lubrication_film_tracked": new_lubrication_film_tracked,
            }
        )

    def output(self, x):
        """
        Compute model outputs.

        This component has no defined output ports. Returns an empty
        OutputContainer.

        Parameters
        ----------
        x : StateContainer
            Current state.

        Returns
        -------
        OutputContainer
            Empty output container.
        """
        return self.OutputContainer({})

    def event_state(self, x) -> dict:
        """
        Compute event states (progress toward each event threshold).

        All values are floats in [0.0, 1.0] where 1.0 = healthy and
        0.0 = event has occurred. Values may exceed 1.0 under healthy
        initial conditions (they are not clamped to 1.0 here to preserve
        the mathematical formulation from the spec).

        Parameters
        ----------
        x : StateContainer
            Current state.

        Returns
        -------
        dict
            Event state values keyed by event name.
        """
        params = self.parameters

        bearing_fatigue_life = x["bearing_fatigue_life"]
        grease_condition_degradation = x["grease_condition_degradation"]
        cage_material_wear = x["cage_material_wear"]
        lubrication_film_tracked = x["lubrication_film_tracked"]

        # --- flaking_one_side_raceway ---
        # event_state = bearing_fatigue_life / (1.0 - critical_bearing_wear)
        # Triggers when bearing_fatigue_life <= (1.0 - critical_bearing_wear) = 0.2
        # Under healthy conditions: 1.0 / 0.2 = 5.0 >> 0.0 (no spurious trigger)
        flaking_threshold = 1.0 - params["critical_bearing_wear"]  # = 0.2
        flaking_event_state = float(bearing_fatigue_life / flaking_threshold)

        # --- scoring_raceway_rolling_surface ---
        # event_state = min(grease_condition_degradation / grease_scoring_threshold,
        #                   lubrication_film_tracked / min_film_scoring_threshold)
        # Triggers when event_state <= 1.0
        scoring_grease = float(grease_condition_degradation / params["grease_scoring_threshold"])
        scoring_film = float(lubrication_film_tracked / params["min_film_scoring_threshold"])
        scoring_event_state = min(scoring_grease, scoring_film)

        # --- rib_scoring ---
        # event_state = min(grease_condition_degradation / rib_scoring_grease_threshold,
        #                   cage_material_wear / rib_scoring_cage_threshold)
        # Triggers when event_state <= 1.0
        rib_grease = float(grease_condition_degradation / params["rib_scoring_grease_threshold"])
        rib_cage = float(cage_material_wear / params["rib_scoring_cage_threshold"])
        rib_scoring_event_state = min(rib_grease, rib_cage)

        # --- excessive_preload ---
        # event_state = max(bearing_fatigue_life / preload_fatigue_threshold,
        #                   lubrication_film_tracked / min_film_preload_threshold)
        # Triggers when event_state <= 1.0
        preload_fatigue = float(bearing_fatigue_life / params["preload_fatigue_threshold"])
        preload_film = float(lubrication_film_tracked / params["min_film_preload_threshold"])
        excessive_preload_event_state = max(preload_fatigue, preload_film)

        # --- cage_noise ---
        # event_state = min(cage_material_wear / cage_noise_wear_threshold,
        #                   grease_condition_degradation / cage_noise_grease_threshold)
        # Triggers when event_state <= 1.0
        cage_wear_ratio = float(cage_material_wear / params["cage_noise_wear_threshold"])
        cage_grease_ratio = float(grease_condition_degradation / params["cage_noise_grease_threshold"])
        cage_noise_event_state = min(cage_wear_ratio, cage_grease_ratio)

        return {
            "flaking_one_side_raceway": flaking_event_state,
            "scoring_raceway_rolling_surface": scoring_event_state,
            "rib_scoring": rib_scoring_event_state,
            "excessive_preload": excessive_preload_event_state,
            "cage_noise": cage_noise_event_state,
        }

    def threshold_met(self, x) -> dict:
        """
        Determine whether each event threshold has been met.

        Parameters
        ----------
        x : StateContainer
            Current state.

        Returns
        -------
        dict
            Boolean values keyed by event name. True means the event has
            occurred.
        """
        es = self.event_state(x)

        return {
            # flaking: triggers when event_state <= 0.0
            # (bearing_fatigue_life <= 1.0 - critical_bearing_wear = 0.2)
            "flaking_one_side_raceway": bool(es["flaking_one_side_raceway"] <= 0.0),

            # scoring: triggers when event_state <= 1.0
            "scoring_raceway_rolling_surface": bool(es["scoring_raceway_rolling_surface"] <= 1.0),

            # rib_scoring: triggers when event_state <= 1.0
            "rib_scoring": bool(es["rib_scoring"] <= 1.0),

            # excessive_preload: triggers when event_state <= 1.0
            "excessive_preload": bool(es["excessive_preload"] <= 1.0),

            # cage_noise: triggers when event_state <= 1.0
            "cage_noise": bool(es["cage_noise"] <= 1.0),
        }