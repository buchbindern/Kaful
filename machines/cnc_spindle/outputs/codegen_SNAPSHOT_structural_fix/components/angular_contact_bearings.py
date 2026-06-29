import numpy as np
from progpy import PrognosticsModel


class AngularContactBearings(PrognosticsModel):
    """
    Prognostics model for angular contact bearings in a precision spindle or axis drive.

    Tracks grease condition, bearing surface wear, cage wear, and ball surface damage.
    Emits vibration velocity, vibration acceleration, and lubrication film outputs.
    """

    inputs = []

    states = [
        "grease_degradation",
        "bearing_wear",
        "cage_wear",
        "ball_surface_damage",
        "contamination_level",
        "bearing_temperature_tracked",
    ]

    outputs = [
        "vibration_velocity",
        "vibration_acceleration",
        "lubrication_film",
    ]

    events = [
        "excessive_temperature",
        "cage_noise",
        "grease_leakage",
        "contamination_ingress",
    ]

    default_parameters = {
        # Degradation rates
        "bearing_wear_rate": 0.02,
        "grease_degradation_rate": 0.03,
        "contamination_buildup_rate": 0.01,
        "cage_wear_rate": 0.02,
        "ball_surface_damage_rate": 0.03,
        # Coupling multipliers
        "contamination_wear_coupling": 2.0,
        "grease_contamination_coupling": 1.5,
        # Temperature model
        "temperature_lag_coefficient": 0.1,
        "temperature_base": 20.0,
        "temperature_wear_gain": 80.0,
        "temperature_grease_gain": 60.0,
        # Thresholds
        "overheat_threshold": 140.0,
        "critical_temperature": 150.0,
        "critical_bearing_wear": 0.8,
        "vibration_smooth_threshold": 1.8,
        "vibration_rough_threshold": 5.4,
        "vibration_critical_threshold": 10.7,
        "acceleration_normal_threshold": 0.2,
        "acceleration_critical_threshold": 0.4,
        # Lubrication film
        "lubrication_film_max": 77.0,
        "lubrication_film_min": 0.06,
        # Vibration output gains
        "vibration_velocity_base": 0.2,
        "vibration_velocity_wear_gain": 10.5,
        "vibration_velocity_cage_gain": 3.0,
        "vibration_acceleration_base": 0.01,
        "vibration_acceleration_ball_gain": 0.25,
        "vibration_acceleration_cage_gain": 0.14,
        # Event thresholds
        "grease_leakage_threshold": 0.2,
        "contamination_ingress_threshold": 0.5,
        "cage_noise_grease_threshold": 0.3,
        "cage_noise_wear_threshold": 0.4,
        # Initial states
        "x0": {
            "grease_degradation": 1.0,
            "bearing_wear": 1.0,
            "cage_wear": 1.0,
            "ball_surface_damage": 1.0,
            "contamination_level": 0.0,
            "bearing_temperature_tracked": 20.0,
        },
    }

    # Units for all states, inputs, and outputs
    units = {
        "grease_degradation": "dimensionless",
        "bearing_wear": "dimensionless",
        "cage_wear": "dimensionless",
        "ball_surface_damage": "dimensionless",
        "contamination_level": "dimensionless",
        "bearing_temperature_tracked": "degC",
        "vibration_velocity": "mm/s",
        "vibration_acceleration": "g",
        "lubrication_film": "cc/brg",
    }

    state_limits = {
        "grease_degradation": (0.0, 1.0),
        "bearing_wear": (0.0, 1.0),
        "cage_wear": (0.0, 1.0),
        "ball_surface_damage": (0.0, 1.0),
        "contamination_level": (0.0, 1.0),
        "bearing_temperature_tracked": (0.0, 300.0),
    }

    def initialize(self, u=None, z=None):
        """Initialize states to healthy defaults."""
        return self.StateContainer({
            "grease_degradation": self.parameters["x0"]["grease_degradation"],
            "bearing_wear": self.parameters["x0"]["bearing_wear"],
            "cage_wear": self.parameters["x0"]["cage_wear"],
            "ball_surface_damage": self.parameters["x0"]["ball_surface_damage"],
            "contamination_level": self.parameters["x0"]["contamination_level"],
            "bearing_temperature_tracked": self.parameters["x0"]["bearing_temperature_tracked"],
        })

    def dx(self, x, u):
        """
        Continuous-time state derivatives.

        All degradation drivers are treated as normalized unit drivers (value = 1.0)
        since this component has no inputs.
        """
        p = self.parameters

        # Normalized unit drivers (no inputs)
        thermal_stress_driver = 1.0
        load_speed_driver = 1.0
        wear_particles_driver = 1.0

        # Current states
        grease_deg = x["grease_degradation"]
        bearing_w = x["bearing_wear"]
        cage_w = x["cage_wear"]
        ball_dmg = x["ball_surface_damage"]
        contam = x["contamination_level"]
        temp_tracked = x["bearing_temperature_tracked"]

        # grease_degradation: decreases under thermal stress, amplified by contamination
        d_grease = -(
            p["grease_degradation_rate"]
            * (1.0 + p["grease_contamination_coupling"] * contam)
            * thermal_stress_driver
        )

        # bearing_wear: decreases under load/speed, amplified by contamination
        d_bearing_wear = -(
            p["bearing_wear_rate"]
            * (1.0 + p["contamination_wear_coupling"] * contam)
            * load_speed_driver
        )

        # cage_wear: decreases under load/speed, amplified by contamination
        d_cage_wear = -(
            p["cage_wear_rate"]
            * (1.0 + p["contamination_wear_coupling"] * contam)
            * load_speed_driver
        )

        # ball_surface_damage: decreases under thermal stress
        d_ball_damage = -(
            p["ball_surface_damage_rate"]
            * thermal_stress_driver
        )

        # contamination_level: increases from metallic wear particles
        d_contamination = (
            p["contamination_buildup_rate"]
            * wear_particles_driver
        )

        # bearing_temperature_tracked: first-order lag toward target temperature
        target_temp = (
            p["temperature_base"]
            + p["temperature_wear_gain"] * (1.0 - bearing_w)
            + p["temperature_grease_gain"] * (1.0 - grease_deg)
        )
        # Cap lag coefficient at 0.5 for numerical stability
        lag_coeff = min(p["temperature_lag_coefficient"], 0.5)
        d_temp = lag_coeff * (target_temp - temp_tracked)

        return self.StateContainer({
            "grease_degradation": d_grease,
            "bearing_wear": d_bearing_wear,
            "cage_wear": d_cage_wear,
            "ball_surface_damage": d_ball_damage,
            "contamination_level": d_contamination,
            "bearing_temperature_tracked": d_temp,
        })

    def output(self, x):
        """Compute observable outputs from current state."""
        p = self.parameters

        grease_deg = x["grease_degradation"]
        bearing_w = x["bearing_wear"]
        cage_w = x["cage_wear"]
        ball_dmg = x["ball_surface_damage"]

        # Vibration velocity: rises as bearing_wear and cage_wear degrade
        vib_vel = (
            p["vibration_velocity_base"]
            + p["vibration_velocity_wear_gain"] * (1.0 - bearing_w)
            + p["vibration_velocity_cage_gain"] * (1.0 - cage_w)
        )
        vib_vel = float(np.clip(vib_vel, 0.0, p["vibration_critical_threshold"]))

        # Vibration acceleration: rises with ball surface damage and cage wear
        vib_acc = (
            p["vibration_acceleration_base"]
            + p["vibration_acceleration_ball_gain"] * (1.0 - ball_dmg)
            + p["vibration_acceleration_cage_gain"] * (1.0 - cage_w)
        )
        vib_acc = float(np.clip(vib_acc, 0.0, p["acceleration_critical_threshold"]))

        # Lubrication film: linear mapping from grease_degradation health index
        lub_film = (
            p["lubrication_film_min"]
            + (p["lubrication_film_max"] - p["lubrication_film_min"]) * grease_deg
        )
        lub_film = float(np.clip(lub_film, p["lubrication_film_min"], p["lubrication_film_max"]))

        return self.OutputContainer({
            "vibration_velocity": vib_vel,
            "vibration_acceleration": vib_acc,
            "lubrication_film": lub_film,
        })

    def event_state(self, x) -> dict:
        """
        Compute event states (1.0 = healthy, 0.0 = event occurred).
        All values must be in [0.0, 1.0].
        """
        p = self.parameters

        grease_deg = x["grease_degradation"]
        bearing_w = x["bearing_wear"]
        cage_w = x["cage_wear"]
        contam = x["contamination_level"]
        temp_tracked = x["bearing_temperature_tracked"]

        # excessive_temperature: falls linearly from overheat_threshold to critical_temperature
        temp_range = p["critical_temperature"] - p["overheat_threshold"]
        temp_excess = (temp_tracked - p["overheat_threshold"]) / temp_range
        excessive_temp_state = float(max(0.0, 1.0 - max(0.0, temp_excess)))

        # cage_noise: min of cage_wear margin and grease margin relative to thresholds
        bearing_health_margin = min(1.0, cage_w / p["cage_noise_wear_threshold"])
        grease_health_margin = min(1.0, grease_deg / p["cage_noise_grease_threshold"])
        cage_noise_state = float(min(bearing_health_margin, grease_health_margin))

        # grease_leakage: falls as grease_degradation drops below grease_leakage_threshold
        grease_leakage_state = float(min(1.0, grease_deg / p["grease_leakage_threshold"]))

        # contamination_ingress: falls as contamination_level rises above threshold
        contamination_ingress_state = float(
            max(0.0, 1.0 - contam / p["contamination_ingress_threshold"])
        )

        return {
            "excessive_temperature": excessive_temp_state,
            "cage_noise": cage_noise_state,
            "grease_leakage": grease_leakage_state,
            "contamination_ingress": contamination_ingress_state,
        }

    def threshold_met(self, x) -> dict:
        """
        Determine if each event threshold has been crossed.
        Returns dict of bool values.
        """
        event_states = self.event_state(x)
        return {
            "excessive_temperature": bool(event_states["excessive_temperature"] <= 0.0),
            "cage_noise": bool(event_states["cage_noise"] <= 0.0),
            "grease_leakage": bool(event_states["grease_leakage"] <= 0.0),
            "contamination_ingress": bool(event_states["contamination_ingress"] <= 0.0),
        }