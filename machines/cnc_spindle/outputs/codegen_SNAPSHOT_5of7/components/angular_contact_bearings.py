import numpy as np
from progpy import PrognosticsModel


class AngularContactBearings(PrognosticsModel):
    """
    PrognosticsModel for angular contact bearings in a precision spindle or axis drive.

    Tracks grease condition, bearing surface wear, cage wear, and ball surface damage
    to predict vibration outputs and lubrication film thickness.
    """

    inputs = []

    states = [
        "grease_degradation",
        "bearing_surface_wear",
        "cage_wear",
        "ball_surface_damage",
        "bearing_temperature_tracked",
        "vibration_velocity_tracked",
        "vibration_acceleration_tracked",
        "lubrication_film_tracked",
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

    units = {
        # states
        "grease_degradation": "dimensionless",
        "bearing_surface_wear": "dimensionless",
        "cage_wear": "dimensionless",
        "ball_surface_damage": "dimensionless",
        "bearing_temperature_tracked": "degC",
        "vibration_velocity_tracked": "mm/s",
        "vibration_acceleration_tracked": "g",
        "lubrication_film_tracked": "cc/brg",
        # outputs
        "vibration_velocity": "mm/s",
        "vibration_acceleration": "g",
        "lubrication_film": "cc/brg",
    }

    default_parameters = {
        # degradation rates (per timestep, per unit driver)
        "grease_degradation_rate": 0.03,
        "bearing_wear_rate": 0.02,
        "preload_degradation_rate": 0.015,
        "contamination_buildup_rate": 0.01,
        # temperature thresholds
        "overheat_threshold": 140.0,
        "critical_temperature": 150.0,
        # bearing wear threshold
        "critical_bearing_wear": 0.8,
        # vibration velocity thresholds
        "vibration_smooth_threshold": 1.8,
        "vibration_rough_threshold": 5.4,
        "vibration_critical_threshold": 10.7,
        # vibration acceleration thresholds
        "acceleration_normal_threshold": 0.2,
        "acceleration_critical_threshold": 0.4,
        # output gains
        "vibration_velocity_base": 0.1,
        "vibration_velocity_wear_gain": 8.0,
        "vibration_velocity_grease_gain": 2.0,
        "vibration_acceleration_base": 0.01,
        "vibration_acceleration_ball_gain": 0.25,
        "vibration_acceleration_cage_gain": 0.14,
        # lubrication film
        "lubrication_film_max": 77.0,
        "lubrication_film_min": 0.06,
        # temperature model
        "temperature_base": 20.0,
        "temperature_wear_gain": 80.0,
        "temperature_grease_gain": 60.0,
        # tracking coefficients
        "temperature_tracking_coeff": 0.3,
        "vibration_velocity_tracking_coeff": 0.4,
        "vibration_acceleration_tracking_coeff": 0.4,
        "lubrication_film_tracking_coeff": 0.3,
        # event thresholds
        "grease_leakage_threshold": 0.3,
        "lubrication_film_low_threshold": 5.0,
        "cage_noise_wear_threshold": 0.5,
        "contamination_damage_threshold": 0.5,
        # driver values
        "thermal_stress_driver_value": 1.0,
        "load_cycles_driver_value": 1.0,
        "metallic_wear_driver_value": 1.0,
        # initial state values
        "x0": {
            "grease_degradation": 1.0,
            "bearing_surface_wear": 1.0,
            "cage_wear": 1.0,
            "ball_surface_damage": 1.0,
            "bearing_temperature_tracked": 20.0,
            "vibration_velocity_tracked": 0.0,
            "vibration_acceleration_tracked": 0.0,
            "lubrication_film_tracked": 77.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "grease_degradation": 1.0,
            "bearing_surface_wear": 1.0,
            "cage_wear": 1.0,
            "ball_surface_damage": 1.0,
            "bearing_temperature_tracked": 20.0,
            "vibration_velocity_tracked": 0.0,
            "vibration_acceleration_tracked": 0.0,
            "lubrication_film_tracked": 77.0,
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # --- DEGRADATION STATES ---
        # grease_degradation: driven by thermal_stress_and_oxidation
        grease_deg_rate = p["grease_degradation_rate"] / 1000.0
        new_grease = x["grease_degradation"] - grease_deg_rate * p["thermal_stress_driver_value"] * dt
        new_grease = max(0.0, min(1.0, new_grease))

        # bearing_surface_wear: driven by load_cycles_and_speed
        bearing_wear_rate = p["bearing_wear_rate"] / 1000.0
        new_bearing_wear = x["bearing_surface_wear"] - bearing_wear_rate * p["load_cycles_driver_value"] * dt
        new_bearing_wear = max(0.0, min(1.0, new_bearing_wear))

        # cage_wear: driven by thermal_cycling_and_wear (preload_degradation_rate)
        cage_wear_rate = p["preload_degradation_rate"] / 1000.0
        new_cage_wear = x["cage_wear"] - cage_wear_rate * p["thermal_stress_driver_value"] * dt
        new_cage_wear = max(0.0, min(1.0, new_cage_wear))

        # ball_surface_damage: driven by metallic_wear_particles
        ball_damage_rate = p["contamination_buildup_rate"] / 1000.0
        new_ball_damage = x["ball_surface_damage"] - ball_damage_rate * p["metallic_wear_driver_value"] * dt
        new_ball_damage = max(0.0, min(1.0, new_ball_damage))

        # --- TRACKING STATES ---

        # bearing_temperature_tracked
        target_temperature = (
            p["temperature_base"]
            + p["temperature_wear_gain"] * (1.0 - new_bearing_wear)
            + p["temperature_grease_gain"] * (1.0 - new_grease)
        )
        temp_coeff = min(p["temperature_tracking_coeff"], 0.5)
        new_temp = x["bearing_temperature_tracked"] + temp_coeff * (target_temperature - x["bearing_temperature_tracked"]) * dt
        new_temp = max(0.0, min(200.0, new_temp))

        # vibration_velocity_tracked
        target_vib_vel = (
            p["vibration_velocity_base"]
            + p["vibration_velocity_wear_gain"] * (1.0 - new_bearing_wear)
            + p["vibration_velocity_grease_gain"] * (1.0 - new_grease)
        )
        vv_coeff = min(p["vibration_velocity_tracking_coeff"], 0.5)
        new_vib_vel = x["vibration_velocity_tracked"] + vv_coeff * (target_vib_vel - x["vibration_velocity_tracked"]) * dt
        new_vib_vel = max(0.0, min(15.0, new_vib_vel))

        # vibration_acceleration_tracked
        target_vib_acc = (
            p["vibration_acceleration_base"]
            + p["vibration_acceleration_ball_gain"] * (1.0 - new_ball_damage)
            + p["vibration_acceleration_cage_gain"] * (1.0 - new_cage_wear)
        )
        va_coeff = min(p["vibration_acceleration_tracking_coeff"], 0.5)
        new_vib_acc = x["vibration_acceleration_tracked"] + va_coeff * (target_vib_acc - x["vibration_acceleration_tracked"]) * dt
        new_vib_acc = max(0.0, min(0.6, new_vib_acc))

        # lubrication_film_tracked
        target_lub_film = (
            p["lubrication_film_min"]
            + (p["lubrication_film_max"] - p["lubrication_film_min"]) * new_grease
        )
        lf_coeff = min(p["lubrication_film_tracking_coeff"], 0.5)
        new_lub_film = x["lubrication_film_tracked"] + lf_coeff * (target_lub_film - x["lubrication_film_tracked"]) * dt
        new_lub_film = max(0.0, min(77.0, new_lub_film))

        return self.StateContainer({
            "grease_degradation": new_grease,
            "bearing_surface_wear": new_bearing_wear,
            "cage_wear": new_cage_wear,
            "ball_surface_damage": new_ball_damage,
            "bearing_temperature_tracked": new_temp,
            "vibration_velocity_tracked": new_vib_vel,
            "vibration_acceleration_tracked": new_vib_acc,
            "lubrication_film_tracked": new_lub_film,
        })

    def output(self, x):
        return self.OutputContainer({
            "vibration_velocity": x["vibration_velocity_tracked"],
            "vibration_acceleration": x["vibration_acceleration_tracked"],
            "lubrication_film": x["lubrication_film_tracked"],
        })

    def event_state(self, x) -> dict:
        p = self.parameters

        # excessive_temperature: 1.0 when temp < overheat_threshold, 0.0 when >= threshold
        temp = x["bearing_temperature_tracked"]
        overheat = p["overheat_threshold"]
        if temp < overheat:
            excessive_temperature_es = 1.0
        else:
            excessive_temperature_es = 0.0

        # cage_noise: 1.0 when cage_wear > threshold AND vib_acc < acceleration_normal_threshold
        cage_w = x["cage_wear"]
        vib_acc = x["vibration_acceleration_tracked"]
        cage_noise_wear_thr = p["cage_noise_wear_threshold"]
        acc_normal_thr = p["acceleration_normal_threshold"]
        if cage_w > cage_noise_wear_thr and vib_acc < acc_normal_thr:
            cage_noise_es = 1.0
        else:
            cage_noise_es = 0.0

        # grease_leakage: 1.0 when grease_degradation > threshold AND lub_film > low_threshold
        grease = x["grease_degradation"]
        lub_film = x["lubrication_film_tracked"]
        grease_leak_thr = p["grease_leakage_threshold"]
        lub_low_thr = p["lubrication_film_low_threshold"]
        if grease > grease_leak_thr and lub_film > lub_low_thr:
            grease_leakage_es = 1.0
        else:
            grease_leakage_es = 0.0

        # contamination_ingress: 1.0 when ball_surface_damage > threshold AND vib_vel < rough_threshold
        ball_dmg = x["ball_surface_damage"]
        vib_vel = x["vibration_velocity_tracked"]
        contam_thr = p["contamination_damage_threshold"]
        rough_thr = p["vibration_rough_threshold"]
        if ball_dmg > contam_thr and vib_vel < rough_thr:
            contamination_ingress_es = 1.0
        else:
            contamination_ingress_es = 0.0

        return {
            "excessive_temperature": excessive_temperature_es,
            "cage_noise": cage_noise_es,
            "grease_leakage": grease_leakage_es,
            "contamination_ingress": contamination_ingress_es,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        # excessive_temperature: threshold met when temp >= overheat_threshold
        excessive_temperature_met = bool(x["bearing_temperature_tracked"] >= p["overheat_threshold"])

        # cage_noise: threshold met when cage_wear <= cage_noise_wear_threshold
        #             OR vibration_acceleration_tracked >= acceleration_normal_threshold
        cage_noise_met = bool(
            x["cage_wear"] <= p["cage_noise_wear_threshold"]
            or x["vibration_acceleration_tracked"] >= p["acceleration_normal_threshold"]
        )

        # grease_leakage: threshold met when grease_degradation <= grease_leakage_threshold
        #                 OR lubrication_film_tracked <= lubrication_film_low_threshold
        grease_leakage_met = bool(
            x["grease_degradation"] <= p["grease_leakage_threshold"]
            or x["lubrication_film_tracked"] <= p["lubrication_film_low_threshold"]
        )

        # contamination_ingress: threshold met when ball_surface_damage <= contamination_damage_threshold
        #                        OR vibration_velocity_tracked >= vibration_rough_threshold
        contamination_ingress_met = bool(
            x["ball_surface_damage"] <= p["contamination_damage_threshold"]
            or x["vibration_velocity_tracked"] >= p["vibration_rough_threshold"]
        )

        return {
            "excessive_temperature": excessive_temperature_met,
            "cage_noise": cage_noise_met,
            "grease_leakage": grease_leakage_met,
            "contamination_ingress": contamination_ingress_met,
        }