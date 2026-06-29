from progpy import PrognosticsModel
import numpy as np


class SpindleAmplifier(PrognosticsModel):
    """
    Prognostics model for the spindle amplifier control system.
    Tracks degradation of velocity loop gain stability, torque response,
    and cutting capability, and raises alarms when control deviations
    exceed safe thresholds.
    """

    inputs = [
        "spindle_speed_command",
        "torque_command",
        "motor_voltage",
        "acceleration_deceleration_time",
        "current_command",
        "maximum_motor_speed",
        "motor_power_off_delay",
        "torque_limitation_active",
        "operation_mode",
        "acceleration_value",
        "deceleration_time_constant",
        "position_pulses",
    ]

    states = [
        "velocity_loop_gain_stability",
        "torque_response_degradation",
        "cutting_capability_degradation",
        "contamination_level",
        "tracked_velocity_error",
        "tracked_positional_deviation",
    ]

    outputs = [
        "velocity_error",
        "speed_arrival_signal",
        "load_detection_level_1",
        "speed_arrival_detection",
        "spindle_alarm",
        "alarm_detection_status",
    ]

    events = [
        "velocity_error_excess_alarm",
        "excessive_speed_deviation_alarm",
        "excessive_positional_deviation_alarm",
    ]

    units = {
        # inputs
        "spindle_speed_command": "rpm",
        "torque_command": "",
        "motor_voltage": "",
        "acceleration_deceleration_time": "sec",
        "current_command": "a",
        "maximum_motor_speed": "rpm",
        "motor_power_off_delay": "ms",
        "torque_limitation_active": "boolean",
        "operation_mode": "mode_type",
        "acceleration_value": "rpm/s",
        "deceleration_time_constant": "sec",
        "position_pulses": "pulses/rev",
        # states
        "velocity_loop_gain_stability": "dimensionless",
        "torque_response_degradation": "dimensionless",
        "cutting_capability_degradation": "dimensionless",
        "contamination_level": "dimensionless",
        "tracked_velocity_error": "rpm",
        "tracked_positional_deviation": "pulses",
        # outputs
        "velocity_error": "rpm",
        "speed_arrival_signal": "boolean",
        "load_detection_level_1": "",
        "speed_arrival_detection": "boolean",
        "spindle_alarm": "boolean",
        "alarm_detection_status": "boolean",
    }

    default_parameters = {
        "bearing_wear_rate": 0.02,
        "preload_degradation_rate": 0.015,
        "grease_degradation_rate": 0.03,
        "contamination_buildup_rate": 0.01,
        "velocity_error_lag_coefficient": 0.3,
        "positional_deviation_lag_coefficient": 0.3,
        "speed_arrival_threshold": 15.0,
        "zero_speed_threshold": 0.75,
        "vibration_smooth_threshold": 1.8,
        "vibration_rough_threshold": 5.4,
        "vibration_critical_threshold": 10.7,
        "acceleration_normal_threshold": 0.2,
        "acceleration_critical_threshold": 0.4,
        "critical_bearing_wear": 0.8,
        "critical_temperature": 150.0,
        "overheat_threshold": 140.0,
        "critical_vibration": 10.7,
        "critical_position_error": 2000.0,
        "velocity_error_alarm_threshold": 15.0,
        "speed_deviation_alarm_threshold": 15.0,
        "velocity_loop_stability_alarm_level": 0.2,
        "torque_response_alarm_level": 0.2,
        "cutting_capability_alarm_level": 0.2,
        "contamination_alarm_level": 0.7,
        "max_contamination": 1.0,
        "velocity_error_scale": 256.0,
        "positional_deviation_scale": 4000.0,
        "load_detection_scale": 100.0,
        "x0": {
            "velocity_loop_gain_stability": 1.0,
            "torque_response_degradation": 1.0,
            "cutting_capability_degradation": 1.0,
            "contamination_level": 0.0,
            "tracked_velocity_error": 0.0,
            "tracked_positional_deviation": 0.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "velocity_loop_gain_stability": self.parameters["x0"]["velocity_loop_gain_stability"],
            "torque_response_degradation": self.parameters["x0"]["torque_response_degradation"],
            "cutting_capability_degradation": self.parameters["x0"]["cutting_capability_degradation"],
            "contamination_level": self.parameters["x0"]["contamination_level"],
            "tracked_velocity_error": self.parameters["x0"]["tracked_velocity_error"],
            "tracked_positional_deviation": self.parameters["x0"]["tracked_positional_deviation"],
        })

    def dx(self, x, u):
        params = self.parameters

        # Safe input extraction with defaults
        spindle_speed_command = u["spindle_speed_command"] if u is not None and u["spindle_speed_command"] is not None else 0.0
        torque_command = u["torque_command"] if u is not None and u["torque_command"] is not None else 0.0
        motor_voltage = u["motor_voltage"] if u is not None and u["motor_voltage"] is not None else 0.0
        maximum_motor_speed = u["maximum_motor_speed"] if u is not None and u["maximum_motor_speed"] is not None else 1.0

        # Avoid division by zero
        if maximum_motor_speed == 0.0:
            maximum_motor_speed = 1.0

        # Current states
        vlgs = x["velocity_loop_gain_stability"]
        trd = x["torque_response_degradation"]
        ccd = x["cutting_capability_degradation"]
        cont = x["contamination_level"]
        tve = x["tracked_velocity_error"]
        tpd = x["tracked_positional_deviation"]

        # --- Degradation drivers ---
        # load_cycles_and_speed_driver
        speed_frac = float(spindle_speed_command) / float(maximum_motor_speed)
        torque_frac = float(torque_command) / 100.0
        load_cycles_and_speed_driver = speed_frac * torque_frac

        # thermal_cycling_and_wear_driver
        thermal_cycling_and_wear_driver = torque_frac * (1.0 + float(cont))

        # thermal_stress_and_oxidation_driver
        voltage_frac = float(motor_voltage) / 100.0
        thermal_stress_and_oxidation_driver = voltage_frac * torque_frac

        # metallic_wear_particles_driver
        metallic_wear_particles_driver = (1.0 - float(vlgs)) * torque_frac

        # --- Rates (per 1000 operating hours convention -> divide by 1000) ---
        bearing_wear_rate = params["bearing_wear_rate"] / 1000.0
        preload_degradation_rate = params["preload_degradation_rate"] / 1000.0
        grease_degradation_rate = params["grease_degradation_rate"] / 1000.0
        contamination_buildup_rate = params["contamination_buildup_rate"] / 1000.0

        # --- State derivatives ---
        # Degradation states (decrease)
        d_vlgs = -bearing_wear_rate * load_cycles_and_speed_driver
        d_trd = -preload_degradation_rate * thermal_cycling_and_wear_driver
        d_ccd = -grease_degradation_rate * thermal_stress_and_oxidation_driver

        # Accumulation state (increase)
        d_cont = contamination_buildup_rate * metallic_wear_particles_driver

        # Tracking states (first-order lag)
        vel_lag = min(params["velocity_error_lag_coefficient"], 0.5)
        pos_lag = min(params["positional_deviation_lag_coefficient"], 0.5)

        # target_velocity_error
        sign_speed = 1.0 if float(spindle_speed_command) > 0 else (-1.0 if float(spindle_speed_command) < 0 else 0.0)
        target_velocity_error = (1.0 - float(vlgs)) * params["velocity_error_scale"] * sign_speed

        # target_positional_deviation
        target_positional_deviation = (1.0 - float(ccd)) * float(cont) * params["positional_deviation_scale"]

        d_tve = vel_lag * (target_velocity_error - float(tve))
        d_tpd = pos_lag * (target_positional_deviation - float(tpd))

        return self.StateContainer({
            "velocity_loop_gain_stability": d_vlgs,
            "torque_response_degradation": d_trd,
            "cutting_capability_degradation": d_ccd,
            "contamination_level": d_cont,
            "tracked_velocity_error": d_tve,
            "tracked_positional_deviation": d_tpd,
        })

    def next_state(self, x, u, dt):
        # Use dx and apply Euler integration with clamping
        dxdt = self.dx(x, u)

        params = self.parameters

        vlgs = float(x["velocity_loop_gain_stability"]) + float(dxdt["velocity_loop_gain_stability"]) * dt
        trd = float(x["torque_response_degradation"]) + float(dxdt["torque_response_degradation"]) * dt
        ccd = float(x["cutting_capability_degradation"]) + float(dxdt["cutting_capability_degradation"]) * dt
        cont = float(x["contamination_level"]) + float(dxdt["contamination_level"]) * dt
        tve = float(x["tracked_velocity_error"]) + float(dxdt["tracked_velocity_error"]) * dt
        tpd = float(x["tracked_positional_deviation"]) + float(dxdt["tracked_positional_deviation"]) * dt

        # Clamp
        vlgs = max(0.0, min(1.0, vlgs))
        trd = max(0.0, min(1.0, trd))
        ccd = max(0.0, min(1.0, ccd))
        cont = max(0.0, min(float(params["max_contamination"]), cont))
        tve = max(-256.0, min(256.0, tve))
        tpd = max(-4000.0, min(4000.0, tpd))

        return self.StateContainer({
            "velocity_loop_gain_stability": vlgs,
            "torque_response_degradation": trd,
            "cutting_capability_degradation": ccd,
            "contamination_level": cont,
            "tracked_velocity_error": tve,
            "tracked_positional_deviation": tpd,
        })

    def output(self, x):
        params = self.parameters

        tve = float(x["tracked_velocity_error"])
        tpd = float(x["tracked_positional_deviation"])
        trd = float(x["torque_response_degradation"])
        vlgs = float(x["velocity_loop_gain_stability"])
        ccd = float(x["cutting_capability_degradation"])
        cont = float(x["contamination_level"])

        # velocity_error: tracked_velocity_error directly
        velocity_error = tve

        # speed_arrival_signal: 1 if |tve| <= threshold AND spindle_speed_command > 0
        # We don't have u here, so we use the state proxy: if tve is near 0 and vlgs is healthy
        # Per spec: output logic uses tracked_velocity_error and speed_arrival_threshold
        # speed_arrival_signal uses spindle_speed_command > 0 — but output() only receives x
        # We approximate: speed arrival is 1 if |tve| <= speed_arrival_threshold
        # (spindle_speed_command > 0 cannot be checked here; use tve != 0 as proxy for active)
        speed_arrival_threshold = params["speed_arrival_threshold"]
        speed_arrival = 1.0 if abs(tve) <= speed_arrival_threshold else 0.0

        # load_detection_level_1
        load_detection = float(0.0)
        # torque_command not available in output; use degradation-based estimate
        # Per spec: torque_command * (1.0 + (1.0 - trd) * load_detection_scale / 100.0)
        # Since torque_command is not available in output(), we output the degradation factor
        # scaled to [0, 100]: (1.0 - trd) * load_detection_scale
        load_detection = (1.0 - trd) * params["load_detection_scale"]
        load_detection = max(0.0, min(100.0, load_detection))

        # speed_arrival_detection: same as speed_arrival_signal
        speed_arrival_detection = speed_arrival

        # spindle_alarm: any event active
        tm = self.threshold_met(x)
        any_alarm = (
            tm["velocity_error_excess_alarm"]
            or tm["excessive_speed_deviation_alarm"]
            or tm["excessive_positional_deviation_alarm"]
        )
        spindle_alarm = 1.0 if any_alarm else 0.0

        # alarm_detection_status: mirrors spindle_alarm
        alarm_detection_status = spindle_alarm

        return self.OutputContainer({
            "velocity_error": velocity_error,
            "speed_arrival_signal": speed_arrival,
            "load_detection_level_1": load_detection,
            "speed_arrival_detection": speed_arrival_detection,
            "spindle_alarm": spindle_alarm,
            "alarm_detection_status": alarm_detection_status,
        })

    def event_state(self, x) -> dict:
        params = self.parameters

        vlgs = float(x["velocity_loop_gain_stability"])
        trd = float(x["torque_response_degradation"])
        ccd = float(x["cutting_capability_degradation"])
        cont = float(x["contamination_level"])
        tve = float(x["tracked_velocity_error"])
        tpd = float(x["tracked_positional_deviation"])

        # --- velocity_error_excess_alarm ---
        # Triggers when vlgs <= velocity_loop_stability_alarm_level AND |tve| >= velocity_error_alarm_threshold
        vlgs_alarm_level = params["velocity_loop_stability_alarm_level"]
        vel_err_threshold = params["velocity_error_alarm_threshold"]

        # Health component: how far vlgs is above alarm level (1.0 when healthy, 0.0 at alarm)
        vlgs_health = min(1.0, max(0.0, (vlgs - vlgs_alarm_level) / (1.0 - vlgs_alarm_level + 1e-9)))
        # Error component: how far |tve| is below threshold (1.0 when no error, 0.0 at threshold)
        tve_health = min(1.0, max(0.0, 1.0 - abs(tve) / (vel_err_threshold + 1e-9)))
        # Event state is 1.0 when healthy, 0.0 when both conditions met
        velocity_error_excess_alarm_es = max(vlgs_health, tve_health)

        # --- excessive_speed_deviation_alarm ---
        trd_alarm_level = params["torque_response_alarm_level"]
        spd_dev_threshold = params["speed_deviation_alarm_threshold"]

        trd_health = min(1.0, max(0.0, (trd - trd_alarm_level) / (1.0 - trd_alarm_level + 1e-9)))
        tve_health2 = min(1.0, max(0.0, 1.0 - abs(tve) / (spd_dev_threshold + 1e-9)))
        # All three must be at alarm level for event to trigger
        excessive_speed_deviation_alarm_es = max(vlgs_health, trd_health, tve_health2)

        # --- excessive_positional_deviation_alarm ---
        ccd_alarm_level = params["cutting_capability_alarm_level"]
        cont_alarm_level = params["contamination_alarm_level"]
        crit_pos_err = params["critical_position_error"]

        ccd_health = min(1.0, max(0.0, (ccd - ccd_alarm_level) / (1.0 - ccd_alarm_level + 1e-9)))
        # contamination: event needs cont >= contamination_alarm_level; health = 1 when cont < alarm
        cont_health = min(1.0, max(0.0, 1.0 - cont / (cont_alarm_level + 1e-9)))
        tpd_health = min(1.0, max(0.0, 1.0 - abs(tpd) / (crit_pos_err + 1e-9)))
        excessive_positional_deviation_alarm_es = max(ccd_health, cont_health, tpd_health)

        return {
            "velocity_error_excess_alarm": float(velocity_error_excess_alarm_es),
            "excessive_speed_deviation_alarm": float(excessive_speed_deviation_alarm_es),
            "excessive_positional_deviation_alarm": float(excessive_positional_deviation_alarm_es),
        }

    def threshold_met(self, x) -> dict:
        params = self.parameters

        vlgs = float(x["velocity_loop_gain_stability"])
        trd = float(x["torque_response_degradation"])
        ccd = float(x["cutting_capability_degradation"])
        cont = float(x["contamination_level"])
        tve = float(x["tracked_velocity_error"])
        tpd = float(x["tracked_positional_deviation"])

        # velocity_error_excess_alarm
        vel_err_alarm = bool(
            vlgs <= params["velocity_loop_stability_alarm_level"]
            and abs(tve) >= params["velocity_error_alarm_threshold"]
        )

        # excessive_speed_deviation_alarm
        spd_dev_alarm = bool(
            vlgs <= params["velocity_loop_stability_alarm_level"]
            and trd <= params["torque_response_alarm_level"]
            and abs(tve) >= params["speed_deviation_alarm_threshold"]
        )

        # excessive_positional_deviation_alarm
        pos_dev_alarm = bool(
            ccd <= params["cutting_capability_alarm_level"]
            and cont >= params["contamination_alarm_level"]
            and abs(tpd) >= params["critical_position_error"]
        )

        return {
            "velocity_error_excess_alarm": vel_err_alarm,
            "excessive_speed_deviation_alarm": spd_dev_alarm,
            "excessive_positional_deviation_alarm": pos_dev_alarm,
        }