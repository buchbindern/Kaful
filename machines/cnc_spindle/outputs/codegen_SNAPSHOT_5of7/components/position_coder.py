from progpy import PrognosticsModel
import numpy as np


class PositionCoder(PrognosticsModel):
    """
    PrognosticsModel for the position coder component.
    Models signal integrity degradation mechanisms affecting position feedback quality,
    synchronous error, and orientation completion reliability.
    """

    inputs = ["orientation_stop_position"]
    states = [
        "signal_degradation",
        "one_rotation_signal_degradation",
        "signal_integrity_degradation",
        "contamination_level",
        "position_error_tracking",
    ]
    outputs = [
        "position_error",
        "synchronous_error",
        "orientation_time",
        "spindle_sequence_state",
        "orientation_completed",
    ]
    events = [
        "position_coder_disconnected",
        "one_rotation_signal_detection_error",
        "one_rotation_signal_not_detected",
        "position_coder_signal_error",
    ]

    units = {
        "orientation_stop_position": "pulses",
        "signal_degradation": "dimensionless",
        "one_rotation_signal_degradation": "dimensionless",
        "signal_integrity_degradation": "dimensionless",
        "contamination_level": "dimensionless",
        "position_error_tracking": "pulses",
        "position_error": "pulses",
        "synchronous_error": "pulses",
        "orientation_time": "ms",
        "spindle_sequence_state": "state",
        "orientation_completed": None,
    }

    default_parameters = {
        "signal_degradation_rate": 0.02,
        "one_rotation_degradation_rate": 0.015,
        "signal_integrity_degradation_rate": 0.03,
        "contamination_buildup_rate": 0.01,
        "critical_signal_degradation_threshold": 0.2,
        "one_rotation_detection_error_threshold": 0.4,
        "one_rotation_not_detected_threshold": 0.2,
        "signal_error_integrity_threshold": 0.3,
        "contamination_warning_threshold": 0.5,
        "critical_position_error": 2000.0,
        "position_error_lag_coefficient": 0.3,
        "base_position_error_scale": 4096.0,
        "base_synchronous_error_scale": 512.0,
        "base_orientation_time": 5000.0,
        "orientation_time_degradation_scale": 55000.0,
        "orientation_completion_threshold": 0.25,
        "x0": {
            "signal_degradation": 1.0,
            "one_rotation_signal_degradation": 1.0,
            "signal_integrity_degradation": 1.0,
            "contamination_level": 0.0,
            "position_error_tracking": 0.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "signal_degradation": self.parameters["x0"]["signal_degradation"],
            "one_rotation_signal_degradation": self.parameters["x0"]["one_rotation_signal_degradation"],
            "signal_integrity_degradation": self.parameters["x0"]["signal_integrity_degradation"],
            "contamination_level": self.parameters["x0"]["contamination_level"],
            "position_error_tracking": self.parameters["x0"]["position_error_tracking"],
        })

    def dx(self, x, u):
        p = self.parameters

        # Retrieve driver inputs safely
        if u is not None and u["orientation_stop_position"] is not None:
            orientation_stop_position = float(u["orientation_stop_position"])
        else:
            orientation_stop_position = 0.0

        # Normalize driver inputs from orientation_stop_position
        # load_cycles_and_speed: normalized [0,1] from orientation_stop_position range [0,4095]
        load_cycles_and_speed = orientation_stop_position / 4095.0
        thermal_cycling_and_wear = load_cycles_and_speed
        thermal_stress_and_oxidation = load_cycles_and_speed
        metallic_wear_particles = load_cycles_and_speed

        # Convert rates: rates are per 1000 operating hours convention
        signal_deg_rate = p["signal_degradation_rate"] / 1000.0
        one_rot_deg_rate = p["one_rotation_degradation_rate"] / 1000.0
        sig_int_deg_rate = p["signal_integrity_degradation_rate"] / 1000.0
        contam_rate = p["contamination_buildup_rate"] / 1000.0

        # signal_degradation: degradation state
        d_signal_degradation = -signal_deg_rate * load_cycles_and_speed

        # one_rotation_signal_degradation: degradation state
        d_one_rotation_signal_degradation = -one_rot_deg_rate * thermal_cycling_and_wear

        # signal_integrity_degradation: degradation state
        d_signal_integrity_degradation = -sig_int_deg_rate * thermal_stress_and_oxidation

        # contamination_level: accumulation state
        d_contamination_level = contam_rate * metallic_wear_particles

        # position_error_tracking: tracking state
        instantaneous_position_error = p["base_position_error_scale"] * (1.0 - x["signal_degradation"])
        lag_coeff = min(p["position_error_lag_coefficient"], 0.5)
        d_position_error_tracking = lag_coeff * (instantaneous_position_error - x["position_error_tracking"])

        return self.StateContainer({
            "signal_degradation": d_signal_degradation,
            "one_rotation_signal_degradation": d_one_rotation_signal_degradation,
            "signal_integrity_degradation": d_signal_integrity_degradation,
            "contamination_level": d_contamination_level,
            "position_error_tracking": d_position_error_tracking,
        })

    def next_state(self, x, u, dt):
        # Get derivatives
        dxdt = self.dx(x, u)

        # signal_degradation
        new_signal_degradation = x["signal_degradation"] + dxdt["signal_degradation"] * dt
        new_signal_degradation = max(0.0, min(1.0, new_signal_degradation))

        # one_rotation_signal_degradation
        new_one_rotation_signal_degradation = x["one_rotation_signal_degradation"] + dxdt["one_rotation_signal_degradation"] * dt
        new_one_rotation_signal_degradation = max(0.0, min(1.0, new_one_rotation_signal_degradation))

        # signal_integrity_degradation
        new_signal_integrity_degradation = x["signal_integrity_degradation"] + dxdt["signal_integrity_degradation"] * dt
        new_signal_integrity_degradation = max(0.0, min(1.0, new_signal_integrity_degradation))

        # contamination_level
        new_contamination_level = x["contamination_level"] + dxdt["contamination_level"] * dt
        new_contamination_level = max(0.0, min(1.0, new_contamination_level))

        # position_error_tracking
        new_position_error_tracking = x["position_error_tracking"] + dxdt["position_error_tracking"] * dt
        new_position_error_tracking = max(-4096.0, min(4096.0, new_position_error_tracking))

        return self.StateContainer({
            "signal_degradation": new_signal_degradation,
            "one_rotation_signal_degradation": new_one_rotation_signal_degradation,
            "signal_integrity_degradation": new_signal_integrity_degradation,
            "contamination_level": new_contamination_level,
            "position_error_tracking": new_position_error_tracking,
        })

    def output(self, x):
        p = self.parameters

        signal_degradation = x["signal_degradation"]
        one_rotation_signal_degradation = x["one_rotation_signal_degradation"]
        signal_integrity_degradation = x["signal_integrity_degradation"]
        contamination_level = x["contamination_level"]
        position_error_tracking = x["position_error_tracking"]

        # position_error: sign_factor based on position_error_tracking parity (use abs value)
        # Since we don't have orientation_stop_position in output(), use position_error_tracking sign
        # Use a fixed sign_factor of +1 as default (no input available in output())
        sign_factor = 1.0
        position_error = p["base_position_error_scale"] * (1.0 - signal_degradation) * sign_factor

        # synchronous_error
        synchronous_error = (
            p["base_synchronous_error_scale"]
            * (1.0 - signal_integrity_degradation)
            * (1.0 + contamination_level)
        )

        # orientation_time
        min_health = min(float(signal_degradation), float(one_rotation_signal_degradation))
        orientation_time = (
            p["base_orientation_time"]
            + p["orientation_time_degradation_scale"] * (1.0 - min_health)
        )

        # spindle_sequence_state
        one_rot_not_detected_thresh = p["one_rotation_not_detected_threshold"]
        sig_error_thresh = p["signal_error_integrity_threshold"]

        if float(one_rotation_signal_degradation) < one_rot_not_detected_thresh:
            spindle_sequence_state = "b"
        elif float(signal_integrity_degradation) < sig_error_thresh:
            spindle_sequence_state = "c"
        elif float(one_rotation_signal_degradation) > 0.5 and float(signal_integrity_degradation) > 0.5:
            spindle_sequence_state = "e"
        else:
            spindle_sequence_state = "d"

        # orientation_completed
        min_combined = min(
            float(signal_degradation),
            float(one_rotation_signal_degradation),
            float(signal_integrity_degradation),
        )
        orientation_completed = bool(
            min_combined >= p["orientation_completion_threshold"]
            and abs(float(position_error_tracking)) < p["critical_position_error"]
        )

        return self.OutputContainer({
            "position_error": position_error,
            "synchronous_error": synchronous_error,
            "orientation_time": orientation_time,
            "spindle_sequence_state": spindle_sequence_state,
            "orientation_completed": orientation_completed,
        })

    def event_state(self, x) -> dict:
        p = self.parameters

        signal_degradation = float(x["signal_degradation"])
        one_rotation_signal_degradation = float(x["one_rotation_signal_degradation"])
        signal_integrity_degradation = float(x["signal_integrity_degradation"])
        contamination_level = float(x["contamination_level"])
        position_error_tracking = float(x["position_error_tracking"])

        # position_coder_disconnected (SP0027)
        csd_thresh = p["critical_signal_degradation_threshold"]
        es_disconnected = (signal_degradation - csd_thresh) / (1.0 - csd_thresh)
        es_disconnected = max(0.0, min(1.0, es_disconnected))

        # one_rotation_signal_detection_error (SP0041)
        det_err_thresh = p["one_rotation_detection_error_threshold"]
        contam_warn_thresh = p["contamination_warning_threshold"]
        es_det_err_health = (one_rotation_signal_degradation - det_err_thresh) / (1.0 - det_err_thresh)
        es_det_err_contam = (contam_warn_thresh - contamination_level) / contam_warn_thresh
        es_detection_error = min(es_det_err_health, es_det_err_contam)
        es_detection_error = max(0.0, min(1.0, es_detection_error))

        # one_rotation_signal_not_detected (SP0042)
        not_det_thresh = p["one_rotation_not_detected_threshold"]
        es_not_detected = (one_rotation_signal_degradation - not_det_thresh) / (1.0 - not_det_thresh)
        es_not_detected = max(0.0, min(1.0, es_not_detected))

        # position_coder_signal_error (SP0047)
        sig_err_thresh = p["signal_error_integrity_threshold"]
        crit_pos_err = p["critical_position_error"]
        es_sig_err_integrity = (signal_integrity_degradation - sig_err_thresh) / (1.0 - sig_err_thresh)
        es_sig_err_position = (crit_pos_err - abs(position_error_tracking)) / crit_pos_err
        es_signal_error = min(es_sig_err_integrity, es_sig_err_position)
        es_signal_error = max(0.0, min(1.0, es_signal_error))

        return {
            "position_coder_disconnected": es_disconnected,
            "one_rotation_signal_detection_error": es_detection_error,
            "one_rotation_signal_not_detected": es_not_detected,
            "position_coder_signal_error": es_signal_error,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        signal_degradation = float(x["signal_degradation"])
        one_rotation_signal_degradation = float(x["one_rotation_signal_degradation"])
        signal_integrity_degradation = float(x["signal_integrity_degradation"])
        contamination_level = float(x["contamination_level"])
        position_error_tracking = float(x["position_error_tracking"])

        return {
            "position_coder_disconnected": bool(
                signal_degradation <= p["critical_signal_degradation_threshold"]
            ),
            "one_rotation_signal_detection_error": bool(
                one_rotation_signal_degradation <= p["one_rotation_detection_error_threshold"]
                or contamination_level >= p["contamination_warning_threshold"]
            ),
            "one_rotation_signal_not_detected": bool(
                one_rotation_signal_degradation <= p["one_rotation_not_detected_threshold"]
            ),
            "position_coder_signal_error": bool(
                signal_integrity_degradation <= p["signal_error_integrity_threshold"]
                or abs(position_error_tracking) >= p["critical_position_error"]
            ),
        }