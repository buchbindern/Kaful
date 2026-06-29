from progpy import PrognosticsModel
import numpy as np


class PositionCoder(PrognosticsModel):
    """
    ProgPy model for the position coder component responsible for tool
    positioning accuracy and spindle orientation control.
    """

    inputs = [
        "orientation_stop_position",
        "bearing_wear",
        "grease_degradation",
        "metallic_wear_particles",
    ]

    states = [
        "signal_degradation",
        "one_rotation_signal_degradation",
        "signal_integrity_degradation",
        "contamination_level",
        "position_error_tracked",
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
        # inputs
        "orientation_stop_position": "pulses",
        "bearing_wear": "dimensionless",
        "grease_degradation": "dimensionless",
        "metallic_wear_particles": "dimensionless",
        # states
        "signal_degradation": "dimensionless",
        "one_rotation_signal_degradation": "dimensionless",
        "signal_integrity_degradation": "dimensionless",
        "contamination_level": "dimensionless",
        "position_error_tracked": "pulses",
        # outputs
        "position_error": "pulses",
        "synchronous_error": "pulses",
        "orientation_time": "ms",
        "spindle_sequence_state": "state",
        "orientation_completed": None,
    }

    default_parameters = {
        "signal_degradation_rate": 0.02,
        "one_rotation_signal_degradation_rate": 0.015,
        "signal_integrity_degradation_rate": 0.03,
        "contamination_buildup_rate": 0.01,
        "critical_signal_degradation_threshold": 0.2,
        "one_rotation_warning_threshold": 0.3,
        "one_rotation_critical_threshold": 0.1,
        "signal_integrity_critical_threshold": 0.15,
        "contamination_critical_threshold": 0.85,
        "position_error_base_amplitude": 10.0,
        "critical_position_error": 2000.0,
        "synchronous_error_base_amplitude": 5.0,
        "orientation_time_nominal": 500.0,
        "orientation_time_max": 60000.0,
        "position_error_lag_coefficient": 0.3,
        "contamination_signal_coupling": 0.5,
        "orientation_completion_threshold": 0.2,
        "x0": {
            "signal_degradation": 1.0,
            "one_rotation_signal_degradation": 1.0,
            "signal_integrity_degradation": 1.0,
            "contamination_level": 0.0,
            "position_error_tracked": 0.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "signal_degradation": self.parameters["x0"]["signal_degradation"],
            "one_rotation_signal_degradation": self.parameters["x0"]["one_rotation_signal_degradation"],
            "signal_integrity_degradation": self.parameters["x0"]["signal_integrity_degradation"],
            "contamination_level": self.parameters["x0"]["contamination_level"],
            "position_error_tracked": self.parameters["x0"]["position_error_tracked"],
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # Safely extract driver inputs, defaulting to 1.0 if None
        bearing_wear = 1.0
        grease_degradation = 1.0
        metallic_wear = 1.0

        if u is not None:
            bw = u["bearing_wear"]
            if bw is not None:
                bearing_wear = float(bw)
            gd = u["grease_degradation"]
            if gd is not None:
                grease_degradation = float(gd)
            mw = u["metallic_wear_particles"]
            if mw is not None:
                metallic_wear = float(mw)

        # --- DEGRADATION: signal_degradation ---
        new_signal_degradation = x["signal_degradation"] - p["signal_degradation_rate"] * bearing_wear * dt
        new_signal_degradation = max(0.0, min(1.0, new_signal_degradation))

        # --- DEGRADATION: one_rotation_signal_degradation ---
        new_one_rotation = x["one_rotation_signal_degradation"] - p["one_rotation_signal_degradation_rate"] * bearing_wear * dt
        new_one_rotation = max(0.0, min(1.0, new_one_rotation))

        # --- DEGRADATION: signal_integrity_degradation ---
        contamination_level_current = x["contamination_level"]
        integrity_driver = grease_degradation + p["contamination_signal_coupling"] * contamination_level_current
        new_signal_integrity = x["signal_integrity_degradation"] - p["signal_integrity_degradation_rate"] * integrity_driver * dt
        new_signal_integrity = max(0.0, min(1.0, new_signal_integrity))

        # --- ACCUMULATION: contamination_level ---
        new_contamination = x["contamination_level"] + p["contamination_buildup_rate"] * metallic_wear * dt
        new_contamination = max(0.0, min(1.0, new_contamination))

        # --- TRACKING: position_error_tracked ---
        orientation_stop_position = 0.0
        if u is not None:
            osp = u["orientation_stop_position"]
            if osp is not None:
                orientation_stop_position = float(osp)

        position_error_target = (
            p["position_error_base_amplitude"]
            * (1.0 - x["signal_degradation"])
            * orientation_stop_position
            / 4095.0
        )

        # Stability cap: coefficient = min(raw_coefficient, 0.5)
        lag_coeff = min(p["position_error_lag_coefficient"], 0.5)
        new_position_error_tracked = (
            x["position_error_tracked"]
            + lag_coeff * (position_error_target - x["position_error_tracked"]) * dt
        )
        new_position_error_tracked = max(-4096.0, min(4096.0, new_position_error_tracked))

        return self.StateContainer({
            "signal_degradation": new_signal_degradation,
            "one_rotation_signal_degradation": new_one_rotation,
            "signal_integrity_degradation": new_signal_integrity,
            "contamination_level": new_contamination,
            "position_error_tracked": new_position_error_tracked,
        })

    def output(self, x):
        p = self.parameters

        # position_error: directly from tracking state
        position_error = x["position_error_tracked"]

        # synchronous_error: proportional to loss of one-rotation signal health
        synchronous_error = p["synchronous_error_base_amplitude"] * (1.0 - x["one_rotation_signal_degradation"])
        synchronous_error = max(-512.0, min(512.0, synchronous_error))

        # orientation_time: hyperbolic increase as signal integrity degrades
        sig_int = max(x["signal_integrity_degradation"], 0.01)
        orientation_time = p["orientation_time_nominal"] / sig_int
        orientation_time = max(0.0, min(p["orientation_time_max"], orientation_time))

        # spindle_sequence_state: discrete state from signal_integrity_degradation
        sid = x["signal_integrity_degradation"]
        if sid >= 0.8:
            spindle_sequence_state = "a"
        elif sid >= 0.6:
            spindle_sequence_state = "b"
        elif sid >= 0.4:
            spindle_sequence_state = "c"
        elif sid >= 0.2:
            spindle_sequence_state = "d"
        else:
            spindle_sequence_state = "e"

        # orientation_completed: both signal integrity and index pulse above critical thresholds
        orientation_completed = bool(
            x["signal_integrity_degradation"] >= p["orientation_completion_threshold"]
            and x["one_rotation_signal_degradation"] >= p["one_rotation_critical_threshold"]
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

        # position_coder_disconnected (SP0027)
        crit_thresh = p["critical_signal_degradation_threshold"]
        es_disconnected = (x["signal_degradation"] - crit_thresh) / (1.0 - crit_thresh)
        es_disconnected = max(0.0, min(1.0, es_disconnected))

        # one_rotation_signal_detection_error (SP0041)
        warn_thresh = p["one_rotation_warning_threshold"]
        es_detection_error = (x["one_rotation_signal_degradation"] - warn_thresh) / (1.0 - warn_thresh)
        es_detection_error = max(0.0, min(1.0, es_detection_error))

        # one_rotation_signal_not_detected (SP0042)
        crit_one_rot = p["one_rotation_critical_threshold"]
        es_not_detected = (x["one_rotation_signal_degradation"] - crit_one_rot) / (1.0 - crit_one_rot)
        es_not_detected = max(0.0, min(1.0, es_not_detected))

        # position_coder_signal_error (SP0047)
        integrity_crit = p["signal_integrity_critical_threshold"]
        contamination_crit = p["contamination_critical_threshold"]

        integrity_event_state = (x["signal_integrity_degradation"] - integrity_crit) / (1.0 - integrity_crit)
        integrity_event_state = max(0.0, min(1.0, integrity_event_state))

        contamination_event_state = (contamination_crit - x["contamination_level"]) / contamination_crit
        contamination_event_state = max(0.0, min(1.0, contamination_event_state))

        es_signal_error = min(integrity_event_state, contamination_event_state)
        es_signal_error = max(0.0, min(1.0, es_signal_error))

        return {
            "position_coder_disconnected": es_disconnected,
            "one_rotation_signal_detection_error": es_detection_error,
            "one_rotation_signal_not_detected": es_not_detected,
            "position_coder_signal_error": es_signal_error,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        disconnected = bool(x["signal_degradation"] <= p["critical_signal_degradation_threshold"])
        detection_error = bool(x["one_rotation_signal_degradation"] <= p["one_rotation_warning_threshold"])
        not_detected = bool(x["one_rotation_signal_degradation"] <= p["one_rotation_critical_threshold"])
        signal_error = bool(
            x["signal_integrity_degradation"] <= p["signal_integrity_critical_threshold"]
            or x["contamination_level"] >= p["contamination_critical_threshold"]
        )

        return {
            "position_coder_disconnected": disconnected,
            "one_rotation_signal_detection_error": detection_error,
            "one_rotation_signal_not_detected": not_detected,
            "position_coder_signal_error": signal_error,
        }