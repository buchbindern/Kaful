from progpy import PrognosticsModel


class ToolInterface(PrognosticsModel):
    """
    Prognostics model for the cutting tool interface responsible for tool
    positioning accuracy, rigidity, and orientation repeatability.
    """

    inputs = []
    outputs = []
    states = [
        "positioning_accuracy_degradation",
        "interface_rigidity_degradation",
        "orientation_accuracy_degradation",
        "thermal_stress_accumulation",
        "velocity_error_accumulation",
        "feedback_signal_integrity",
        "sensor_polarity_state",
        "gear_ratio_parameter_state",
    ]
    events = [
        "motor_overheat",
        "excessive_velocity_error",
        "position_sensor_polarity_incorrect",
        "position_coder_disconnected",
        "gear_ratio_parameter_error",
    ]

    default_parameters = {
        "positioning_accuracy_degradation_rate": 0.02,
        "interface_rigidity_degradation_rate": 0.015,
        "orientation_accuracy_degradation_rate": 0.03,
        "feedback_signal_degradation_rate": 0.01,
        "thermal_stress_rise_rate": 2.0,
        "thermal_stress_decay_rate": 1.0,
        "thermal_stress_ambient": 25.0,
        "thermal_stress_capacity": 200.0,
        "velocity_error_rise_rate": 5.0,
        "velocity_error_decay_rate": 2.0,
        "velocity_error_capacity": 1000.0,
        "overheat_threshold": 140.0,
        "critical_bearing_wear": 0.8,
        "velocity_error_fault_threshold": 500.0,
        "feedback_loss_threshold": 0.2,
        "orientation_critical_threshold": 0.2,
        "load_cycle_driver_scale": 1.0,
        "thermal_cycle_driver_scale": 1.0,
        "thermal_stress_driver_scale": 1.0,
        "contamination_driver_scale": 1.0,
        "x0": {
            "positioning_accuracy_degradation": 1.0,
            "interface_rigidity_degradation": 1.0,
            "orientation_accuracy_degradation": 1.0,
            "thermal_stress_accumulation": 25.0,
            "velocity_error_accumulation": 0.0,
            "feedback_signal_integrity": 1.0,
            "sensor_polarity_state": 1.0,
            "gear_ratio_parameter_state": 1.0,
        },
    }

    units = {
        "positioning_accuracy_degradation": "dimensionless",
        "interface_rigidity_degradation": "dimensionless",
        "orientation_accuracy_degradation": "dimensionless",
        "thermal_stress_accumulation": "degC",
        "velocity_error_accumulation": "rpm",
        "feedback_signal_integrity": "dimensionless",
        "sensor_polarity_state": "dimensionless",
        "gear_ratio_parameter_state": "dimensionless",
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "positioning_accuracy_degradation": 1.0,
            "interface_rigidity_degradation": 1.0,
            "orientation_accuracy_degradation": 1.0,
            "thermal_stress_accumulation": 25.0,
            "velocity_error_accumulation": 0.0,
            "feedback_signal_integrity": 1.0,
            "sensor_polarity_state": 1.0,
            "gear_ratio_parameter_state": 1.0,
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # --- positioning_accuracy_degradation (degradation) ---
        rate_pad = p["positioning_accuracy_degradation_rate"] / 1000.0
        new_pad = x["positioning_accuracy_degradation"] - rate_pad * p["load_cycle_driver_scale"] * dt
        new_pad = max(0.0, min(1.0, new_pad))

        # --- interface_rigidity_degradation (degradation) ---
        rate_ird = p["interface_rigidity_degradation_rate"] / 1000.0
        new_ird = x["interface_rigidity_degradation"] - rate_ird * p["thermal_cycle_driver_scale"] * dt
        new_ird = max(0.0, min(1.0, new_ird))

        # --- orientation_accuracy_degradation (degradation) ---
        rate_oad = p["orientation_accuracy_degradation_rate"] / 1000.0
        new_oad = x["orientation_accuracy_degradation"] - rate_oad * p["thermal_stress_driver_scale"] * dt
        new_oad = max(0.0, min(1.0, new_oad))

        # --- feedback_signal_integrity (degradation) ---
        rate_fsi = p["feedback_signal_degradation_rate"] / 1000.0
        new_fsi = x["feedback_signal_integrity"] - rate_fsi * p["contamination_driver_scale"] * dt
        new_fsi = max(0.0, min(1.0, new_fsi))

        # --- thermal_stress_accumulation (accumulation) ---
        # Component is considered active (no input ports); always rising
        new_tsa = x["thermal_stress_accumulation"] + p["thermal_stress_rise_rate"] * dt
        new_tsa = max(0.0, min(p["thermal_stress_capacity"], new_tsa))

        # --- velocity_error_accumulation (accumulation) ---
        if new_pad < p["critical_bearing_wear"]:
            new_vea = x["velocity_error_accumulation"] + p["velocity_error_rise_rate"] * (1.0 - new_pad) * dt
        else:
            new_vea = x["velocity_error_accumulation"] - p["velocity_error_decay_rate"] * dt
        new_vea = max(0.0, min(p["velocity_error_capacity"], new_vea))

        # --- sensor_polarity_state (static) ---
        new_sps = x["sensor_polarity_state"]

        # --- gear_ratio_parameter_state (static) ---
        new_grps = x["gear_ratio_parameter_state"]

        return self.StateContainer({
            "positioning_accuracy_degradation": new_pad,
            "interface_rigidity_degradation": new_ird,
            "orientation_accuracy_degradation": new_oad,
            "thermal_stress_accumulation": new_tsa,
            "velocity_error_accumulation": new_vea,
            "feedback_signal_integrity": new_fsi,
            "sensor_polarity_state": new_sps,
            "gear_ratio_parameter_state": new_grps,
        })

    def output(self, x):
        return self.OutputContainer({})

    def event_state(self, x) -> dict:
        p = self.parameters

        # motor_overheat (SP0001)
        tsa = x["thermal_stress_accumulation"]
        if tsa < p["overheat_threshold"]:
            motor_overheat_es = 1.0
        else:
            motor_overheat_es = 0.0

        # excessive_velocity_error (SP0002)
        vea = x["velocity_error_accumulation"]
        pad = x["positioning_accuracy_degradation"]
        if vea < p["velocity_error_fault_threshold"] and pad >= p["critical_bearing_wear"]:
            excessive_velocity_error_es = 1.0
        else:
            excessive_velocity_error_es = 0.0

        # position_sensor_polarity_incorrect (SP0021)
        sps = x["sensor_polarity_state"]
        if sps == 1.0:
            position_sensor_polarity_incorrect_es = 1.0
        else:
            position_sensor_polarity_incorrect_es = 0.0

        # position_coder_disconnected (SP0027)
        fsi = x["feedback_signal_integrity"]
        if fsi > p["feedback_loss_threshold"]:
            position_coder_disconnected_es = 1.0
        else:
            position_coder_disconnected_es = 0.0

        # gear_ratio_parameter_error (SP0035)
        grps = x["gear_ratio_parameter_state"]
        oad = x["orientation_accuracy_degradation"]
        if grps == 1.0 and oad > p["orientation_critical_threshold"]:
            gear_ratio_parameter_error_es = 1.0
        else:
            gear_ratio_parameter_error_es = 0.0

        return {
            "motor_overheat": motor_overheat_es,
            "excessive_velocity_error": excessive_velocity_error_es,
            "position_sensor_polarity_incorrect": position_sensor_polarity_incorrect_es,
            "position_coder_disconnected": position_coder_disconnected_es,
            "gear_ratio_parameter_error": gear_ratio_parameter_error_es,
        }

    def threshold_met(self, x) -> dict:
        es = self.event_state(x)
        return {
            "motor_overheat": bool(es["motor_overheat"] <= 0.0),
            "excessive_velocity_error": bool(es["excessive_velocity_error"] <= 0.0),
            "position_sensor_polarity_incorrect": bool(es["position_sensor_polarity_incorrect"] <= 0.0),
            "position_coder_disconnected": bool(es["position_coder_disconnected"] <= 0.0),
            "gear_ratio_parameter_error": bool(es["gear_ratio_parameter_error"] <= 0.0),
        }