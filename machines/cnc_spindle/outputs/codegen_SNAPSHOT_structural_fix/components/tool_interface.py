from progpy import PrognosticsModel


class ToolInterface(PrognosticsModel):
    """
    PrognosticsModel for the cutting tool interface subsystem.

    Tracks degradation of positioning accuracy, interface rigidity, and
    orientation accuracy. Raises faults for overheat, velocity error,
    sensor polarity, coder disconnection, and gear ratio parameter anomalies.
    """

    inputs = []
    states = [
        "positioning_accuracy_degradation",
        "interface_rigidity_degradation",
        "orientation_accuracy_degradation",
        "thermal_stress_accumulation",
        "velocity_error_accumulation",
        "gear_ratio_error_accumulation",
        "sensor_health",
    ]
    outputs = []
    events = [
        "motor_overheat",
        "excessive_velocity_error",
        "position_sensor_polarity_incorrect",
        "position_coder_disconnected",
        "gear_ratio_parameter_error",
    ]

    default_parameters = {
        # Degradation rates
        "positioning_accuracy_degradation_rate": 0.02,
        "interface_rigidity_degradation_rate": 0.015,
        "orientation_accuracy_degradation_rate": 0.03,
        "sensor_health_degradation_rate": 0.01,
        # Thermal stress parameters
        "thermal_stress_buildup_rate": 0.5,
        "thermal_stress_dissipation_rate": 0.2,
        "thermal_stress_capacity": 200.0,
        # Event thresholds
        "overheat_threshold": 140.0,
        "velocity_error_buildup_rate": 10.0,
        "velocity_error_threshold": 500.0,
        "gear_ratio_error_buildup_rate": 0.01,
        "gear_ratio_error_threshold": 0.5,
        "sensor_polarity_threshold": 0.4,
        "coder_disconnect_threshold": 0.2,
        # Ground truth reference parameters
        "critical_bearing_wear": 0.8,
        "critical_position_error": 2000.0,
        "critical_vibration": 10.7,
        "critical_temperature": 150.0,
        # Initial state values
        "x0": {
            "positioning_accuracy_degradation": 1.0,
            "interface_rigidity_degradation": 1.0,
            "orientation_accuracy_degradation": 1.0,
            "thermal_stress_accumulation": 25.0,
            "velocity_error_accumulation": 0.0,
            "gear_ratio_error_accumulation": 0.0,
            "sensor_health": 1.0,
        },
    }

    # Units for all states, inputs, and outputs
    units = {
        "positioning_accuracy_degradation": "dimensionless",
        "interface_rigidity_degradation": "dimensionless",
        "orientation_accuracy_degradation": "dimensionless",
        "thermal_stress_accumulation": "degC",
        "velocity_error_accumulation": "rpm",
        "gear_ratio_error_accumulation": "dimensionless",
        "sensor_health": "dimensionless",
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "positioning_accuracy_degradation": 1.0,
            "interface_rigidity_degradation": 1.0,
            "orientation_accuracy_degradation": 1.0,
            "thermal_stress_accumulation": 25.0,
            "velocity_error_accumulation": 0.0,
            "gear_ratio_error_accumulation": 0.0,
            "sensor_health": 1.0,
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # Internal normalized drivers (no input ports; treated as constant 1.0
        # representing full operational stress in a standalone simulation).
        # In a composite integration these would be driven by upstream outputs.
        load_cycles_and_speed_driver = 1.0
        thermal_cycling_and_wear_driver = 1.0
        thermal_stress_and_oxidation_driver = 1.0
        metallic_wear_particles_driver = 1.0

        # --- DEGRADATION states ---

        # positioning_accuracy_degradation
        new_pad = (
            x["positioning_accuracy_degradation"]
            - p["positioning_accuracy_degradation_rate"]
            * load_cycles_and_speed_driver
            * dt
        )
        new_pad = max(0.0, min(1.0, new_pad))

        # interface_rigidity_degradation
        new_ird = (
            x["interface_rigidity_degradation"]
            - p["interface_rigidity_degradation_rate"]
            * thermal_cycling_and_wear_driver
            * dt
        )
        new_ird = max(0.0, min(1.0, new_ird))

        # orientation_accuracy_degradation
        new_oad = (
            x["orientation_accuracy_degradation"]
            - p["orientation_accuracy_degradation_rate"]
            * thermal_stress_and_oxidation_driver
            * dt
        )
        new_oad = max(0.0, min(1.0, new_oad))

        # sensor_health
        new_sh = (
            x["sensor_health"]
            - p["sensor_health_degradation_rate"]
            * metallic_wear_particles_driver
            * dt
        )
        new_sh = max(0.0, min(1.0, new_sh))

        # --- ACCUMULATION states ---

        # thermal_stress_accumulation:
        # Builds up due to degraded motor power control (interface_rigidity_degradation),
        # dissipates at base rate.
        thermal_buildup = (
            p["thermal_stress_buildup_rate"]
            * (1.0 - x["interface_rigidity_degradation"])
            * dt
        )
        thermal_dissipation = p["thermal_stress_dissipation_rate"] * dt
        new_tsa = x["thermal_stress_accumulation"] + thermal_buildup - thermal_dissipation
        new_tsa = max(0.0, min(p["thermal_stress_capacity"], new_tsa))

        # velocity_error_accumulation:
        # Grows as positioning_accuracy_degradation worsens.
        new_vea = (
            x["velocity_error_accumulation"]
            + p["velocity_error_buildup_rate"]
            * (1.0 - x["positioning_accuracy_degradation"])
            * dt
        )
        new_vea = max(0.0, min(1000.0, new_vea))

        # gear_ratio_error_accumulation:
        # Grows as orientation_accuracy_degradation worsens.
        new_grea = (
            x["gear_ratio_error_accumulation"]
            + p["gear_ratio_error_buildup_rate"]
            * (1.0 - x["orientation_accuracy_degradation"])
            * dt
        )
        new_grea = max(0.0, min(1.0, new_grea))

        return self.StateContainer({
            "positioning_accuracy_degradation": new_pad,
            "interface_rigidity_degradation": new_ird,
            "orientation_accuracy_degradation": new_oad,
            "thermal_stress_accumulation": new_tsa,
            "velocity_error_accumulation": new_vea,
            "gear_ratio_error_accumulation": new_grea,
            "sensor_health": new_sh,
        })

    def output(self, x):
        # No output ports defined in the spec.
        return self.OutputContainer({})

    def event_state(self, x) -> dict:
        p = self.parameters

        # motor_overheat (SP0001):
        # Normalized so that event_state = 1.0 at initial thermal_stress = 25.0 degC.
        initial_thermal_stress = 25.0
        overheat_range = p["overheat_threshold"] - initial_thermal_stress
        if overheat_range <= 0.0:
            motor_overheat_es = 0.0
        else:
            motor_overheat_es = (
                p["overheat_threshold"] - x["thermal_stress_accumulation"]
            ) / overheat_range
        motor_overheat_es = max(0.0, min(1.0, motor_overheat_es))

        # excessive_velocity_error (SP0002):
        vel_thresh = p["velocity_error_threshold"]
        if vel_thresh <= 0.0:
            excessive_velocity_error_es = 0.0
        else:
            excessive_velocity_error_es = (
                vel_thresh - x["velocity_error_accumulation"]
            ) / vel_thresh
        excessive_velocity_error_es = max(0.0, min(1.0, excessive_velocity_error_es))

        # position_sensor_polarity_incorrect (SP0021):
        polarity_thresh = p["sensor_polarity_threshold"]
        polarity_range = 1.0 - polarity_thresh
        if polarity_range <= 0.0:
            polarity_es = 0.0
        else:
            polarity_es = (x["sensor_health"] - polarity_thresh) / polarity_range
        polarity_es = max(0.0, min(1.0, polarity_es))

        # position_coder_disconnected (SP0027):
        coder_thresh = p["coder_disconnect_threshold"]
        coder_range = 1.0 - coder_thresh
        if coder_range <= 0.0:
            coder_es = 0.0
        else:
            coder_es = (x["sensor_health"] - coder_thresh) / coder_range
        coder_es = max(0.0, min(1.0, coder_es))

        # gear_ratio_parameter_error (SP0035):
        gr_thresh = p["gear_ratio_error_threshold"]
        if gr_thresh <= 0.0:
            gear_ratio_es = 0.0
        else:
            gear_ratio_es = (
                gr_thresh - x["gear_ratio_error_accumulation"]
            ) / gr_thresh
        gear_ratio_es = max(0.0, min(1.0, gear_ratio_es))

        return {
            "motor_overheat": motor_overheat_es,
            "excessive_velocity_error": excessive_velocity_error_es,
            "position_sensor_polarity_incorrect": polarity_es,
            "position_coder_disconnected": coder_es,
            "gear_ratio_parameter_error": gear_ratio_es,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        return {
            "motor_overheat": bool(
                x["thermal_stress_accumulation"] >= p["overheat_threshold"]
            ),
            "excessive_velocity_error": bool(
                x["velocity_error_accumulation"] >= p["velocity_error_threshold"]
            ),
            "position_sensor_polarity_incorrect": bool(
                x["sensor_health"] <= p["sensor_polarity_threshold"]
            ),
            "position_coder_disconnected": bool(
                x["sensor_health"] <= p["coder_disconnect_threshold"]
            ),
            "gear_ratio_parameter_error": bool(
                x["gear_ratio_error_accumulation"] >= p["gear_ratio_error_threshold"]
            ),
        }