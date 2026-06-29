from progpy import PrognosticsModel


class SpindleAmplifier(PrognosticsModel):
    """
    Prognostics model for the spindle amplifier control system.

    Models velocity loop regulation, torque delivery, and cutting capability
    management. Tracks degradation of velocity loop gain stability, torque
    response, and cutting capability, and raises alarms when control
    deviations exceed safe thresholds.

    Events:
        velocity_error_excess_alarm: SPM alarm 02
        excessive_speed_deviation_alarm: SPM alarm C8
        excessive_positional_deviation_alarm: SPM alarm C9
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

    default_parameters = {
        "bearing_wear_rate": 0.02,
        "preload_degradation_rate": 0.015,
        "grease_degradation_rate": 0.03,
        "critical_bearing_wear": 0.8,
        "velocity_error_alarm_threshold": 256.0,
        "speed_deviation_alarm_threshold": 15.0,
        "critical_position_error": 2000.0,
        "velocity_error_lag_coefficient": 0.3,
        "positional_deviation_lag_coefficient": 0.2,
        "speed_arrival_threshold": 15.0,
        "load_scale_factor": 1.0,
        "torque_degradation_sensitivity": 0.5,
        "velocity_error_gain_sensitivity": 0.5,
        "positional_integration_gain": 0.1,
        "x0": {
            "velocity_loop_gain_stability": 1.0,
            "torque_response_degradation": 1.0,
            "cutting_capability_degradation": 1.0,
            "tracked_velocity_error": 0.0,
            "tracked_positional_deviation": 0.0,
        },
    }

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

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "velocity_loop_gain_stability": self.parameters["x0"]["velocity_loop_gain_stability"],
            "torque_response_degradation": self.parameters["x0"]["torque_response_degradation"],
            "cutting_capability_degradation": self.parameters["x0"]["cutting_capability_degradation"],
            "tracked_velocity_error": self.parameters["x0"]["tracked_velocity_error"],
            "tracked_positional_deviation": self.parameters["x0"]["tracked_positional_deviation"],
        })

    def next_state(self, x, u, dt):
        params = self.parameters

        # --- Safe input extraction ---
        spindle_speed_command = u["spindle_speed_command"] if (u is not None and u["spindle_speed_command"] is not None) else 0.0
        torque_command = u["torque_command"] if (u is not None and u["torque_command"] is not None) else 0.0
        motor_voltage = u["motor_voltage"] if (u is not None and u["motor_voltage"] is not None) else 0.0
        maximum_motor_speed = u["maximum_motor_speed"] if (u is not None and u["maximum_motor_speed"] is not None) else 32767.0

        # --- Degradation drivers ---
        # load_cycles_and_speed_driver: normalized spindle speed
        if maximum_motor_speed is not None and float(maximum_motor_speed) > 0.0:
            load_cycles_and_speed_driver = float(spindle_speed_command) / float(maximum_motor_speed)
        else:
            load_cycles_and_speed_driver = 0.0
        load_cycles_and_speed_driver = max(0.0, min(1.0, load_cycles_and_speed_driver))

        # thermal_cycling_and_wear_driver: normalized torque command
        thermal_cycling_and_wear_driver = float(torque_command) / 100.0
        thermal_cycling_and_wear_driver = max(0.0, min(1.0, thermal_cycling_and_wear_driver))

        # thermal_stress_and_oxidation_driver: normalized motor voltage
        thermal_stress_and_oxidation_driver = float(motor_voltage) / 100.0
        thermal_stress_and_oxidation_driver = max(0.0, min(1.0, thermal_stress_and_oxidation_driver))

        # --- Degradation state updates ---
        # velocity_loop_gain_stability
        new_vlgs = x["velocity_loop_gain_stability"] - params["bearing_wear_rate"] * load_cycles_and_speed_driver * dt
        new_vlgs = max(0.0, min(1.0, new_vlgs))

        # torque_response_degradation
        new_trd = x["torque_response_degradation"] - params["preload_degradation_rate"] * thermal_cycling_and_wear_driver * dt
        new_trd = max(0.0, min(1.0, new_trd))

        # cutting_capability_degradation
        new_ccd = x["cutting_capability_degradation"] - params["grease_degradation_rate"] * thermal_stress_and_oxidation_driver * dt
        new_ccd = max(0.0, min(1.0, new_ccd))

        # --- Tracking state: tracked_velocity_error ---
        # target_velocity_error: spindle_speed_command - actual_speed_estimate
        # actual_speed_estimate approximated as spindle_speed_command minus inherent error contribution
        # (in a full implementation, actual feedback would be used)
        # Here we model the error as the amplifier's deviation from commanded speed
        # due to degradation effects
        gain_loss_factor = (
            1.0
            + params["velocity_error_gain_sensitivity"] * (1.0 - x["velocity_loop_gain_stability"])
            + params["torque_degradation_sensitivity"] * (1.0 - x["torque_response_degradation"])
        )
        # actual_speed_estimate = spindle_speed_command / gain_loss_factor (degradation causes speed error)
        actual_speed_estimate = float(spindle_speed_command) / gain_loss_factor if gain_loss_factor > 0.0 else float(spindle_speed_command)
        target_velocity_error = (float(spindle_speed_command) - actual_speed_estimate) * gain_loss_factor

        vel_lag = min(params["velocity_error_lag_coefficient"], 0.5)
        new_tve = x["tracked_velocity_error"] + vel_lag * (target_velocity_error - x["tracked_velocity_error"]) * dt
        new_tve = max(-256.0, min(256.0, new_tve))

        # --- Tracking state: tracked_positional_deviation ---
        target_positional_deviation = x["tracked_positional_deviation"] + params["positional_integration_gain"] * x["tracked_velocity_error"] * dt
        pos_lag = min(params["positional_deviation_lag_coefficient"], 0.5)
        new_tpd = x["tracked_positional_deviation"] + pos_lag * (target_positional_deviation - x["tracked_positional_deviation"]) * dt
        new_tpd = max(-4000.0, min(4000.0, new_tpd))

        return self.StateContainer({
            "velocity_loop_gain_stability": new_vlgs,
            "torque_response_degradation": new_trd,
            "cutting_capability_degradation": new_ccd,
            "tracked_velocity_error": new_tve,
            "tracked_positional_deviation": new_tpd,
        })

    def output(self, x):
        params = self.parameters

        tracked_ve = x["tracked_velocity_error"]
        tracked_pd = x["tracked_positional_deviation"]
        cutting_cap = x["cutting_capability_degradation"]

        # velocity_error: tracked velocity error directly
        velocity_error = tracked_ve

        # speed_arrival_signal and speed_arrival_detection
        speed_arrived = 1 if abs(tracked_ve) <= params["speed_arrival_threshold"] else 0

        # load_detection_level_1: torque_command * cutting_capability_degradation * load_scale_factor
        # We need torque_command from state — but output only receives x.
        # Per spec: "output torque_command * cutting_capability_degradation * load_scale_factor"
        # Since torque_command is an input and not stored in state, we approximate using
        # the tracked_velocity_error indirectly. However, the spec says to use torque_command.
        # Per ProgPy rules, output() only receives x. We must use state values only.
        # We use cutting_capability_degradation scaled by 100 as a proxy for full load,
        # which reflects the degradation effect on load delivery.
        # NOTE: This is a known limitation — torque_command is not available in output().
        # We output cutting_capability_degradation * 100 * load_scale_factor as the load level.
        load_detection = cutting_cap * 100.0 * params["load_scale_factor"]
        load_detection = max(0.0, min(100.0, load_detection))

        # Determine alarm conditions using event_state logic
        es = self.event_state(x)
        velocity_error_alarm = es["velocity_error_excess_alarm"] <= 0.0
        speed_dev_alarm = es["excessive_speed_deviation_alarm"] <= 0.0
        pos_dev_alarm = es["excessive_positional_deviation_alarm"] <= 0.0

        spindle_alarm = 1 if (velocity_error_alarm or speed_dev_alarm or pos_dev_alarm) else 0
        alarm_detection_status = spindle_alarm

        return self.OutputContainer({
            "velocity_error": velocity_error,
            "speed_arrival_signal": float(speed_arrived),
            "load_detection_level_1": load_detection,
            "speed_arrival_detection": float(speed_arrived),
            "spindle_alarm": float(spindle_alarm),
            "alarm_detection_status": float(alarm_detection_status),
        })

    def event_state(self, x) -> dict:
        params = self.parameters

        vlgs = x["velocity_loop_gain_stability"]
        trd = x["torque_response_degradation"]
        ccd = x["cutting_capability_degradation"]
        tve = x["tracked_velocity_error"]
        tpd = x["tracked_positional_deviation"]

        # velocity_error_excess_alarm
        # event_state = 1.0
        #   - max(0.0, (abs(tve) - velocity_error_alarm_threshold) / velocity_error_alarm_threshold)
        #   - max(0.0, (critical_bearing_wear - vlgs) / critical_bearing_wear)
        vel_err_term = max(0.0, (abs(tve) - params["velocity_error_alarm_threshold"]) / params["velocity_error_alarm_threshold"])
        bearing_deg_term = max(0.0, (params["critical_bearing_wear"] - vlgs) / params["critical_bearing_wear"])
        es_velocity_error = 1.0 - vel_err_term - bearing_deg_term
        es_velocity_error = max(0.0, min(1.0, es_velocity_error))

        # excessive_speed_deviation_alarm
        # event_state = 1.0
        #   - max(0.0, (abs(tve) - speed_deviation_alarm_threshold) / speed_deviation_alarm_threshold)
        #   - max(0.0, (1.0 - trd))
        speed_dev_term = max(0.0, (abs(tve) - params["speed_deviation_alarm_threshold"]) / params["speed_deviation_alarm_threshold"])
        torque_deg_term = max(0.0, 1.0 - trd)
        es_speed_deviation = 1.0 - speed_dev_term - torque_deg_term
        es_speed_deviation = max(0.0, min(1.0, es_speed_deviation))

        # excessive_positional_deviation_alarm
        # event_state = 1.0
        #   - max(0.0, (abs(tpd) - critical_position_error) / critical_position_error)
        #   - max(0.0, (1.0 - ccd))
        pos_dev_term = max(0.0, (abs(tpd) - params["critical_position_error"]) / params["critical_position_error"])
        cutting_deg_term = max(0.0, 1.0 - ccd)
        es_pos_deviation = 1.0 - pos_dev_term - cutting_deg_term
        es_pos_deviation = max(0.0, min(1.0, es_pos_deviation))

        return {
            "velocity_error_excess_alarm": es_velocity_error,
            "excessive_speed_deviation_alarm": es_speed_deviation,
            "excessive_positional_deviation_alarm": es_pos_deviation,
        }

    def threshold_met(self, x) -> dict:
        es = self.event_state(x)
        return {
            "velocity_error_excess_alarm": bool(es["velocity_error_excess_alarm"] <= 0.0),
            "excessive_speed_deviation_alarm": bool(es["excessive_speed_deviation_alarm"] <= 0.0),
            "excessive_positional_deviation_alarm": bool(es["excessive_positional_deviation_alarm"] <= 0.0),
        }