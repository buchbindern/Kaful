from progpy import PrognosticsModel
import numpy as np


class SpindleMotor(PrognosticsModel):
    """
    ProgPy PrognosticsModel for the spindle motor component.

    Converts electrical voltage commands into mechanical torque and speed,
    while managing thermal dynamics and velocity control accuracy.
    """

    inputs = ["motor_voltage_command"]
    states = [
        "motor_temperature",
        "velocity_error_accumulation",
        "position_accuracy_degradation",
        "commanded_speed_tracking",
    ]
    outputs = [
        "actual_spindle_speed",
        "zero_speed_detection",
        "motor_current",
        "drive_torque",
    ]
    events = [
        "motor_overheat",
        "excessive_velocity_error",
        "overspeed",
        "short_time_overload",
        "motor_lock",
        "current_overload",
    ]

    units = {
        # inputs
        "motor_voltage_command": "%",
        # states
        "motor_temperature": "celsius",
        "velocity_error_accumulation": "rpm",
        "position_accuracy_degradation": "dimensionless",
        "commanded_speed_tracking": "rpm",
        # outputs
        "actual_spindle_speed": "rpm",
        "zero_speed_detection": "boolean",
        "motor_current": "a",
        "drive_torque": "Nm",
    }

    default_parameters = {
        "motor_heating_rate": 0.5,
        "motor_cooling_rate": 0.05,
        "ambient_temperature": 25.0,
        "overheat_threshold": 140.0,
        "critical_temperature": 150.0,
        "velocity_error_buildup_rate": 0.5,
        "velocity_error_decay_rate": 0.1,
        "velocity_error_threshold": 500.0,
        "position_accuracy_degradation_rate": 0.02,
        "speed_tracking_lag_coefficient": 0.3,
        "max_commanded_speed": 32767.0,
        "overspeed_threshold": 32767.0,
        "zero_speed_threshold": 0.75,
        "motor_torque_constant": 1.5,
        "motor_current_gain": 0.8,
        "current_overload_accuracy_threshold": 0.2,
        "motor_lock_accuracy_threshold": 0.15,
        "motor_lock_error_threshold": 2000.0,
        "short_time_overload_temp_threshold": 120.0,
        "short_time_overload_error_threshold": 300.0,
        "speed_arrival_threshold": 15.0,
        "critical_position_error": 2000.0,
        "x0": {
            "motor_temperature": 25.0,
            "velocity_error_accumulation": 0.0,
            "position_accuracy_degradation": 1.0,
            "commanded_speed_tracking": 0.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "motor_temperature": self.parameters["x0"]["motor_temperature"],
            "velocity_error_accumulation": self.parameters["x0"]["velocity_error_accumulation"],
            "position_accuracy_degradation": self.parameters["x0"]["position_accuracy_degradation"],
            "commanded_speed_tracking": self.parameters["x0"]["commanded_speed_tracking"],
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # Safe input extraction
        if u is not None and u["motor_voltage_command"] is not None:
            voltage_cmd = float(u["motor_voltage_command"])
        else:
            voltage_cmd = 0.0

        voltage_norm = voltage_cmd / 100.0

        # --- motor_temperature (accumulation) ---
        heating_power = p["motor_heating_rate"] * (voltage_norm ** 2)
        cooling_power = p["motor_cooling_rate"] * (x["motor_temperature"] - p["ambient_temperature"])
        motor_temp_new = x["motor_temperature"] + (heating_power - cooling_power) * dt
        motor_temp_new = max(25.0, min(200.0, motor_temp_new))

        # --- commanded_speed_tracking (tracking) ---
        speed_target = voltage_norm * p["max_commanded_speed"]
        lag_coeff = min(p["speed_tracking_lag_coefficient"], 0.5)
        cmd_speed_new = x["commanded_speed_tracking"] + lag_coeff * (speed_target - x["commanded_speed_tracking"]) * dt
        cmd_speed_new = max(0.0, min(32767.0, cmd_speed_new))

        # --- velocity_error_accumulation (accumulation) ---
        # Internal estimate of actual speed
        actual_speed_est = x["commanded_speed_tracking"] * x["position_accuracy_degradation"]
        speed_error_magnitude = abs(x["commanded_speed_tracking"] - actual_speed_est)

        if speed_error_magnitude > p["speed_arrival_threshold"]:
            vel_err_new = x["velocity_error_accumulation"] + p["velocity_error_buildup_rate"] * voltage_norm * dt
        else:
            vel_err_new = x["velocity_error_accumulation"] - p["velocity_error_decay_rate"] * dt
        vel_err_new = max(0.0, min(5000.0, vel_err_new))

        # --- position_accuracy_degradation (degradation) ---
        # rate is per step; driver is voltage_norm (proxy for load_cycles_and_speed)
        rate = p["position_accuracy_degradation_rate"] / 1000.0
        pos_acc_new = x["position_accuracy_degradation"] - rate * voltage_norm * dt
        pos_acc_new = max(0.0, min(1.0, pos_acc_new))

        return self.StateContainer({
            "motor_temperature": motor_temp_new,
            "velocity_error_accumulation": vel_err_new,
            "position_accuracy_degradation": pos_acc_new,
            "commanded_speed_tracking": cmd_speed_new,
        })

    def output(self, x):
        p = self.parameters

        # --- actual_spindle_speed ---
        actual_speed = x["commanded_speed_tracking"] * x["position_accuracy_degradation"]
        actual_speed = max(0.0, min(32767.0, actual_speed))

        # --- zero_speed_detection ---
        zero_speed = float(actual_speed <= p["zero_speed_threshold"])

        # --- motor_current ---
        # Reconstruct voltage_norm from commanded_speed_tracking is not possible here;
        # we use the state-based approach: base current is not directly available,
        # so we approximate via commanded_speed_tracking / max_commanded_speed as voltage proxy
        # However, output() only receives x, not u. We use commanded_speed_tracking as proxy.
        voltage_norm_proxy = x["commanded_speed_tracking"] / p["max_commanded_speed"]
        voltage_cmd_proxy = voltage_norm_proxy * 100.0
        base_current = p["motor_current_gain"] * voltage_cmd_proxy
        thermal_factor = 1.0 + 0.3 * (
            (x["motor_temperature"] - p["ambient_temperature"]) /
            (p["critical_temperature"] - p["ambient_temperature"])
        )
        motor_current = base_current * thermal_factor
        motor_current = max(0.0, min(100.0, motor_current))

        # --- drive_torque ---
        drive_torque = motor_current * p["motor_torque_constant"] * x["position_accuracy_degradation"]
        drive_torque = max(0.0, drive_torque)

        return self.OutputContainer({
            "actual_spindle_speed": actual_speed,
            "zero_speed_detection": zero_speed,
            "motor_current": motor_current,
            "drive_torque": drive_torque,
        })

    def event_state(self, x) -> dict:
        p = self.parameters

        # --- motor_overheat ---
        motor_overheat_es = (p["overheat_threshold"] - x["motor_temperature"]) / (
            p["overheat_threshold"] - p["ambient_temperature"]
        )
        motor_overheat_es = float(max(0.0, min(1.0, motor_overheat_es)))

        # --- excessive_velocity_error ---
        excessive_vel_err_es = 1.0 - (x["velocity_error_accumulation"] / p["velocity_error_threshold"])
        excessive_vel_err_es = float(max(0.0, min(1.0, excessive_vel_err_es)))

        # --- overspeed ---
        overspeed_es = 1.0 - (x["commanded_speed_tracking"] / p["overspeed_threshold"])
        overspeed_es = float(max(0.0, min(1.0, overspeed_es)))

        # --- short_time_overload ---
        temp_margin = (p["short_time_overload_temp_threshold"] - x["motor_temperature"]) / (
            p["short_time_overload_temp_threshold"] - p["ambient_temperature"]
        )
        error_margin = 1.0 - (x["velocity_error_accumulation"] / p["short_time_overload_error_threshold"])
        short_time_overload_es = float(min(
            max(0.0, min(1.0, temp_margin)),
            max(0.0, min(1.0, error_margin))
        ))

        # --- motor_lock ---
        lock_error_margin = 1.0 - (x["velocity_error_accumulation"] / p["motor_lock_error_threshold"])
        lock_accuracy_margin = (x["position_accuracy_degradation"] - p["motor_lock_accuracy_threshold"]) / (
            1.0 - p["motor_lock_accuracy_threshold"]
        )
        motor_lock_es = float(min(
            max(0.0, min(1.0, lock_error_margin)),
            max(0.0, min(1.0, lock_accuracy_margin))
        ))

        # --- current_overload ---
        co_temp_margin = (p["critical_temperature"] - x["motor_temperature"]) / (
            p["critical_temperature"] - p["ambient_temperature"]
        )
        co_accuracy_margin = (x["position_accuracy_degradation"] - p["current_overload_accuracy_threshold"]) / (
            1.0 - p["current_overload_accuracy_threshold"]
        )
        current_overload_es = float(min(
            max(0.0, min(1.0, co_temp_margin)),
            max(0.0, min(1.0, co_accuracy_margin))
        ))

        return {
            "motor_overheat": motor_overheat_es,
            "excessive_velocity_error": excessive_vel_err_es,
            "overspeed": overspeed_es,
            "short_time_overload": short_time_overload_es,
            "motor_lock": motor_lock_es,
            "current_overload": current_overload_es,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        motor_overheat = bool(x["motor_temperature"] >= p["overheat_threshold"])
        excessive_velocity_error = bool(x["velocity_error_accumulation"] >= p["velocity_error_threshold"])
        overspeed = bool(x["commanded_speed_tracking"] >= p["overspeed_threshold"])
        short_time_overload = bool(
            x["motor_temperature"] >= p["short_time_overload_temp_threshold"] and
            x["velocity_error_accumulation"] >= p["short_time_overload_error_threshold"]
        )
        motor_lock = bool(
            x["velocity_error_accumulation"] >= p["motor_lock_error_threshold"] and
            x["position_accuracy_degradation"] <= p["motor_lock_accuracy_threshold"]
        )
        current_overload = bool(
            x["motor_temperature"] >= p["critical_temperature"] or
            x["position_accuracy_degradation"] <= p["current_overload_accuracy_threshold"]
        )

        return {
            "motor_overheat": motor_overheat,
            "excessive_velocity_error": excessive_velocity_error,
            "overspeed": overspeed,
            "short_time_overload": short_time_overload,
            "motor_lock": motor_lock,
            "current_overload": current_overload,
        }