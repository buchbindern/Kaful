from progpy import PrognosticsModel
import numpy as np


class SpindleMotor(PrognosticsModel):
    """
    ProgPy PrognosticsModel for a spindle motor drive component.

    Converts electrical voltage commands into mechanical spindle rotation.
    Governs speed, torque, and current delivery to the spindle.
    Subject to thermal accumulation, velocity control degradation,
    and position accuracy loss.
    """

    # --- Class-level port declarations (REQUIRED by ProgPy) ---
    inputs = ["motor_voltage_command"]

    states = [
        "motor_temperature",
        "velocity_error_accumulation",
        "position_accuracy_degradation",
        "bearing_wear",
        "grease_degradation",
        "preload_degradation",
        "contamination_level",
        "actual_speed_tracked",
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

    # --- Units ---
    units = {
        # inputs
        "motor_voltage_command": "%",
        # states
        "motor_temperature": "celsius",
        "velocity_error_accumulation": "rpm",
        "position_accuracy_degradation": "dimensionless",
        "bearing_wear": "dimensionless",
        "grease_degradation": "dimensionless",
        "preload_degradation": "dimensionless",
        "contamination_level": "dimensionless",
        "actual_speed_tracked": "rpm",
        # outputs
        "actual_spindle_speed": "rpm",
        "zero_speed_detection": "boolean",
        "motor_current": "a",
        "drive_torque": "Nm",
    }

    # --- Default parameters ---
    default_parameters = {
        # Thermal
        "motor_thermal_gain": 0.5,           # celsius/(%·s)
        "motor_thermal_dissipation": 0.05,   # 1/s
        "ambient_temperature": 25.0,         # celsius
        "overheat_threshold": 140.0,         # celsius
        "critical_temperature": 150.0,       # celsius
        # Velocity error
        "velocity_error_buildup_rate": 0.5,  # rpm/(%·s)
        "velocity_error_decay_rate": 0.1,    # 1/s
        "max_velocity_error": 500.0,         # rpm
        "critical_position_error": 2000.0,   # rpm
        # Degradation rates
        "bearing_wear_rate": 0.02,                       # 1/cycle
        "critical_bearing_wear": 0.2,                    # dimensionless
        "preload_degradation_rate": 0.015,               # 1/cycle
        "grease_degradation_rate": 0.03,                 # 1/cycle
        "contamination_buildup_rate": 0.01,              # 1/cycle
        "position_accuracy_degradation_rate": 0.02,      # 1/cycle
        # Speed
        "speed_scale_factor": 327.67,        # rpm/%
        "speed_tracking_coefficient": 0.5,   # dimensionless
        "zero_speed_threshold": 0.75,        # rpm
        "overspeed_threshold": 32767.0,      # rpm
        "speed_arrival_threshold": 15.0,     # rpm
        # Current / torque
        "nominal_current_scale": 50.0,       # a
        "current_overload_threshold": 90.0,  # a
        "nominal_torque_scale": 100.0,       # Nm
        # Event thresholds
        "short_time_overload_temp_threshold": 110.0,     # celsius
        "short_time_overload_bearing_threshold": 0.4,    # dimensionless
        "motor_lock_combined_threshold": 0.25,           # dimensionless
        # Initial state values (used by initialize)
        "x0": {
            "motor_temperature": 25.0,
            "velocity_error_accumulation": 0.0,
            "position_accuracy_degradation": 1.0,
            "bearing_wear": 1.0,
            "grease_degradation": 1.0,
            "preload_degradation": 1.0,
            "contamination_level": 1.0,
            "actual_speed_tracked": 0.0,
        },
    }

    # ------------------------------------------------------------------ #
    #  initialize                                                          #
    # ------------------------------------------------------------------ #
    def initialize(self, u=None, z=None):
        """Return initial state container using spec initial values."""
        x0 = dict(self.parameters["x0"])  # copy defaults

        # Optionally seed actual_speed_tracked from input if available
        if u is not None and u.get("motor_voltage_command") is not None:
            pass  # keep default; speed starts at 0 regardless of command

        return self.StateContainer(x0)

    # ------------------------------------------------------------------ #
    #  next_state                                                          #
    # ------------------------------------------------------------------ #
    def next_state(self, x, u, dt):
        params = self.parameters

        # --- Safely read input ---
        v_cmd = u["motor_voltage_command"] if u is not None else 0.0
        if v_cmd is None:
            v_cmd = 0.0
        v_norm = v_cmd / 100.0  # normalised [0, 1]

        # --- Current state values ---
        T = x["motor_temperature"]
        vel_err = x["velocity_error_accumulation"]
        pos_acc = x["position_accuracy_degradation"]
        bw = x["bearing_wear"]
        gd = x["grease_degradation"]
        pd = x["preload_degradation"]
        cl = x["contamination_level"]
        spd = x["actual_speed_tracked"]

        T_amb = params["ambient_temperature"]
        T_crit = params["critical_temperature"]

        # ---- motor_temperature (accumulation) ----
        dT = (
            params["motor_thermal_gain"] * v_norm
            - params["motor_thermal_dissipation"] * (T - T_amb)
        ) * dt
        T_new = float(np.clip(T + dT, 25.0, 200.0))

        # ---- velocity_error_accumulation (accumulation) ----
        d_vel = (
            params["velocity_error_buildup_rate"] * v_norm * (1.0 - pos_acc)
            - params["velocity_error_decay_rate"] * vel_err
        ) * dt
        vel_err_new = float(np.clip(vel_err + d_vel, 0.0, 5000.0))

        # ---- position_accuracy_degradation (degradation) ----
        # rate * normalized_voltage * dt
        pos_acc_new = float(np.clip(
            pos_acc - params["position_accuracy_degradation_rate"] * v_norm * dt,
            0.0, 1.0
        ))

        # ---- bearing_wear (degradation) ----
        bw_new = float(np.clip(
            bw - params["bearing_wear_rate"] * v_norm * dt,
            0.0, 1.0
        ))

        # ---- grease_degradation (degradation) ----
        # driver: normalized temperature above ambient
        thermal_stress = (T - T_amb) / (T_crit - T_amb)
        thermal_stress = float(np.clip(thermal_stress, 0.0, 1.0))
        gd_new = float(np.clip(
            gd - params["grease_degradation_rate"] * thermal_stress * dt,
            0.0, 1.0
        ))

        # ---- preload_degradation (degradation) ----
        # driver: product of normalized temperature and normalized voltage
        preload_driver = thermal_stress * v_norm
        pd_new = float(np.clip(
            pd - params["preload_degradation_rate"] * preload_driver * dt,
            0.0, 1.0
        ))

        # ---- contamination_level (degradation) ----
        # driver: (1 - bearing_wear) — particle generation increases as bearing degrades
        contamination_driver = float(np.clip(1.0 - bw, 0.0, 1.0))
        cl_new = float(np.clip(
            cl - params["contamination_buildup_rate"] * contamination_driver * dt,
            0.0, 1.0
        ))

        # ---- actual_speed_tracked (tracking) ----
        target_speed = v_cmd * params["speed_scale_factor"]
        coeff = min(params["speed_tracking_coefficient"], 0.5)  # stability cap
        spd_new = float(np.clip(
            spd + coeff * (target_speed - spd) * dt,
            0.0, 32767.0
        ))

        return self.StateContainer({
            "motor_temperature": T_new,
            "velocity_error_accumulation": vel_err_new,
            "position_accuracy_degradation": pos_acc_new,
            "bearing_wear": bw_new,
            "grease_degradation": gd_new,
            "preload_degradation": pd_new,
            "contamination_level": cl_new,
            "actual_speed_tracked": spd_new,
        })

    # ------------------------------------------------------------------ #
    #  output                                                              #
    # ------------------------------------------------------------------ #
    def output(self, x):
        params = self.parameters

        pos_acc = x["position_accuracy_degradation"]
        vel_err = x["velocity_error_accumulation"]
        spd = x["actual_speed_tracked"]
        bw = x["bearing_wear"]
        gd = x["grease_degradation"]
        cl = x["contamination_level"]
        pd = x["preload_degradation"]
        T = x["motor_temperature"]

        # actual_spindle_speed
        raw_speed = spd * pos_acc - vel_err * (1.0 - pos_acc)
        actual_speed = float(np.clip(raw_speed, 0.0, 32767.0))

        # zero_speed_detection
        zero_speed = bool(actual_speed <= params["zero_speed_threshold"])

        # motor_current — needs voltage command; approximate from tracked speed
        # We do NOT have u here, so we back-calculate v_norm from actual_speed_tracked
        # using speed_scale_factor as a proxy.
        v_norm_approx = float(np.clip(
            spd / (params["speed_scale_factor"] * 100.0 + 1e-9),
            0.0, 1.0
        ))
        # Use speed_scale_factor: target = v_cmd * speed_scale_factor
        # => v_cmd = spd / speed_scale_factor  (in %)
        v_cmd_approx = spd / (params["speed_scale_factor"] + 1e-9)
        v_norm_approx = float(np.clip(v_cmd_approx / 100.0, 0.0, 1.0))

        motor_current = (
            v_norm_approx
            * params["nominal_current_scale"]
            * (1.0
               + (1.0 - bw) * 0.5
               + (1.0 - gd) * 0.3
               + (1.0 - cl) * 0.2)
        )
        motor_current = float(np.clip(motor_current, 0.0, 100.0))

        # drive_torque
        drive_torque = (
            v_norm_approx
            * params["nominal_torque_scale"]
            * bw * gd * pd
        )
        drive_torque = float(np.clip(drive_torque, 0.0, None))

        return self.OutputContainer({
            "actual_spindle_speed": actual_speed,
            "zero_speed_detection": zero_speed,
            "motor_current": motor_current,
            "drive_torque": drive_torque,
        })

    # ------------------------------------------------------------------ #
    #  event_state                                                         #
    # ------------------------------------------------------------------ #
    def event_state(self, x) -> dict:
        params = self.parameters

        T = x["motor_temperature"]
        vel_err = x["velocity_error_accumulation"]
        pos_acc = x["position_accuracy_degradation"]
        bw = x["bearing_wear"]
        gd = x["grease_degradation"]
        cl = x["contamination_level"]
        spd = x["actual_speed_tracked"]

        # motor_overheat (SP0001)
        # 1.0 when T=0, 0.0 when T=overheat_threshold
        overheat_es = (params["overheat_threshold"] - T) / params["overheat_threshold"]
        overheat_es = float(np.clip(overheat_es, 0.0, 1.0))

        # excessive_velocity_error (SP0002)
        # 1.0 when vel_err=0 and pos_acc=1.0
        vel_err_es = 1.0 - (vel_err / params["critical_position_error"]) * (1.0 - pos_acc + 0.01)
        vel_err_es = float(np.clip(vel_err_es, 0.0, 1.0))

        # overspeed (SP0007)
        # 1.0 when spd=0, 0.0 when spd=overspeed_threshold
        overspeed_es = (params["overspeed_threshold"] - spd) / params["overspeed_threshold"]
        overspeed_es = float(np.clip(overspeed_es, 0.0, 1.0))

        # short_time_overload (SP0029)
        temp_part = (
            (params["short_time_overload_temp_threshold"] - T)
            / params["short_time_overload_temp_threshold"]
        )
        bearing_part = (
            (bw - params["short_time_overload_bearing_threshold"])
            / (1.0 - params["short_time_overload_bearing_threshold"])
        )
        sto_es = float(np.clip(min(temp_part, bearing_part), 0.0, 1.0))

        # motor_lock (SP0031)
        thresh = params["motor_lock_combined_threshold"]
        denom = 1.0 - thresh
        lock_bw = (bw - thresh) / denom
        lock_cl = (cl - thresh) / denom
        lock_gd = (gd - thresh) / denom
        lock_es = float(np.clip(min(lock_bw, lock_cl, lock_gd), 0.0, 1.0))

        # current_overload (SP0054)
        # Compute motor_current from output (uses state-based approximation)
        z = self.output(x)
        motor_current = z["motor_current"]
        current_es = (
            (params["current_overload_threshold"] - motor_current)
            / params["current_overload_threshold"]
        )
        current_es = float(np.clip(current_es, 0.0, 1.0))

        return {
            "motor_overheat": overheat_es,
            "excessive_velocity_error": vel_err_es,
            "overspeed": overspeed_es,
            "short_time_overload": sto_es,
            "motor_lock": lock_es,
            "current_overload": current_es,
        }

    # ------------------------------------------------------------------ #
    #  threshold_met                                                       #
    # ------------------------------------------------------------------ #
    def threshold_met(self, x) -> dict:
        es = self.event_state(x)
        return {
            "motor_overheat": bool(es["motor_overheat"] <= 0.0),
            "excessive_velocity_error": bool(es["excessive_velocity_error"] <= 0.0),
            "overspeed": bool(es["overspeed"] <= 0.0),
            "short_time_overload": bool(es["short_time_overload"] <= 0.0),
            "motor_lock": bool(es["motor_lock"] <= 0.0),
            "current_overload": bool(es["current_overload"] <= 0.0),
        }