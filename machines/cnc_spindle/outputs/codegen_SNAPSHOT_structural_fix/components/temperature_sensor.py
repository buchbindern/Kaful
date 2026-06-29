from progpy import PrognosticsModel


class TemperatureSensor(PrognosticsModel):
    """
    Physics-based digital twin of the motor temperature sensor.
    Models sensor accuracy and signal quality degradation over time,
    reports motor temperature, and detects overheating and sensor
    disconnection fault conditions.
    """

    inputs = []
    states = [
        "sensor_accuracy_degradation",
        "signal_quality_degradation",
        "reported_temperature",
    ]
    outputs = [
        "motor_temperature",
    ]
    events = [
        "motor_overheat",
        "temperature_sensor_disconnected",
    ]

    units = {
        "sensor_accuracy_degradation": "dimensionless",
        "signal_quality_degradation": "dimensionless",
        "reported_temperature": "c",
        "motor_temperature": "c",
    }

    default_parameters = {
        "sensor_accuracy_degradation_rate": 0.015,
        "signal_quality_degradation_rate": 0.03,
        "thermal_cycling_and_wear_driver": 1.0,
        "thermal_stress_and_oxidation_driver": 1.0,
        "temperature_lag_coefficient": 0.3,
        "nominal_ambient_temperature": 25.0,
        "max_sensor_bias": 20.0,
        "overheat_threshold": 140.0,
        "critical_temperature": 150.0,
        "signal_disconnection_threshold": 0.1,
        "min_reported_temperature": -40.0,
        "max_reported_temperature": 200.0,
        "x0": {
            "sensor_accuracy_degradation": 1.0,
            "signal_quality_degradation": 1.0,
            "reported_temperature": 25.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "sensor_accuracy_degradation": self.parameters["x0"]["sensor_accuracy_degradation"],
            "signal_quality_degradation": self.parameters["x0"]["signal_quality_degradation"],
            "reported_temperature": self.parameters["x0"]["reported_temperature"],
        })

    def next_state(self, x, u, dt):
        p = self.parameters

        # --- sensor_accuracy_degradation (DEGRADATION) ---
        sad_rate = p["sensor_accuracy_degradation_rate"]
        driver_sad = p["thermal_cycling_and_wear_driver"]
        new_sad = x["sensor_accuracy_degradation"] - sad_rate * driver_sad * dt
        new_sad = max(0.0, min(1.0, new_sad))

        # --- signal_quality_degradation (DEGRADATION) ---
        sqd_rate = p["signal_quality_degradation_rate"]
        driver_sqd = p["thermal_stress_and_oxidation_driver"]
        new_sqd = x["signal_quality_degradation"] - sqd_rate * driver_sqd * dt
        new_sqd = max(0.0, min(1.0, new_sqd))

        # --- reported_temperature (TRACKING) ---
        raw_coeff = p["temperature_lag_coefficient"]
        coeff = min(raw_coeff, 0.5)  # mandatory stability cap
        target = p["nominal_ambient_temperature"]
        new_rt = x["reported_temperature"] + coeff * (target - x["reported_temperature"]) * dt
        new_rt = max(p["min_reported_temperature"], min(p["max_reported_temperature"], new_rt))

        return self.StateContainer({
            "sensor_accuracy_degradation": new_sad,
            "signal_quality_degradation": new_sqd,
            "reported_temperature": new_rt,
        })

    def output(self, x):
        p = self.parameters

        sad = x["sensor_accuracy_degradation"]
        sqd = x["signal_quality_degradation"]
        rt = x["reported_temperature"]

        # If signal quality is below disconnection threshold, report sentinel
        if sqd < p["signal_disconnection_threshold"]:
            motor_temp = -999.0
        else:
            bias_term = p["max_sensor_bias"] * (1.0 - sad)
            motor_temp = rt + bias_term

        return self.OutputContainer({
            "motor_temperature": motor_temp,
        })

    def event_state(self, x) -> dict:
        p = self.parameters

        # motor_overheat (SP0001)
        rt = x["reported_temperature"]
        overheat_thresh = p["overheat_threshold"]
        critical_temp = p["critical_temperature"]
        denom_overheat = critical_temp - overheat_thresh
        if denom_overheat == 0.0:
            overheat_es = 0.0 if rt >= critical_temp else 1.0
        else:
            overheat_es = 1.0 - max(0.0, (rt - overheat_thresh) / denom_overheat)
        overheat_es = max(0.0, min(1.0, overheat_es))

        # temperature_sensor_disconnected (SP0006)
        sqd = x["signal_quality_degradation"]
        disc_thresh = p["signal_disconnection_threshold"]
        denom_disc = 1.0 - disc_thresh
        if denom_disc == 0.0:
            disc_es = 0.0 if sqd <= disc_thresh else 1.0
        else:
            disc_es = (sqd - disc_thresh) / denom_disc
        disc_es = max(0.0, min(1.0, disc_es))

        return {
            "motor_overheat": overheat_es,
            "temperature_sensor_disconnected": disc_es,
        }

    def threshold_met(self, x) -> dict:
        p = self.parameters

        rt = x["reported_temperature"]
        sqd = x["signal_quality_degradation"]

        return {
            "motor_overheat": bool(rt >= p["critical_temperature"]),
            "temperature_sensor_disconnected": bool(sqd <= p["signal_disconnection_threshold"]),
        }