from progpy import PrognosticsModel


class TemperatureSensor(PrognosticsModel):
    """
    Physics-based digital twin of the motor temperature sensor.
    Models sensor accuracy and signal quality degradation over time and
    environmental stress, produces a motor temperature reading used for
    thermal protection, and raises events for motor overheat and sensor
    disconnection.
    """

    inputs = []
    states = [
        "sensor_accuracy_degradation",
        "signal_quality_degradation",
        "true_motor_temperature",
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
        "true_motor_temperature": "c",
        "motor_temperature": "c",
    }

    default_parameters = {
        "sensor_accuracy_degradation_rate": 0.015,
        "signal_quality_degradation_rate": 0.03,
        "temperature_lag_coefficient": 0.1,
        "nominal_operating_temperature": 75.0,
        "overheat_threshold": 140.0,
        "critical_temperature": 150.0,
        "sensor_disconnection_threshold": 0.1,
        "accuracy_bias_scale": 10.0,
        "signal_noise_scale": 5.0,
        "thermal_cycling_driver_value": 1.0,
        "thermal_stress_oxidation_driver_value": 1.0,
        "x0": {
            "sensor_accuracy_degradation": 1.0,
            "signal_quality_degradation": 1.0,
            "true_motor_temperature": 25.0,
        },
    }

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            "sensor_accuracy_degradation": self.parameters["x0"]["sensor_accuracy_degradation"],
            "signal_quality_degradation": self.parameters["x0"]["signal_quality_degradation"],
            "true_motor_temperature": self.parameters["x0"]["true_motor_temperature"],
        })

    def dx(self, x, u):
        params = self.parameters

        # sensor_accuracy_degradation: degradation state
        # Rate is per 1000 operating hours convention: rate_per_step = rate / 1000.0
        sad_rate = params["sensor_accuracy_degradation_rate"] / 1000.0
        sad_driver = params["thermal_cycling_driver_value"]
        d_sad = -sad_rate * sad_driver

        # signal_quality_degradation: degradation state
        sqd_rate = params["signal_quality_degradation_rate"] / 1000.0
        sqd_driver = params["thermal_stress_oxidation_driver_value"]
        d_sqd = -sqd_rate * sqd_driver

        # true_motor_temperature: tracking state
        lag_coeff = min(params["temperature_lag_coefficient"], 0.5)
        target = params["nominal_operating_temperature"]
        d_tmt = lag_coeff * (target - x["true_motor_temperature"])

        return self.StateContainer({
            "sensor_accuracy_degradation": d_sad,
            "signal_quality_degradation": d_sqd,
            "true_motor_temperature": d_tmt,
        })

    def next_state(self, x, u, dt):
        params = self.parameters

        # sensor_accuracy_degradation: degradation state
        sad_rate = params["sensor_accuracy_degradation_rate"] / 1000.0
        sad_driver = params["thermal_cycling_driver_value"]
        new_sad = x["sensor_accuracy_degradation"] - sad_rate * sad_driver * dt
        new_sad = max(0.0, min(1.0, new_sad))

        # signal_quality_degradation: degradation state
        sqd_rate = params["signal_quality_degradation_rate"] / 1000.0
        sqd_driver = params["thermal_stress_oxidation_driver_value"]
        new_sqd = x["signal_quality_degradation"] - sqd_rate * sqd_driver * dt
        new_sqd = max(0.0, min(1.0, new_sqd))

        # true_motor_temperature: tracking state
        lag_coeff = min(params["temperature_lag_coefficient"], 0.5)
        target = params["nominal_operating_temperature"]
        new_tmt = x["true_motor_temperature"] + lag_coeff * (target - x["true_motor_temperature"]) * dt
        new_tmt = max(-40.0, min(200.0, new_tmt))

        return self.StateContainer({
            "sensor_accuracy_degradation": new_sad,
            "signal_quality_degradation": new_sqd,
            "true_motor_temperature": new_tmt,
        })

    def output(self, x):
        params = self.parameters

        true_temp = x["true_motor_temperature"]
        accuracy_bias = params["accuracy_bias_scale"] * (1.0 - x["sensor_accuracy_degradation"])
        signal_noise = params["signal_noise_scale"] * (1.0 - x["signal_quality_degradation"])
        motor_temperature = true_temp + accuracy_bias + signal_noise

        return self.OutputContainer({
            "motor_temperature": motor_temperature,
        })

    def event_state(self, x) -> dict:
        params = self.parameters

        # motor_overheat: based on true_motor_temperature
        overheat_thresh = params["overheat_threshold"]
        critical_temp = params["critical_temperature"]
        temp_range = critical_temp - overheat_thresh
        overheat_es = 1.0 - max(0.0, x["true_motor_temperature"] - overheat_thresh) / temp_range
        overheat_es = max(0.0, min(1.0, overheat_es))

        # temperature_sensor_disconnected: based on signal_quality_degradation
        disc_thresh = params["sensor_disconnection_threshold"]
        healthy_range = 1.0 - disc_thresh
        disc_es = (x["signal_quality_degradation"] - disc_thresh) / healthy_range
        disc_es = max(0.0, min(1.0, disc_es))

        return {
            "motor_overheat": overheat_es,
            "temperature_sensor_disconnected": disc_es,
        }

    def threshold_met(self, x) -> dict:
        es = self.event_state(x)
        return {
            "motor_overheat": bool(es["motor_overheat"] <= 0.0),
            "temperature_sensor_disconnected": bool(es["temperature_sensor_disconnected"] <= 0.0),
        }