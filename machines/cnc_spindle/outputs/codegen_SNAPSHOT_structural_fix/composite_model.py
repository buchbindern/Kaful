import sys
import os

# Add component directory to sys.path
component_dir = "/Users/nechamab/Documents/Kaful/machines/cnc_spindle/outputs/codegen/components"
if component_dir not in sys.path:
    sys.path.insert(0, component_dir)

# Import each component class
from angular_contact_bearings import AngularContactBearings
from spindle_motor import SpindleMotor
from spindle_amplifier import SpindleAmplifier
from position_coder import PositionCoder
from temperature_sensor import TemperatureSensor
from cylindrical_roller_bearings import CylindricalRollerBearings
from tool_interface import ToolInterface

from progpy import CompositeModel

# Instantiate each component
angular_contact_bearings = AngularContactBearings()
spindle_motor = SpindleMotor()
spindle_amplifier = SpindleAmplifier()
position_coder = PositionCoder()
temperature_sensor = TemperatureSensor()
cylindrical_roller_bearings = CylindricalRollerBearings()
tool_interface = ToolInterface()

# Build models list with named tuples
models = [
    ("angular_contact_bearings_component", angular_contact_bearings),
    ("spindle_motor_component", spindle_motor),
    ("spindle_amplifier_component", spindle_amplifier),
    ("position_coder_component", position_coder),
    ("temperature_sensor_component", temperature_sensor),
    ("cylindrical_roller_bearings_component", cylindrical_roller_bearings),
    ("tool_interface_component", tool_interface),
]

# Use EXACTLY the verified connections
connections = [
    (
        "angular_contact_bearings_component.lubrication_film",
        "cylindrical_roller_bearings_component.lubrication_film",
    )
]

# Instantiate CompositeModel
composite_model = CompositeModel(models, connections=connections)

# Define external input nominal values
_external_input_nominals = {
    "spindle_motor_component.motor_voltage_command": 45.0,
    "spindle_amplifier_component.spindle_speed_command": 16383.5,
    "spindle_amplifier_component.torque_command": 50.0,
    "spindle_amplifier_component.motor_voltage": 50.0,
    "spindle_amplifier_component.acceleration_deceleration_time": 127.5,
    "spindle_amplifier_component.current_command": 0.0,
    "spindle_amplifier_component.maximum_motor_speed": 16383.5,
    "spindle_amplifier_component.motor_power_off_delay": 500.0,
    "spindle_amplifier_component.torque_limitation_active": 0.5,
    "spindle_amplifier_component.acceleration_value": 16383.5,
    "spindle_amplifier_component.deceleration_time_constant": 127.5,
    "spindle_amplifier_component.position_pulses": 4096.0,
    "position_coder_component.orientation_stop_position": 2047.5,
    "spindle_amplifier_component.operation_mode": 0.0,
}

# Build the set of valid composite inputs once
_valid_composite_inputs = set(composite_model.inputs)

# Filter nominal values to only include keys that are actual composite inputs
_filtered_nominals = {
    k: v for k, v in _external_input_nominals.items() if k in _valid_composite_inputs
}


def future_loading_eqn(t, x=None):
    return composite_model.InputContainer(_filtered_nominals)