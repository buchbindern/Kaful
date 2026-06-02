import sys
import os

# Add component directory to path
component_dir = "/Users/nechamab/Documents/Kaful/machines/cdv/outputs/codegen/components"
if component_dir not in sys.path:
    sys.path.insert(0, component_dir)

# Import component classes
from process_tube import ProcessTube
from heating_zones import HeatingZones
from vacuum_pump import VacuumPump
from mass_flow_controllers import MassFlowControllers
from pressure_control_valve import PressureControlValve
from temperature_controllers import TemperatureControllers
from gas_delivery_system import GasDeliverySystem

from progpy import CompositeModel

# Instantiate each component
process_tube = ProcessTube()
heating_zones = HeatingZones()
vacuum_pump = VacuumPump()
mass_flow_controllers = MassFlowControllers()
pressure_control_valve = PressureControlValve()
temperature_controllers = TemperatureControllers()
gas_delivery_system = GasDeliverySystem()

# Build models list
models = [
    ("process_tube_component", process_tube),
    ("heating_zones_component", heating_zones),
    ("vacuum_pump_component", vacuum_pump),
    ("mass_flow_controllers_component", mass_flow_controllers),
    ("pressure_control_valve_component", pressure_control_valve),
    ("temperature_controllers_component", temperature_controllers),
    ("gas_delivery_system_component", gas_delivery_system),
]

# Verified connections — used exactly as provided
connections = [
    ("gas_delivery_system_component.nitrogen_flow_rate", "process_tube_component.nitrogen_flow_rate"),
    ("gas_delivery_system_component.silane_flow_rate", "process_tube_component.silane_flow_rate"),
]

# Instantiate CompositeModel
composite_model = CompositeModel(
    models,
    connections=connections,
)

# External input nominal values
_external_inputs_nominal = {
    "mass_flow_controllers_component.flow_setpoint": 60.0,
    "process_tube_component.tube_name": 1.0,
    "vacuum_pump_component.main_vacuum_valve": 1.0,
    "mass_flow_controllers_component.valve_override_state": 1.0,
    "gas_delivery_system_component.n2_purge_valve": 0.0,
    "gas_delivery_system_component.process_valve": 1.0,
}

# Cache the set of valid composite inputs
_valid_composite_inputs = set(composite_model.inputs)


def future_loading_eqn(t, x=None):
    load = {}
    for key, value in _external_inputs_nominal.items():
        if key in _valid_composite_inputs:
            load[key] = value
    return composite_model.InputContainer(load)