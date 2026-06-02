import sys
sys.path.insert(0, '/Users/nechamab/Documents/Kaful/machines/air_compressor/outputs/codegen/components')

from progpy import CompositeModel

from compressor_element import CompressorElement
from air_receiver_oil_separator import AirReceiverOilSeparator
from drive_motor import DriveMotor
from inlet_valve import InletValve
from air_filter import AirFilter
from oil_cooler import OilCooler
from air_cooler import AirCooler
from minimum_pressure_valve import MinimumPressureValve
from oil_stop_valve import OilStopValve
from elektronikon_regulator import ElektronikonRegulator
from solenoid_valve import SolenoidValve
from condensate_trap import CondensateTrap
from safety_valve import SafetyValve
from oil_separator_element import OilSeparatorElement

# Instantiate each component
compressor_element = CompressorElement()
air_receiver_oil_separator = AirReceiverOilSeparator()
drive_motor = DriveMotor()
inlet_valve = InletValve()
air_filter = AirFilter()
oil_cooler = OilCooler()
air_cooler = AirCooler()
minimum_pressure_valve = MinimumPressureValve()
oil_stop_valve = OilStopValve()
elektronikon_regulator = ElektronikonRegulator()
solenoid_valve = SolenoidValve()
condensate_trap = CondensateTrap()
safety_valve = SafetyValve()
oil_separator_element = OilSeparatorElement()

# Build models list with named tuples
models = [
    ('air_filter_component', air_filter),
    ('compressor_element_component', compressor_element),
    ('air_receiver_oil_separator_component', air_receiver_oil_separator),
    ('drive_motor_component', drive_motor),
    ('inlet_valve_component', inlet_valve),
    ('oil_cooler_component', oil_cooler),
    ('air_cooler_component', air_cooler),
    ('minimum_pressure_valve_component', minimum_pressure_valve),
    ('oil_stop_valve_component', oil_stop_valve),
    ('elektronikon_regulator_component', elektronikon_regulator),
    ('solenoid_valve_component', solenoid_valve),
    ('condensate_trap_component', condensate_trap),
    ('safety_valve_component', safety_valve),
    ('oil_separator_element_component', oil_separator_element),
]

# Verified connections — used exactly as provided
connections = [
    ('air_filter_component.filtered_air_flow', 'compressor_element_component.filtered_air_flow'),
    ('compressor_element_component.compressed_air_oil_mixture', 'air_receiver_oil_separator_component.compressed_air_oil_mixture'),
    ('air_receiver_oil_separator_component.separated_compressed_air', 'air_cooler_component.separated_compressed_air'),
    ('air_cooler_component.cooled_compressed_air', 'condensate_trap_component.cooled_compressed_air'),
    ('air_receiver_oil_separator_component.hot_oil_flow', 'oil_cooler_component.hot_oil_flow'),
    ('oil_cooler_component.cooled_oil_flow', 'compressor_element_component.cooled_oil_flow'),
    ('air_receiver_oil_separator_component.control_pressure', 'solenoid_valve_component.control_pressure'),
    ('solenoid_valve_component.pneumatic_control_signal', 'inlet_valve_component.pneumatic_control_signal'),
]

# Build the CompositeModel
composite_model = CompositeModel(
    models=models,
    connections=connections,
)

# External input nominal values
_external_inputs = {
    'elektronikon_regulator_component.loading_pressure': 6.75,
    'elektronikon_regulator_component.unloading_pressure': 7.25,
    'elektronikon_regulator_component.pressure_regulation_range': 0.65,
    'elektronikon_regulator_component.minimum_stop_time': 15.5,
}

# Cache the set of valid composite inputs for fast lookup
_valid_composite_inputs = set(composite_model.inputs)


def future_loading_eqn(t, x=None):
    valid_inputs = {}
    for key, value in _external_inputs.items():
        if key in _valid_composite_inputs:
            valid_inputs[key] = value
    # Also include any other unconnected inputs from components that need values.
    # For components with inputs not covered by connections or external_inputs,
    # we check composite_model.inputs and provide defaults if needed.
    for inp in _valid_composite_inputs:
        if inp not in valid_inputs:
            # Provide a default of 0.0 for any unconnected input not in external_inputs
            valid_inputs[inp] = 0.0
    return composite_model.InputContainer(valid_inputs)