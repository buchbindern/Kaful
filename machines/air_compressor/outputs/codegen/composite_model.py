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

composite_model = CompositeModel(
    models=models,
    connections=connections,
)

_external_inputs = {
    'elektronikon_regulator_component.loading_pressure': 6.75,
    'elektronikon_regulator_component.unloading_pressure': 7.25,
    'elektronikon_regulator_component.pressure_regulation_range': 0.65,
    'elektronikon_regulator_component.minimum_stop_time': 15.5,
}

_valid_external_inputs = {
    k: v for k, v in _external_inputs.items() if k in composite_model.inputs
}

_unconnected_inputs_with_defaults = {}
for inp in composite_model.inputs:
    if inp in _valid_external_inputs:
        _unconnected_inputs_with_defaults[inp] = _valid_external_inputs[inp]
    else:
        _unconnected_inputs_with_defaults[inp] = 0.0


def future_loading_eqn(t, x=None):
    return composite_model.InputContainer(_unconnected_inputs_with_defaults)