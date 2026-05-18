import sys
sys.path.insert(0, '/Users/nechamab/Documents/Kaful/machines/eversys_coffee_machine/outputs/codegen/components')

from progpy import CompositeModel

from ceramic_grinders import CeramicGrinders
from brewing_chamber import BrewingChamber
from coffee_boiler import CoffeeBoiler
from steam_boiler import SteamBoiler
from water_pump import WaterPump
from milk_system import MilkSystem
from powder_unit import PowderUnit
from cleaning_system import CleaningSystem

# Instantiate each component
ceramic_grinders = CeramicGrinders()
brewing_chamber = BrewingChamber()
coffee_boiler = CoffeeBoiler()
steam_boiler = SteamBoiler()
water_pump = WaterPump()
milk_system = MilkSystem()
powder_unit = PowderUnit()
cleaning_system = CleaningSystem()

# Build models list with named tuples
models = [
    ('ceramic_grinders_component', ceramic_grinders),
    ('brewing_chamber_component', brewing_chamber),
    ('coffee_boiler_component', coffee_boiler),
    ('steam_boiler_component', steam_boiler),
    ('water_pump_component', water_pump),
    ('milk_system_component', milk_system),
    ('powder_unit_component', powder_unit),
    ('cleaning_system_component', cleaning_system),
]

# Verified connections — used exactly as provided
connections = [
    ('ceramic_grinders_component.ground_coffee', 'brewing_chamber_component.ground_coffee'),
    ('coffee_boiler_component.brewing_water', 'brewing_chamber_component.brewing_water'),
    ('steam_boiler_component.steam_flow', 'milk_system_component.steam_flow'),
    ('water_pump_component.water_supply', 'steam_boiler_component.water_supply'),
]

# Instantiate the CompositeModel
composite_model = CompositeModel(
    models,
    connections=connections,
)

# External input nominal values
_external_inputs = {
    'brewing_chamber_component.cake_thickness': 14.0,
    'brewing_chamber_component.tamping_pressure': 20.0,
    'brewing_chamber_component.pre_infusion_time': 0.8,
    'brewing_chamber_component.relax_time': 2.0,
    'brewing_chamber_component.second_tamping': 2.0,
    'brewing_chamber_component.coffee_cycles': 1.0,
    'brewing_chamber_component.bypass_quantity': 0.0,
    'coffee_boiler_component.water_supply': 3.25,
    'coffee_boiler_component.water_quantity': 500.0,
    'steam_boiler_component.water_temperature': 80.0,
    'steam_boiler_component.hot_water_duration': 500.0,
    'milk_system_component.milk_quantity': 50.0,
    'milk_system_component.stop_temperature': 65.0,
    'milk_system_component.foam_texture': 65.0,
    'milk_system_component.milk_delay_time': 0.0,
    'powder_unit_component.powder_density': 3.0,
    'ceramic_grinders_component.bean_flow': 0.5,
    'milk_system_component.milk_temperature_setting': 65.0,
}

# Pre-compute valid external inputs (only those that are actual unconnected inputs of the composite)
_valid_inputs = {k: v for k, v in _external_inputs.items() if k in composite_model.inputs}


def future_loading_eqn(t, x=None):
    return composite_model.InputContainer(_valid_inputs)