import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SimulatorConfig:
    # Product definitions
    coffee_products = ['espresso', 'americano', 'lungo', 'ristretto']
    milk_products = ['cappuccino', 'latte', 'macchiato', 'flat_white']
    hot_water_products = ['hot_water', 'tea_water']
    powder_products = ['hot_chocolate', 'chai_latte']
    
    # Bean hopper options
    bean_hoppers = ['left', 'right']
    
    # Product parameter ranges
    water_quantity_range = (10, 750)  # ticks
    milk_quantity_range = (0, 100)   # seconds
    cake_thickness_range = (12.0, 16.0)  # mm
    tamping_pressure_range = (15, 25)  # kg
    pre_infusion_range = (0.5, 1.2)  # seconds
    relax_time_range = (1.5, 3.0)  # seconds
    second_tamping_range = (1.0, 3.0)  # mm
    extraction_time_range = (18, 28)  # seconds
    water_temp_range = (75, 85)  # celsius
    stop_temp_range = (60, 70)  # celsius
    foam_texture_range = (40, 90)
    hot_water_duration_range = (5, 60)  # seconds
    powder_density_range = (4.0, 6.0)  # g/100ml
    
    # Degradation parameters
    grinder_wear_rate = 0.0001  # per coffee event
    brewing_wear_rate = 0.00008  # per coffee event
    boiler_scale_rate = 0.00005  # per water operation
    milk_contamination_rate = 0.002  # per milk event
    extraction_efficiency_decay = 0.00003  # per coffee event
    heating_element_decay = 0.000015  # per heating cycle
    
    # Degradation thresholds
    grinder_wear_threshold = 0.8
    brewing_wear_threshold = 0.75
    boiler_scale_threshold = 0.7
    milk_cleanliness_threshold = 0.3
    extraction_efficiency_threshold = 0.6
    heating_efficiency_threshold = 0.7
    
    # Maintenance restoration factors
    cleaning_milk_restoration = 0.9
    cleaning_scale_reduction = 0.3
    service_restoration_factor = 0.6
    filter_change_scale_reduction = 0.4
    
    # Failure probabilities
    base_failure_rate = 0.02
    wear_failure_multiplier = 3.0
    
    # Manual stop probabilities
    base_manual_stop_rate = 0.05
    long_extraction_stop_multiplier = 2.0
    
    # Noise parameters
    base_noise_level = 0.02
    degradation_noise_multiplier = 1.5
    
    # Timing parameters
    base_extraction_time = 22.0  # seconds
    extraction_time_variance = 2.0
    
    # Service intervals
    service_product_interval = 50000
    service_time_interval_days = 365
    filter_change_water_amount = 100000  # arbitrary water units
    
    # Cleaning parameters
    daily_cleaning_probability = 0.8
    cleaning_ball_consumption_rate = 0.1
    detergent_detection_probability = 0.95

@dataclass
class GrinderState:
    wear_level: float = 0.0
    burr_alignment: float = 1.0
    last_service_products: int = 0

@dataclass
class BrewingState:
    chamber_wear: float = 0.0
    piston_wear: float = 0.0
    extraction_efficiency: float = 1.0
    last_service_products: int = 0

@dataclass
class HydraulicState:
    boiler_scale: float = 0.0
    heating_element_efficiency: float = 1.0
    pressure_stability: float = 1.0
    total_water_processed: int = 0
    last_filter_change: int = 0

@dataclass
class MilkState:
    system_cleanliness: float = 1.0
    heating_element_efficiency: float = 1.0
    last_cleaning: int = 0

@dataclass
class PowderState:
    system_cleanliness: float = 1.0
    dispenser_wear: float = 0.0

@dataclass
class CleaningState:
    cleaning_ball_supply: float = 1.0
    detergent_effectiveness: float = 1.0
    cycles_since_cleaning: int = 0

@dataclass
class ControlState:
    sensor_accuracy: float = 1.0
    system_stability: float = 1.0

@dataclass
class MachineState:
    grinder: GrinderState = field(default_factory=GrinderState)
    brewing: BrewingState = field(default_factory=BrewingState)
    hydraulic: HydraulicState = field(default_factory=HydraulicState)
    milk: MilkState = field(default_factory=MilkState)
    powder: PowderState = field(default_factory=PowderState)
    cleaning: CleaningState = field(default_factory=CleaningState)
    control: ControlState = field(default_factory=ControlState)
    total_products: int = 0
    last_service_date: int = 0
    current_day: int = 0

@dataclass
class UsageProfile:
    name: str
    description: str
    operating_hours: str
    daily_events: int
    operation_mix: Dict[str, float]
    maintenance_frequency: float = 1.0

class CoffeeMachineSimulator:
    def __init__(self, config: SimulatorConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        self.usage_profiles = {
            'light_commercial': UsageProfile(
                name='light_commercial',
                description='Low volume office usage',
                operating_hours='8-10 hours/day',
                daily_events=50,
                operation_mix={
                    'coffee_brewing': 0.6,
                    'milk_product_dispensing': 0.25,
                    'hot_water_dispensing': 0.1,
                    'cleaning_cycle': 0.05
                }
            ),
            'medium_commercial': UsageProfile(
                name='medium_commercial',
                description='Medium volume restaurant usage',
                operating_hours='12-14 hours/day',
                daily_events=120,
                operation_mix={
                    'coffee_brewing': 0.5,
                    'milk_product_dispensing': 0.35,
                    'hot_water_dispensing': 0.1,
                    'cleaning_cycle': 0.05
                }
            ),
            'heavy_commercial': UsageProfile(
                name='heavy_commercial',
                description='High volume cafe usage',
                operating_hours='16+ hours/day',
                daily_events=200,
                operation_mix={
                    'coffee_brewing': 0.45,
                    'milk_product_dispensing': 0.4,
                    'hot_water_dispensing': 0.1,
                    'cleaning_cycle': 0.05
                }
            )
        }
    
    def generate_event_times(self, profile: UsageProfile, days: int) -> List[datetime]:
        events = []
        base_time = datetime(2024, 1, 1, 6, 0, 0)
        
        for day in range(days):
            day_start = base_time + timedelta(days=day, hours=6)
            day_end = day_start + timedelta(hours=12)
            
            daily_events = int(self.rng.poisson(profile.daily_events))
            
            for _ in range(daily_events):
                event_time = day_start + timedelta(
                    seconds=self.rng.uniform(0, (day_end - day_start).total_seconds())
                )
                events.append(event_time)
        
        return sorted(events)
    
    def select_operation_type(self, profile: UsageProfile) -> str:
        operations = list(profile.operation_mix.keys())
        probabilities = list(profile.operation_mix.values())
        return self.rng.choice(operations, p=probabilities)
    
    def select_product(self, operation_type: str) -> str:
        if operation_type == 'coffee_brewing':
            return self.rng.choice(self.config.coffee_products)
        elif operation_type == 'milk_product_dispensing':
            return self.rng.choice(self.config.milk_products)
        elif operation_type == 'hot_water_dispensing':
            return self.rng.choice(self.config.hot_water_products)
        elif operation_type == 'powder_product_dispensing':
            return self.rng.choice(self.config.powder_products)
        else:
            return 'cleaning'
    
    def compute_degradation_effect(self, base_value: float, degradation: float, 
                                 noise_level: float = None) -> float:
        if noise_level is None:
            noise_level = self.config.base_noise_level
        
        degradation_factor = 1.0 + degradation * self.config.degradation_noise_multiplier
        noise = self.rng.normal(0, noise_level * degradation_factor)
        
        return base_value * (1.0 - degradation * 0.3) + noise
    
    def generate_coffee_event(self, state: MachineState, operation_type: str, 
                            event_time: datetime) -> Dict[str, Any]:
        product = self.select_product(operation_type)
        
        # Control inputs
        bean_hopper = self.rng.choice(self.config.bean_hoppers)
        double_product = self.rng.choice([True, False], p=[0.2, 0.8])
        
        # Base parameters with degradation effects
        grinder_degradation = state.grinder.wear_level
        brewing_degradation = state.brewing.chamber_wear
        extraction_degradation = 1.0 - state.brewing.extraction_efficiency
        
        base_water_qty = self.rng.uniform(*self.config.water_quantity_range)
        water_quantity = int(self.compute_degradation_effect(
            base_water_qty, extraction_degradation, 0.05
        ))
        
        base_cake_thickness = self.rng.uniform(*self.config.cake_thickness_range)
        cake_thickness = self.compute_degradation_effect(
            base_cake_thickness, grinder_degradation, 0.02
        )
        
        tamping_pressure = int(self.rng.uniform(*self.config.tamping_pressure_range))
        pre_infusion_time = self.rng.uniform(*self.config.pre_infusion_range)
        relax_time = self.rng.uniform(*self.config.relax_time_range)
        second_tamping = self.rng.uniform(*self.config.second_tamping_range)
        
        # Extraction time affected by multiple degradation factors
        base_extraction = self.config.base_extraction_time
        scale_effect = state.hydraulic.boiler_scale * 2.0
        wear_effect = grinder_degradation * 3.0 + brewing_degradation * 2.0
        
        extraction_time = self.compute_degradation_effect(
            base_extraction + scale_effect + wear_effect,
            (grinder_degradation + brewing_degradation) / 2,
            self.config.extraction_time_variance
        )
        
        # Manual stop probability increases with long extraction times
        manual_stop_prob = self.config.base_manual_stop_rate
        if extraction_time > 25:
            manual_stop_prob *= self.config.long_extraction_stop_multiplier
        
        manual_stop_pressed = self.rng.random() < manual_stop_prob
        manual_stop_triggered = manual_stop_pressed
        
        # Timing
        dispensing_start_time = event_time
        if manual_stop_triggered:
            actual_extraction = extraction_time * self.rng.uniform(0.3, 0.8)
        else:
            actual_extraction = extraction_time
        
        dispensing_stop_time = dispensing_start_time + timedelta(seconds=actual_extraction)
        dispensing_end_time = dispensing_stop_time
        
        # Coffee cycles and bypass
        coffee_cycles = 1 if not double_product else 2
        bypass_quantity = int(self.rng.uniform(0, 5))  # Usually 0
        
        return {
            'product_button_pressed': product,
            'dispensing_start_time': dispensing_start_time,
            'dispensing_stop_time': dispensing_stop_time,
            'water_quantity': water_quantity,
            'milk_quantity': None,
            'cake_thickness': round(cake_thickness, 1),
            'tamping_pressure': tamping_pressure,
            'pre_infusion_time': round(pre_infusion_time, 1),
            'relax_time': round(relax_time, 1),
            'second_tamping': round(second_tamping, 1),
            'extraction_time': round(actual_extraction, 1),
            'water_temperature': None,
            'stop_temperature': None,
            'foam_texture': None,
            'milk_delay_time': None,
            'coffee_cycles': coffee_cycles,
            'bypass_quantity': bypass_quantity,
            'bean_hopper_selection': bean_hopper,
            'manual_stop_pressed': manual_stop_pressed,
            'product_key_name': product.title(),
            'product_type': product,
            'milk_temperature_setting': None,
            'double_product': double_product,
            'manual_stop_triggered': manual_stop_triggered,
            'hot_water_duration': None,
            'powder_density': None,
            'dispensing_end_time': dispensing_end_time
        }
    
    def generate_milk_event(self, state: MachineState, operation_type: str, 
                          event_time: datetime) -> Dict[str, Any]:
        product = self.select_product(operation_type)
        
        # Milk system degradation effects
        milk_degradation = 1.0 - state.milk.system_cleanliness
        heating_degradation = 1.0 - state.milk.heating_element_efficiency
        
        # Base parameters
        base_water_qty = self.rng.uniform(30, 60)  # Less water for milk products
        water_quantity = int(self.compute_degradation_effect(
            base_water_qty, milk_degradation * 0.1, 0.03
        ))
        
        base_milk_qty = self.rng.uniform(*self.config.milk_quantity_range)
        milk_quantity = self.compute_degradation_effect(
            base_milk_qty, milk_degradation, 0.1
        )
        
        cake_thickness = self.rng.uniform(*self.config.cake_thickness_range)
        tamping_pressure = int(self.rng.uniform(*self.config.tamping_pressure_range))
        pre_infusion_time = self.rng.uniform(*self.config.pre_infusion_range)
        relax_time = self.rng.uniform(*self.config.relax_time_range)
        second_tamping = self.rng.uniform(*self.config.second_tamping_range)
        
        # Temperature settings affected by heating element degradation
        base_stop_temp = self.rng.uniform(*self.config.stop_temp_range)
        stop_temperature = int(self.compute_degradation_effect(
            base_stop_temp, heating_degradation, 0.02
        ))
        
        # Foam texture affected by milk system cleanliness
        base_foam = self.rng.uniform(*self.config.foam_texture_range)
        foam_texture = int(self.compute_degradation_effect(
            base_foam, milk_degradation, 0.05
        ))
        
        milk_delay_time = self.rng.uniform(0, 2.0)
        milk_temperature_setting = self.rng.choice(['warm', 'hot'])
        
        # Extraction time
        base_extraction = self.config.base_extraction_time
        extraction_time = self.compute_degradation_effect(
            base_extraction, milk_degradation * 0.2, 1.5
        )
        
        # Manual stop
        manual_stop_pressed = self.rng.random() < self.config.base_manual_stop_rate
        manual_stop_triggered = manual_stop_pressed
        
        # Timing
        dispensing_start_time = event_time
        total_time = extraction_time + milk_quantity
        if manual_stop_triggered:
            total_time *= self.rng.uniform(0.4, 0.9)
        
        dispensing_stop_time = dispensing_start_time + timedelta(seconds=total_time)
        dispensing_end_time = dispensing_stop_time
        
        bean_hopper = self.rng.choice(self.config.bean_hoppers)
        double_product = self.rng.choice([True, False], p=[0.15, 0.85])
        coffee_cycles = 1 if not double_product else 2
        
        return {
            'product_button_pressed': product,
            'dispensing_start_time': dispensing_start_time,
            'dispensing_stop_time': dispensing_stop_time,
            'water_quantity': water_quantity,
            'milk_quantity': round(milk_quantity, 1),
            'cake_thickness': round(cake_thickness, 1),
            'tamping_pressure': tamping_pressure,
            'pre_infusion_time': round(pre_infusion_time, 1),
            'relax_time': round(relax_time, 1),
            'second_tamping': round(second_tamping, 1),
            'extraction_time': round(extraction_time, 1),
            'water_temperature': None,
            'stop_temperature': stop_temperature,
            'foam_texture': foam_texture,
            'milk_delay_time': round(milk_delay_time, 1),
            'coffee_cycles': coffee_cycles,
            'bypass_quantity': 0,
            'bean_hopper_selection': bean_hopper,
            'manual_stop_pressed': manual_stop_pressed,
            'product_key_name': product.replace('_', ' ').title(),
            'product_type': product,
            'milk_temperature_setting': milk_temperature_setting,
            'double_product': double_product,
            'manual_stop_triggered': manual_stop_triggered,
            'hot_water_duration': None,
            'powder_density': None,
            'dispensing_end_time': dispensing_end_time
        }
    
    def generate_hot_water_event(self, state: MachineState, operation_type: str, 
                               event_time: datetime) -> Dict[str, Any]:
        product = self.select_product(operation_type)
        
        # Hot water system affected by boiler scale
        scale_degradation = state.hydraulic.boiler_scale
        heating_degradation = 1.0 - state.hydraulic.heating_element_efficiency
        
        # Hot water duration
        base_duration = self.rng.uniform(*self.config.hot_water_duration_range)
        hot_water_duration = self.compute_degradation_effect(
            base_duration, scale_degradation * 0.2, 0.1
        )
        
        # Water temperature affected by heating efficiency
        base_temp = self.rng.uniform(*self.config.water_temp_range)
        water_temperature = int(self.compute_degradation_effect(
            base_temp, heating_degradation, 0.02
        ))
        
        # Manual stop
        manual_stop_pressed = self.rng.random() < (self.config.base_manual_stop_rate * 0.5)
        manual_stop_triggered = manual_stop_pressed
        
        # Timing
        dispensing_start_time = event_time
        actual_duration = hot_water_duration
        if manual_stop_triggered:
            actual_duration *= self.rng.uniform(0.2, 0.8)
        
        dispensing_stop_time = dispensing_start_time + timedelta(seconds=actual_duration)
        dispensing_end_time = dispensing_stop_time
        
        return {
            'product_button_pressed': product,
            'dispensing_start_time': dispensing_start_time,
            'dispensing_stop_time': dispensing_stop_time,
            'water_quantity': None,
            'milk_quantity': None,
            'cake_thickness': None,
            'tamping_pressure': None,
            'pre_infusion_time': None,
            'relax_time': None,
            'second_tamping': None,
            'extraction_time': None,
            'water_temperature': water_temperature,
            'stop_temperature': None,
            'foam_texture': None,
            'milk_delay_time': None,
            'coffee_cycles': None,
            'bypass_quantity': None,
            'bean_hopper_selection': None,
            'manual_stop_pressed': manual_stop_pressed,
            'product_key_name': product.replace('_', ' ').title(),
            'product_type': product,
            'milk_temperature_setting': None,
            'double_product': False,
            'manual_stop_triggered': manual_stop_triggered,
            'hot_water_duration': round(actual_duration, 1),
            'powder_density': None,
            'dispensing_end_time': dispensing_end_time
        }
    
    def generate_powder_event(self, state: MachineState, operation_type: str, 
                            event_time: datetime) -> Dict[str, Any]:
        product = self.select_product(operation_type)
        
        # Powder system degradation
        powder_degradation = 1.0 - state.powder.system_cleanliness
        
        # Base parameters
        base_water_qty = self.rng.uniform(80, 120)
        water_quantity = int(self.compute_degradation_effect(
            base_water_qty, powder_degradation * 0.1, 0.05
        ))
        
        base_powder_density = self.rng.uniform(*self.config.powder_density_range)
        powder_density = self.compute_degradation_effect(
            base_powder_density, powder_degradation, 0.1
        )
        
        water_temperature = int(self.rng.uniform(*self.config.water_temp_range))
        
        # Duration based on water quantity
        duration = water_quantity * 0.1 + self.rng.uniform(5, 15)
        
        # Manual stop
        manual_stop_pressed = self.rng.random() < self.config.base_manual_stop_rate
        manual_stop_triggered = manual_stop_pressed
        
        # Timing
        dispensing_start_time = event_time
        actual_duration = duration
        if manual_stop_triggered:
            actual_duration *= self.rng.uniform(0.3, 0.8)
        
        dispensing_stop_time = dispensing_start_time + timedelta(seconds=actual_duration)
        dispensing_end_time = dispensing_stop_time
        
        return {
            'product_button_pressed': product,
            'dispensing_start_time': dispensing_start_time,
            'dispensing_stop_time': dispensing_stop_time,
            'water_quantity': water_quantity,
            'milk_quantity': None,
            'cake_thickness': None,
            'tamping_pressure': None,
            'pre_infusion_time': None,
            'relax_time': None,
            'second_tamping': None,
            'extraction_time': None,
            'water_temperature': water_temperature,
            'stop_temperature': None,
            'foam_texture': None,
            'milk_delay_time': None,
            'coffee_cycles': None,
            'bypass_quantity': None,
            'bean_hopper_selection': None,
            'manual_stop_pressed': manual_stop_pressed,
            'product_key_name': product.replace('_', ' ').title(),
            'product_type': product,
            'milk_temperature_setting': None,
            'double_product': False,
            'manual_stop_triggered': manual_stop_triggered,
            'hot_water_duration': None,
            'powder_density': round(powder_density, 1),
            'dispensing_end_time': dispensing_end_time
        }
    
    def generate_cleaning_event(self, state: MachineState, event_time: datetime) -> Dict[str, Any]:
        # Cleaning duration varies based on system condition
        base_duration = 720  # 12 minutes
        milk_contamination = 1.0 - state.milk.system_cleanliness
        scale_buildup = state.hydraulic.boiler_scale
        
        duration_multiplier = 1.0 + (milk_contamination + scale_buildup) * 0.3
        cleaning_duration = base_duration * duration_multiplier
        
        dispensing_start_time = event_time
        dispensing_stop_time = dispensing_start_time + timedelta(seconds=cleaning_duration)
        dispensing_end_time = dispensing_stop_time
        
        return {
            'product_button_pressed': 'cleaning',
            'dispensing_start_time': dispensing_start_time,
            'dispensing_stop_time': dispensing_stop_time,
            'water_quantity': None,
            'milk_quantity': None,
            'cake_thickness': None,
            'tamping_pressure': None,
            'pre_infusion_time': None,
            'relax_time': None,
            'second_tamping': None,
            'extraction_time': None,
            'water_temperature': None,
            'stop_temperature': None,
            'foam_texture': None,
            'milk_delay_time': None,
            'coffee_cycles': None,
            'bypass_quantity': None,
            'bean_hopper_selection': None,
            'manual_stop_pressed': False,
            'product_key_name': 'Cleaning',
            'product_type': 'cleaning',
            'milk_temperature_setting': None,
            'double_product': False,
            'manual_stop_triggered': False,
            'hot_water_duration': None,
            'powder_density': None,
            'dispensing_end_time': dispensing_end_time
        }
    
    def update_state_after_event(self, state: MachineState, event: Dict[str, Any], 
                               operation_type: str) -> None:
        # Update product counter
        if operation_type != 'cleaning_cycle':
            state.total_products += 1
        
        # Update degradation based on operation type
        if operation_type in ['coffee_brewing', 'milk_product_dispensing']:
            # Grinder wear (nonlinear - accelerates with existing wear)
            wear_factor = 1.0 + state.grinder.wear_level * 0.5
            state.grinder.wear_level += self.config.grinder_wear_rate * wear_factor
            state.grinder.wear_level = min(state.grinder.wear_level, 1.0)
            
            # Brewing system wear
            state.brewing.chamber_wear += self.config.brewing_wear_rate
            state.brewing.chamber_wear = min(state.brewing.chamber_wear, 1.0)
            
            # Extraction efficiency decay (nonlinear)
            efficiency_factor = state.brewing.extraction_efficiency ** 0.8
            state.brewing.extraction_efficiency -= self.config.extraction_efficiency_decay * efficiency_factor
            state.brewing.extraction_efficiency = max(state.brewing.extraction_efficiency, 0.3)
        
        # Water operations cause scale buildup
        if operation_type in ['coffee_brewing', 'milk_product_dispensing', 'hot_water_dispensing', 'powder_product_dispensing']:
            # Scale buildup (nonlinear - slows as it approaches saturation)
            scale_factor = (1.0 - state.hydraulic.boiler_scale) ** 1.2
            state.hydraulic.boiler_scale += self.config.boiler_scale_rate * scale_factor
            state.hydraulic.boiler_scale = min(state.hydraulic.boiler_scale, 1.0)
            
            state.hydraulic.total_water_processed += 1
            
            # Heating element degradation
            state.hydraulic.heating_element_efficiency -= self.config.heating_element_decay
            state.hydraulic.heating_element_efficiency = max(state.hydraulic.heating_element_efficiency, 0.5)
        
        # Milk operations cause contamination
        if operation_type == 'milk_product_dispensing':
            contamination_factor = state.milk.system_cleanliness ** 0.7
            state.milk.system_cleanliness -= self.config.milk_contamination_rate * contamination_factor
            state.milk.system_cleanliness = max(state.milk.system_cleanliness, 0.1)
            
            state.milk.heating_element_efficiency -= self.config.heating_element_decay * 0.5
            state.milk.heating_element_efficiency = max(state.milk.heating_element_efficiency, 0.5)
        
        # Powder operations
        if operation_type == 'powder_product_dispensing':
            state.powder.system_cleanliness -= 0.001
            state.powder.system_cleanliness = max(state.powder.system_cleanliness, 0.2)
            
            state.powder.dispenser_wear += 0.0001
            state.powder.dispenser_wear = min(state.powder.dispenser_wear, 1.0)
        
        # Cleaning operations restore some state
        if operation_type == 'cleaning_cycle':
            # Restore milk system cleanliness
            state.milk.system_cleanliness = min(1.0, 
                state.milk.system_cleanliness + self.config.cleaning_milk_restoration)
            
            # Reduce boiler scale
            state.hydraulic.boiler_scale *= (1.0 - self.config.cleaning_scale_reduction)
            
            # Restore powder system
            state.powder.system_cleanliness = min(1.0, state.powder.system_cleanliness + 0.3)
            
            # Consume cleaning balls
            state.cleaning.cleaning_ball_supply -= self.config.cleaning_ball_consumption_rate
            state.cleaning.cleaning_ball_supply = max(state.cleaning.cleaning_ball_supply, 0.0)
            
            state.cleaning.cycles_since_cleaning = 0
        else:
            state.cleaning.cycles_since_cleaning += 1
    
    def check_maintenance_needed(self, state: MachineState) -> List[str]:
        maintenance_events = []
        
        # Service maintenance
        if (state.total_products >= self.config.service_product_interval or
            state.current_day - state.last_service_date >= self.config.service_time_interval_days):
            maintenance_events.append('service_maintenance')
        
        # Water filter change
        if (state.hydraulic.total_water_processed - state.hydraulic.last_filter_change >= 
            self.config.filter_change_water_amount):
            maintenance_events.append('filter_change')
        
        # Daily cleaning (probabilistic)
        if (state.cleaning.cycles_since_cleaning > 50 and 
            self.rng.random() < self.config.daily_cleaning_probability):
            maintenance_events.append('daily_cleaning')
        
        return maintenance_events
    
    def perform_maintenance(self, state: MachineState, maintenance_type: str, 
                          event_time: datetime) -> Dict[str, Any]:
        maintenance_record = {
            'timestamp': event_time,
            'maintenance_type': maintenance_type,
            'total_products': state.total_products,
            'grinder_wear_before': state.grinder.wear_level,
            'boiler_scale_before': state.hydraulic.boiler_scale,
            'milk_cleanliness_before': state.milk.system_cleanliness
        }
        
        if maintenance_type == 'service_maintenance':
            # Partial restoration of all systems
            state.grinder.wear_level *= (1.0 - self.config.service_restoration_factor)
            state.brewing.chamber_wear *= (1.0 - self.config.service_restoration_factor)
            state.brewing.extraction_efficiency = min(1.0, 
                state.brewing.extraction_efficiency + self.config.service_restoration_factor)
            state.hydraulic.boiler_scale *= (1.0 - self.config.service_restoration_factor)
            state.hydraulic.heating_element_efficiency = min(1.0,
                state.hydraulic.heating_element_efficiency + 0.2)
            state.milk.heating_element_efficiency = min(1.0,
                state.milk.heating_element_efficiency + 0.2)
            
            state.last_service_date = state.current_day
            
        elif maintenance_type == 'filter_change':
            # Reduce scale buildup rate effect
            state.hydraulic.boiler_scale *= (1.0 - self.config.filter_change_scale_reduction)
            state.hydraulic.last_filter_change = state.hydraulic.total_water_processed
            
        elif maintenance_type == 'daily_cleaning':
            # Same as cleaning cycle but more thorough
            state.milk.system_cleanliness = min(1.0, 
                state.milk.system_cleanliness + self.config.cleaning_milk_restoration * 1.2)
            state.hydraulic.boiler_scale *= (1.0 - self.config.cleaning_scale_reduction * 1.5)
            state.powder.system_cleanliness = min(1.0, state.powder.system_cleanliness + 0.5)
            state.cleaning.cycles_since_cleaning = 0
        
        maintenance_record.update({
            'grinder_wear_after': state.grinder.wear_level,
            'boiler_scale_after': state.hydraulic.boiler_scale,
            'milk_cleanliness_after': state.milk.system_cleanliness
        })
        
        return maintenance_record
    
    def simulate(self, profile_name: str, days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        profile = self.usage_profiles[profile_name]
        state = MachineState()
        
        event_times = self.generate_event_times(profile, days)
        events = []
        maintenance_log = []
        
        for i, event_time in enumerate(event_times):
            state.current_day = event_time.day
            
            # Check for maintenance needs
            maintenance_needed = self.check_maintenance_needed(state)
            for maintenance_type in maintenance_needed:
                maintenance_record = self.perform_maintenance(state, maintenance_type, event_time)
                maintenance_log.append(maintenance_record)
            
            # Generate operation event
            operation_type = self.select_operation_type(profile)
            
            if operation_type == 'coffee_brewing':
                event = self.generate_coffee_event(state, operation_type, event_time)
            elif operation_type == 'milk_product_dispensing':
                event = self.generate_milk_event(state, operation_type, event_time)
            elif operation_type == 'hot_water_dispensing':
                event = self.generate_hot_water_event(state, operation_type, event_time)
            elif operation_type == 'powder_product_dispensing':
                event = self.generate_powder_event(state, operation_type, event_time)
            else:  # cleaning_cycle
                event = self.generate_cleaning_event(state, event_time)
            
            events.append(event)
            
            # Update state after event
            self.update_state_after_event(state, event, operation_type)
        
        events_df = pd.DataFrame(events)
        maintenance_df = pd.DataFrame(maintenance_log)
        
        return events_df, maintenance_df

if __name__ == "__main__":
    config = SimulatorConfig()
    simulator = CoffeeMachineSimulator(config, seed=42)
    
    print("Commercial Coffee Machine Simulator")
    print("=" * 50)
    
    for profile_name in ['light_commercial', 'medium_commercial', 'heavy_commercial']:
        print(f"\nSimulating {profile_name} usage profile...")
        
        events_df, maintenance_df = simulator.simulate(profile_name, days=30)
        
        print(f"Generated {len(events_df)} events and {len(maintenance_df)} maintenance records")
        
        # Summary statistics
        coffee_events = events_df[events_df['product_type'].isin(config.coffee_products)]
        milk_events = events_df[events_df['product_type'].isin(config.milk_products)]
        
        if len(coffee_events) > 0:
            early_extraction = coffee_events.head(int(len(coffee_events) * 0.2))['extraction_time'].mean()
            late_extraction = coffee_events.tail(int(len(coffee_events) * 0.2))['extraction_time'].mean()
            print(f"Extraction time drift: {early_extraction:.1f}s → {late_extraction:.1f}s")
        
        if len(milk_events) > 0:
            early_foam = milk_events.head(int(len(milk_events) * 0.2))['foam_texture'].mean()
            late_foam = milk_events.tail(int(len(milk_events) * 0.2))['foam_texture'].mean()
            print(f"Foam texture drift: {early_foam:.1f} → {late_foam:.1f}")
        
        manual_stops = events_df['manual_stop_triggered'].sum()
        print(f"Manual stops: {manual_stops} ({manual_stops/len(events_df)*100:.1f}%)")
        
        # Save files
        events_filename = f"coffee_machine_events_{profile_name}.csv"
        maintenance_filename = f"coffee_machine_maintenance_{profile_name}.csv"
        
        events_df.to_csv(events_filename, index=False)
        maintenance_df.to_csv(maintenance_filename, index=False)
        
        print(f"Saved: {events_filename}, {maintenance_filename}")
    
    print("\nSimulation complete!")