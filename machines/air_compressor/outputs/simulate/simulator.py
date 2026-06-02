import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import math

class OperationType(Enum):
    PRESSURE_REGULATION = "pressure_regulation"
    MAINTENANCE_EVENT = "maintenance_event"
    SHUTDOWN_EVENT = "shutdown_event"

class LoadState(Enum):
    LOADED = "loaded"
    UNLOADED = "unloaded"

class ValvePosition(Enum):
    FULLY_OPEN = "fully_open"
    FULLY_CLOSED = "fully_closed"
    OPEN = "open"
    CLOSED = "closed"

@dataclass
class SimulatorConfig:
    # Operating parameters
    base_loading_pressure: float = 6.5
    base_unloading_pressure: float = 7.5
    pressure_regulation_range: float = 1.0
    minimum_stop_time: int = 5
    
    # Temperature parameters
    ambient_temperature: float = 20.0
    base_element_outlet_temp: float = 80.0
    temp_rise_above_ambient: float = 60.0
    
    # Flow parameters
    rated_air_flow: float = 450.0
    base_power_loaded: float = 90.0
    base_power_unloaded: float = 25.0
    
    # Oil system parameters
    base_oil_separator_dp: float = 0.15
    oil_separator_dp_limit: float = 1.0
    
    # Degradation rates (per 1000 operating hours)
    oil_separator_fouling_rate: float = 0.12
    air_filter_restriction_rate: float = 0.08
    cooler_fouling_rate: float = 0.05
    oil_degradation_rate: float = 0.15
    pressure_regulation_drift_rate: float = 0.02
    screw_element_wear_rate: float = 0.03
    sensor_drift_rate: float = 0.01
    
    # Maintenance thresholds
    air_filter_replacement_hours: int = 2000
    oil_separator_replacement_hours: int = 8000
    oil_change_hours: int = 4000
    cooler_cleaning_hours: int = 6000
    
    # Noise parameters
    pressure_noise_std: float = 0.05
    temperature_noise_std: float = 2.0
    flow_noise_std: float = 5.0
    power_noise_std: float = 1.5
    dp_noise_std: float = 0.02
    
    # Failure thresholds
    overtemp_shutdown_threshold: float = 120.0
    overtemp_warning_threshold: float = 110.0
    
    # Load cycle parameters
    min_loading_duration: int = 10
    max_loading_duration: int = 3600
    min_unloading_duration: int = 10
    max_unloading_duration: int = 1200

@dataclass
class PressureControlState:
    pressure_regulation_drift: float = 0.0
    valve_wear: float = 0.0
    last_maintenance_hours: float = 0.0

@dataclass
class AirCompressionState:
    screw_element_wear: float = 0.0
    inlet_valve_degradation: float = 0.0
    last_maintenance_hours: float = 0.0

@dataclass
class OilCirculationState:
    oil_separator_fouling: float = 0.0
    oil_degradation: float = 0.0
    oil_filter_restriction: float = 0.0
    last_oil_change_hours: float = 0.0
    last_separator_replacement_hours: float = 0.0

@dataclass
class CoolingState:
    cooler_fouling: float = 0.0
    fan_degradation: float = 0.0
    last_cleaning_hours: float = 0.0

@dataclass
class FiltrationState:
    air_filter_restriction: float = 0.0
    last_replacement_hours: float = 0.0

@dataclass
class AirTreatmentState:
    dryer_performance_degradation: float = 0.0
    last_maintenance_hours: float = 0.0

@dataclass
class ThermalProtectionState:
    sensor_drift: float = 0.0
    last_calibration_hours: float = 0.0

@dataclass
class CompressorState:
    total_operating_hours: float = 0.0
    loaded_hours: float = 0.0
    cycles_count: int = 0
    current_load_state: LoadState = LoadState.UNLOADED
    
    pressure_control: PressureControlState = field(default_factory=PressureControlState)
    air_compression: AirCompressionState = field(default_factory=AirCompressionState)
    oil_circulation: OilCirculationState = field(default_factory=OilCirculationState)
    cooling: CoolingState = field(default_factory=CoolingState)
    filtration: FiltrationState = field(default_factory=FiltrationState)
    air_treatment: AirTreatmentState = field(default_factory=AirTreatmentState)
    thermal_protection: ThermalProtectionState = field(default_factory=ThermalProtectionState)

@dataclass
class UsageProfile:
    name: str
    description: str
    operating_hours_per_year: int
    relative_intensity: str
    operation_mix: Dict[str, float]
    maintenance_frequency_multiplier: float = 1.0

@dataclass
class MaintenanceEvent:
    timestamp: float
    event_type: str
    description: str
    operating_hours: float
    triggered_by: str

class CompressorSimulator:
    def __init__(self, config: SimulatorConfig, usage_profile: UsageProfile, seed: int = 42):
        self.config = config
        self.usage_profile = usage_profile
        self.rng = np.random.default_rng(seed)
        self.state = CompressorState()
        self.maintenance_log: List[MaintenanceEvent] = []
        
    def update_degradation(self, hours_delta: float, operation_type: OperationType):
        intensity_factor = {
            "high": 1.2,
            "medium": 1.0,
            "low": 0.7
        }[self.usage_profile.relative_intensity]
        
        if operation_type == OperationType.PRESSURE_REGULATION:
            load_factor = 1.0 if self.state.current_load_state == LoadState.LOADED else 0.3
            
            # Oil separator fouling (accelerates with oil degradation)
            fouling_rate = self.config.oil_separator_fouling_rate * intensity_factor * load_factor
            fouling_rate *= (1.0 + 0.5 * self.state.oil_circulation.oil_degradation)
            self.state.oil_circulation.oil_separator_fouling += fouling_rate * hours_delta / 1000
            
            # Air filter restriction (nonlinear with accumulated dust)
            restriction_rate = self.config.air_filter_restriction_rate * intensity_factor
            current_restriction = self.state.filtration.air_filter_restriction
            restriction_rate *= (1.0 + 2.0 * current_restriction**2)
            self.state.filtration.air_filter_restriction += restriction_rate * hours_delta / 1000
            
            # Cooler fouling
            fouling_rate = self.config.cooler_fouling_rate * intensity_factor
            self.state.cooling.cooler_fouling += fouling_rate * hours_delta / 1000
            
            # Oil degradation (accelerates with temperature and contamination)
            temp_factor = 1.0 + 0.02 * max(0, self.get_element_outlet_temperature() - 90)
            oil_rate = self.config.oil_degradation_rate * intensity_factor * temp_factor * load_factor
            self.state.oil_circulation.oil_degradation += oil_rate * hours_delta / 1000
            
            # Pressure regulation drift
            drift_rate = self.config.pressure_regulation_drift_rate * intensity_factor
            self.state.pressure_control.pressure_regulation_drift += drift_rate * hours_delta / 1000
            
            # Screw element wear (accelerates with poor lubrication)
            wear_factor = 1.0 + 0.3 * self.state.oil_circulation.oil_degradation
            wear_rate = self.config.screw_element_wear_rate * intensity_factor * wear_factor * load_factor
            self.state.air_compression.screw_element_wear += wear_rate * hours_delta / 1000
            
            # Sensor drift
            drift_rate = self.config.sensor_drift_rate * intensity_factor
            self.state.thermal_protection.sensor_drift += drift_rate * hours_delta / 1000
        
        # Update operating hours
        self.state.total_operating_hours += hours_delta
        if operation_type == OperationType.PRESSURE_REGULATION and self.state.current_load_state == LoadState.LOADED:
            self.state.loaded_hours += hours_delta
    
    def check_maintenance_needed(self) -> Optional[MaintenanceEvent]:
        hours = self.state.total_operating_hours
        
        # Air filter replacement
        if (hours - self.state.filtration.last_replacement_hours) >= self.config.air_filter_replacement_hours:
            return MaintenanceEvent(
                timestamp=hours,
                event_type="air_filter_replacement",
                description="Scheduled air filter replacement",
                operating_hours=hours,
                triggered_by="scheduled_hours"
            )
        
        # Oil separator replacement
        if (self.state.oil_circulation.oil_separator_fouling > 0.8 or 
            (hours - self.state.oil_circulation.last_separator_replacement_hours) >= self.config.oil_separator_replacement_hours):
            return MaintenanceEvent(
                timestamp=hours,
                event_type="oil_separator_replacement",
                description="Oil separator element replacement",
                operating_hours=hours,
                triggered_by="fouling_threshold" if self.state.oil_circulation.oil_separator_fouling > 0.8 else "scheduled_hours"
            )
        
        # Oil change
        if (hours - self.state.oil_circulation.last_oil_change_hours) >= self.config.oil_change_hours:
            return MaintenanceEvent(
                timestamp=hours,
                event_type="oil_change",
                description="Scheduled oil change",
                operating_hours=hours,
                triggered_by="scheduled_hours"
            )
        
        # Cooler cleaning
        if (self.state.cooling.cooler_fouling > 0.6 or 
            (hours - self.state.cooling.last_cleaning_hours) >= self.config.cooler_cleaning_hours):
            return MaintenanceEvent(
                timestamp=hours,
                event_type="cooler_cleaning",
                description="Cooler cleaning maintenance",
                operating_hours=hours,
                triggered_by="fouling_threshold" if self.state.cooling.cooler_fouling > 0.6 else "scheduled_hours"
            )
        
        return None
    
    def perform_maintenance(self, maintenance_event: MaintenanceEvent):
        hours = self.state.total_operating_hours
        
        if maintenance_event.event_type == "air_filter_replacement":
            self.state.filtration.air_filter_restriction = 0.0
            self.state.filtration.last_replacement_hours = hours
        
        elif maintenance_event.event_type == "oil_separator_replacement":
            self.state.oil_circulation.oil_separator_fouling *= 0.1
            self.state.oil_circulation.last_separator_replacement_hours = hours
        
        elif maintenance_event.event_type == "oil_change":
            self.state.oil_circulation.oil_degradation = 0.0
            self.state.oil_circulation.last_oil_change_hours = hours
        
        elif maintenance_event.event_type == "cooler_cleaning":
            self.state.cooling.cooler_fouling *= 0.2
            self.state.cooling.last_cleaning_hours = hours
        
        self.maintenance_log.append(maintenance_event)
    
    def get_loading_pressure(self) -> float:
        base = self.config.base_loading_pressure
        drift = self.state.pressure_control.pressure_regulation_drift * 0.1
        noise = self.rng.normal(0, self.config.pressure_noise_std * 0.5)
        return max(5.5, base + drift + noise)
    
    def get_unloading_pressure(self) -> float:
        loading = self.get_loading_pressure()
        range_val = self.config.pressure_regulation_range
        drift = self.state.pressure_control.pressure_regulation_drift * 0.1
        noise = self.rng.normal(0, self.config.pressure_noise_std * 0.5)
        return loading + range_val + drift + noise
    
    def get_system_pressure(self, target_state: LoadState) -> float:
        loading = self.get_loading_pressure()
        unloading = self.get_unloading_pressure()
        
        if target_state == LoadState.LOADED:
            base = loading + self.rng.uniform(0, 0.2)
        else:
            base = unloading - self.rng.uniform(0, 0.2)
        
        # Add system effects
        restriction_effect = self.state.filtration.air_filter_restriction * 0.1
        fouling_effect = self.state.oil_circulation.oil_separator_fouling * 0.05
        
        noise_std = self.config.pressure_noise_std * (1.0 + 0.5 * self.state.pressure_control.pressure_regulation_drift)
        noise = self.rng.normal(0, noise_std)
        
        return base - restriction_effect - fouling_effect + noise
    
    def get_oil_separator_pressure_difference(self) -> float:
        base = self.config.base_oil_separator_dp
        fouling_effect = self.state.oil_circulation.oil_separator_fouling * 0.8
        oil_effect = self.state.oil_circulation.oil_degradation * 0.2
        
        # Nonlinear fouling effect
        total_fouling = fouling_effect + oil_effect
        pressure_rise = total_fouling * (1.0 + total_fouling**2)
        
        noise_std = self.config.dp_noise_std * (1.0 + total_fouling)
        noise = self.rng.normal(0, noise_std)
        
        return max(0.0, min(1.2, base + pressure_rise + noise))
    
    def get_element_outlet_temperature(self) -> float:
        base = self.config.ambient_temperature + self.config.temp_rise_above_ambient
        
        # Degradation effects
        cooler_effect = self.state.cooling.cooler_fouling * 15.0
        oil_effect = self.state.oil_circulation.oil_degradation * 10.0
        filter_effect = self.state.filtration.air_filter_restriction * 8.0
        wear_effect = self.state.air_compression.screw_element_wear * 12.0
        
        # Nonlinear temperature rise
        total_degradation = cooler_effect + oil_effect + filter_effect + wear_effect
        temp_rise = total_degradation * (1.0 + 0.1 * total_degradation)
        
        # Sensor drift
        sensor_error = self.state.thermal_protection.sensor_drift * 3.0
        
        noise_std = self.config.temperature_noise_std * (1.0 + 0.3 * total_degradation / 20.0)
        noise = self.rng.normal(0, noise_std)
        
        return base + temp_rise + sensor_error + noise
    
    def get_air_flow_output(self, load_state: LoadState) -> float:
        if load_state == LoadState.UNLOADED:
            return 0.0
        
        base = self.config.rated_air_flow
        
        # Degradation effects
        filter_loss = self.state.filtration.air_filter_restriction * 0.15
        wear_loss = self.state.air_compression.screw_element_wear * 0.12
        
        # Nonlinear efficiency loss
        total_loss = filter_loss + wear_loss
        efficiency = 1.0 - total_loss * (1.0 + 0.5 * total_loss)
        
        noise_std = self.config.flow_noise_std * (1.0 + total_loss)
        noise = self.rng.normal(0, noise_std)
        
        return max(0.0, base * efficiency + noise)
    
    def get_power_consumption(self, load_state: LoadState) -> Tuple[float, float]:
        if load_state == LoadState.LOADED:
            base = self.config.base_power_loaded
            
            # Degradation increases power consumption
            filter_effect = self.state.filtration.air_filter_restriction * 3.0
            fouling_effect = self.state.oil_circulation.oil_separator_fouling * 2.5
            wear_effect = self.state.air_compression.screw_element_wear * 4.0
            
            power_increase = filter_effect + fouling_effect + wear_effect
            loaded_power = base + power_increase * (1.0 + 0.2 * power_increase / 10.0)
            
            noise = self.rng.normal(0, self.config.power_noise_std)
            return max(50.0, loaded_power + noise), None
        
        else:
            base = self.config.base_power_unloaded
            degradation_effect = (self.state.air_compression.screw_element_wear + 
                                self.state.oil_circulation.oil_degradation) * 1.5
            
            unloaded_power = base + degradation_effect
            noise = self.rng.normal(0, self.config.power_noise_std * 0.5)
            return None, max(10.0, unloaded_power + noise)
    
    def get_loading_duration(self) -> int:
        base_duration = self.rng.exponential(300)
        
        # Adjust based on system efficiency
        efficiency_factor = 1.0 + self.state.filtration.air_filter_restriction * 0.3
        efficiency_factor += self.state.air_compression.screw_element_wear * 0.4
        
        duration = base_duration * efficiency_factor
        
        # Add some irregularity
        if self.rng.random() < 0.1:
            duration *= self.rng.uniform(0.5, 2.0)
        
        return max(self.config.min_loading_duration, 
                  min(self.config.max_loading_duration, int(duration)))
    
    def get_unloading_duration(self) -> int:
        base_duration = self.rng.exponential(180)
        
        # Shorter unloading if system is degraded (more frequent cycles)
        degradation_factor = 1.0 - 0.2 * (self.state.filtration.air_filter_restriction + 
                                         self.state.air_compression.screw_element_wear)
        
        duration = base_duration * max(0.3, degradation_factor)
        
        # Add irregularity
        if self.rng.random() < 0.15:
            duration *= self.rng.uniform(0.3, 1.8)
        
        return max(self.config.min_unloading_duration,
                  min(self.config.max_unloading_duration, int(duration)))
    
    def check_failure_conditions(self) -> Optional[str]:
        temp = self.get_element_outlet_temperature()
        if temp > self.config.overtemp_shutdown_threshold:
            return "overtemperature_shutdown"
        
        oil_dp = self.get_oil_separator_pressure_difference()
        if oil_dp > self.config.oil_separator_dp_limit:
            return "oil_separator_failure"
        
        # Pressure regulation failure
        if self.state.pressure_control.pressure_regulation_drift > 0.5:
            return "pressure_regulation_failure"
        
        return None
    
    def generate_pressure_regulation_event(self) -> Dict[str, Any]:
        # Determine next load state
        if self.state.current_load_state == LoadState.UNLOADED:
            next_state = LoadState.LOADED
            duration = self.get_loading_duration()
        else:
            next_state = LoadState.UNLOADED
            duration = self.get_unloading_duration()
        
        # Update state for this cycle
        self.state.current_load_state = next_state
        self.state.cycles_count += 1
        
        # Generate measurements
        loading_pressure = self.get_loading_pressure()
        unloading_pressure = self.get_unloading_pressure()
        system_pressure = self.get_system_pressure(next_state)
        
        air_receiver_pressure = system_pressure + self.rng.normal(0, 0.02)
        control_air_pressure = air_receiver_pressure + self.rng.normal(0, 0.01)
        
        element_temp = self.get_element_outlet_temperature()
        oil_dp = self.get_oil_separator_pressure_difference()
        air_flow = self.get_air_flow_output(next_state)
        
        power_loaded, power_unloaded = self.get_power_consumption(next_state)
        
        # Valve positions
        if next_state == LoadState.LOADED:
            inlet_valve_pos = ValvePosition.FULLY_OPEN
            unloading_valve_pos = ValvePosition.CLOSED
            solenoid_y1_state = True
        else:
            inlet_valve_pos = ValvePosition.FULLY_CLOSED
            unloading_valve_pos = ValvePosition.OPEN
            solenoid_y1_state = False
        
        # Update degradation
        hours_delta = duration / 3600.0
        self.update_degradation(hours_delta, OperationType.PRESSURE_REGULATION)
        
        return {
            "loading_pressure": loading_pressure,
            "unloading_pressure": unloading_pressure,
            "pressure_regulation_range": unloading_pressure - loading_pressure,
            "minimum_stop_time": self.config.minimum_stop_time,
            "system_pressure": system_pressure,
            "air_receiver_pressure": air_receiver_pressure,
            "control_air_pressure": control_air_pressure,
            "compressor_element_outlet_temperature": element_temp,
            "oil_separator_pressure_difference": oil_dp,
            "air_flow_output": air_flow,
            "power_consumption_loaded": power_loaded,
            "power_consumption_unloaded": power_unloaded,
            "loading_duration": duration if next_state == LoadState.LOADED else None,
            "unloading_duration": duration if next_state == LoadState.UNLOADED else None,
            "compressor_load_state": next_state.value,
            "inlet_valve_position": inlet_valve_pos.value,
            "unloading_valve_position": unloading_valve_pos.value,
            "solenoid_valve_y1_state": solenoid_y1_state,
            "automatic_operation_active": True
        }
    
    def generate_maintenance_event(self, maintenance: MaintenanceEvent) -> Dict[str, Any]:
        # Perform the maintenance
        self.perform_maintenance(maintenance)
        
        # Generate event record
        loading_pressure = self.get_loading_pressure()
        unloading_pressure = self.get_unloading_pressure()
        
        return {
            "loading_pressure": loading_pressure,
            "unloading_pressure": unloading_pressure,
            "pressure_regulation_range": unloading_pressure - loading_pressure,
            "minimum_stop_time": self.config.minimum_stop_time,
            "system_pressure": loading_pressure + self.rng.normal(0, 0.1),
            "air_receiver_pressure": loading_pressure + self.rng.normal(0, 0.1),
            "control_air_pressure": loading_pressure + self.rng.normal(0, 0.1),
            "compressor_element_outlet_temperature": self.config.ambient_temperature + self.rng.normal(0, 2.0),
            "oil_separator_pressure_difference": self.get_oil_separator_pressure_difference(),
            "air_flow_output": None,
            "power_consumption_loaded": None,
            "power_consumption_unloaded": None,
            "loading_duration": None,
            "unloading_duration": None,
            "compressor_load_state": "unloaded",
            "inlet_valve_position": "fully_closed",
            "unloading_valve_position": "open",
            "solenoid_valve_y1_state": False,
            "automatic_operation_active": False
        }
    
    def generate_shutdown_event(self, failure_type: str) -> Dict[str, Any]:
        loading_pressure = self.get_loading_pressure()
        unloading_pressure = self.get_unloading_pressure()
        
        return {
            "loading_pressure": loading_pressure,
            "unloading_pressure": unloading_pressure,
            "pressure_regulation_range": unloading_pressure - loading_pressure,
            "minimum_stop_time": self.config.minimum_stop_time,
            "system_pressure": self.get_system_pressure(LoadState.UNLOADED),
            "air_receiver_pressure": loading_pressure + self.rng.normal(0, 0.1),
            "control_air_pressure": 0.0,
            "compressor_element_outlet_temperature": self.get_element_outlet_temperature(),
            "oil_separator_pressure_difference": self.get_oil_separator_pressure_difference(),
            "air_flow_output": None,
            "power_consumption_loaded": None,
            "power_consumption_unloaded": None,
            "loading_duration": None,
            "unloading_duration": None,
            "compressor_load_state": "unloaded",
            "inlet_valve_position": "fully_closed",
            "unloading_valve_position": "open",
            "solenoid_valve_y1_state": False,
            "automatic_operation_active": False
        }
    
    def simulate(self, duration_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = []
        target_hours = duration_days * 24
        
        # Calculate event rate based on usage profile
        daily_hours = self.usage_profile.operating_hours_per_year / 365
        events_per_hour = 12 if self.usage_profile.relative_intensity == "high" else \
                         8 if self.usage_profile.relative_intensity == "medium" else 4
        
        # Generate events using Poisson process
        current_time = 0.0
        
        while current_time < target_hours:
            # Time to next event
            inter_arrival = self.rng.exponential(1.0 / events_per_hour)
            current_time += inter_arrival
            
            if current_time >= target_hours:
                break
            
            # Check for maintenance first
            maintenance_needed = self.check_maintenance_needed()
            if maintenance_needed:
                event_data = self.generate_maintenance_event(maintenance_needed)
                event_data["timestamp"] = current_time
                event_data["operation_type"] = OperationType.MAINTENANCE_EVENT.value
                events.append(event_data)
                continue
            
            # Check for failures
            failure = self.check_failure_conditions()
            if failure and self.rng.random() < 0.02:  # 2% chance when conditions met
                event_data = self.generate_shutdown_event(failure)
                event_data["timestamp"] = current_time
                event_data["operation_type"] = OperationType.SHUTDOWN_EVENT.value
                events.append(event_data)
                
                # Recovery time
                recovery_hours = self.rng.exponential(2.0)
                current_time += recovery_hours
                continue
            
            # Normal pressure regulation event
            event_data = self.generate_pressure_regulation_event()
            event_data["timestamp"] = current_time
            event_data["operation_type"] = OperationType.PRESSURE_REGULATION.value
            events.append(event_data)
        
        events_df = pd.DataFrame(events)
        maintenance_df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "event_type": m.event_type,
                "description": m.description,
                "operating_hours": m.operating_hours,
                "triggered_by": m.triggered_by
            }
            for m in self.maintenance_log
        ])
        
        return events_df, maintenance_df

def create_usage_profiles() -> List[UsageProfile]:
    return [
        UsageProfile(
            name="continuous_industrial",
            description="Heavy industrial use with 6000-8000 operating hours per year",
            operating_hours_per_year=7000,
            relative_intensity="high",
            operation_mix={"pressure_regulation": 0.85, "maintenance_event": 0.1, "shutdown_event": 0.05}
        ),
        UsageProfile(
            name="moderate_commercial",
            description="Commercial/workshop use with 3000-5000 operating hours per year",
            operating_hours_per_year=4000,
            relative_intensity="medium",
            operation_mix={"pressure_regulation": 0.8, "maintenance_event": 0.15, "shutdown_event": 0.05}
        ),
        UsageProfile(
            name="light_intermittent",
            description="Light duty or backup service with 1000-2000 operating hours per year",
            operating_hours_per_year=1500,
            relative_intensity="low",
            operation_mix={"pressure_regulation": 0.7, "maintenance_event": 0.25, "shutdown_event": 0.05}
        )
    ]

if __name__ == "__main__":
    config = SimulatorConfig()
    usage_profiles = create_usage_profiles()
    
    print("Atlas Copco GA90C Compressor Simulator")
    print("=" * 50)
    
    for profile in usage_profiles:
        print(f"\nSimulating {profile.name} usage profile...")
        
        simulator = CompressorSimulator(config, profile, seed=42)
        events_df, maintenance_df = simulator.simulate(duration_days=30)
        
        print(f"Generated {len(events_df)} events")
        print(f"Total operating hours: {simulator.state.total_operating_hours:.1f}")
        print(f"Loaded hours: {simulator.state.loaded_hours:.1f}")
        print(f"Load cycles: {simulator.state.cycles_count}")
        print(f"Maintenance events: {len(maintenance_df)}")
        
        # Show degradation summary
        print(f"Oil separator fouling: {simulator.state.oil_circulation.oil_separator_fouling:.3f}")
        print(f"Air filter restriction: {simulator.state.filtration.air_filter_restriction:.3f}")
        print(f"Screw element wear: {simulator.state.air_compression.screw_element_wear:.3f}")
        
        # Save files
        events_filename = f"compressor_events_{profile.name}.csv"
        maintenance_filename = f"compressor_maintenance_{profile.name}.csv"
        
        events_df.to_csv(events_filename, index=False)
        maintenance_df.to_csv(maintenance_filename, index=False)
        
        print(f"Saved {events_filename} and {maintenance_filename}")
    
    print("\nSimulation complete!")