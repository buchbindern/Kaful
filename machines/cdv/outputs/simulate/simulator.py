import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Config:
    # Temperature control parameters
    base_zone_temp: float = 425.0
    temp_control_precision: float = 2.0
    temp_ramp_rate: float = 10.0
    safe_temp_range: Tuple[float, float] = (116.0, 120.0)
    
    # Gas flow parameters
    n2_flow_range: Tuple[float, float] = (0.0, 20.0)
    o2_flow_range: Tuple[float, float] = (0.0, 200.0)
    sih4_flow_range: Tuple[float, float] = (0.0, 200.0)
    thermal_flow_range: Tuple[float, float] = (0.0, 20000.0)
    flow_control_precision: float = 0.5
    
    # Pressure parameters
    process_pressure_range: Tuple[float, float] = (0.0, 760.0)
    target_process_pressure: float = 250.0
    pressure_control_precision: float = 5.0
    
    # Valve parameters
    valve_drive_precision: float = 1.0
    
    # Boat positioning
    boat_out_position: float = 10.0
    boat_in_position: float = 2000.0
    boat_speed: float = 300.0
    boat_position_precision: float = 2.0
    
    # Process timing
    deposition_time_base: float = 30.0
    evacuation_time_base: float = 5.0
    purge_time_base: float = 5.0
    
    # Degradation rates (per operating hour)
    heating_element_degradation_rate: float = 0.001
    thermocouple_drift_rate: float = 0.0005
    thermal_uniformity_loss_rate: float = 0.0003
    mfc_calibration_drift_rate: float = 0.0008
    valve_wear_rate: float = 0.0004
    pump_throughput_loss_rate: float = 0.0006
    leak_accumulation_rate: float = 0.0002
    mechanical_wear_rate: float = 0.0001
    sensor_drift_rate: float = 0.0003
    alarm_threshold_drift_rate: float = 0.0001
    quartz_devitrification_rate: float = 0.00005
    
    # Buildup rates (per deposition hour)
    gas_line_contamination_rate: float = 0.002
    tube_contamination_rate: float = 0.005
    
    # Failure thresholds
    heating_element_failure_threshold: float = 0.8
    pump_failure_threshold: float = 0.7
    mfc_failure_threshold: float = 0.6
    tube_contamination_threshold: float = 0.9
    
    # Maintenance restoration factors
    tube_cleaning_restoration: float = 0.95
    mfc_calibration_restoration: float = 0.9
    heating_element_replacement_restoration: float = 1.0
    pump_maintenance_restoration: float = 0.7
    
    # Maintenance triggers
    scheduled_maintenance_interval_hours: float = 720.0  # 30 days
    contamination_maintenance_threshold: float = 0.6
    calibration_maintenance_threshold: float = 0.4
    
    # Noise parameters
    temp_noise_base: float = 0.5
    pressure_noise_base: float = 2.0
    flow_noise_base: float = 0.2
    position_noise_base: float = 1.0
    
    # Recipe parameters
    recipe_names: List[str] = field(default_factory=lambda: [
        "LTO_SiO2_425C", "PolySi_610C", "SiN_750C", "TEOS_650C", "Cleaning_400C"
    ])
    
    # Alarm thresholds
    high_flow_alarm_threshold: float = 1.1
    low_flow_alarm_threshold: float = 0.9
    temp_alarm_threshold: float = 10.0

@dataclass
class ThermalControlState:
    heating_element_degradation: float = 0.0
    thermocouple_drift: float = 0.0
    thermal_uniformity_loss: float = 0.0
    operating_hours: float = 0.0
    last_maintenance_hours: float = 0.0
    zone_temp_offsets: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class GasDeliveryState:
    mfc_calibration_drift: float = 0.0
    valve_wear: float = 0.0
    gas_line_contamination: float = 0.0
    operating_hours: float = 0.0
    deposition_hours: float = 0.0
    last_calibration_hours: float = 0.0
    mfc_offsets: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

@dataclass
class VacuumSystemState:
    pump_throughput_loss: float = 0.0
    leak_accumulation: float = 0.0
    operating_hours: float = 0.0
    last_maintenance_hours: float = 0.0
    base_pressure_drift: float = 0.0

@dataclass
class BoatPositioningState:
    mechanical_wear: float = 0.0
    operating_hours: float = 0.0
    position_drift: float = 0.0

@dataclass
class ProcessMonitoringState:
    sensor_drift: float = 0.0
    alarm_threshold_drift: float = 0.0
    operating_hours: float = 0.0

@dataclass
class ProcessTubeState:
    tube_contamination: float = 0.0
    quartz_devitrification: float = 0.0
    deposition_hours: float = 0.0
    last_cleaning_hours: float = 0.0
    contamination_buildup: float = 0.0

@dataclass
class SystemState:
    thermal_control: ThermalControlState = field(default_factory=ThermalControlState)
    gas_delivery: GasDeliveryState = field(default_factory=GasDeliveryState)
    vacuum_system: VacuumSystemState = field(default_factory=VacuumSystemState)
    boat_positioning: BoatPositioningState = field(default_factory=BoatPositioningState)
    process_monitoring: ProcessMonitoringState = field(default_factory=ProcessMonitoringState)
    process_tube: ProcessTubeState = field(default_factory=ProcessTubeState)
    total_operating_hours: float = 0.0
    last_maintenance_check: float = 0.0

class UsageProfile:
    def __init__(self, name: str, description: str, operating_hours_per_day: float,
                 relative_intensity: str, operation_mix: Dict[str, float]):
        self.name = name
        self.description = description
        self.operating_hours_per_day = operating_hours_per_day
        self.relative_intensity = relative_intensity
        self.operation_mix = operation_mix
        
        intensity_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}
        self.intensity_multiplier = intensity_multipliers[relative_intensity]

def update_degradation_state(state: SystemState, config: Config, operation_type: str, 
                           duration_hours: float, rng: np.random.Generator) -> None:
    # Update operating hours for all subsystems
    state.thermal_control.operating_hours += duration_hours
    state.gas_delivery.operating_hours += duration_hours
    state.vacuum_system.operating_hours += duration_hours
    state.boat_positioning.operating_hours += duration_hours
    state.process_monitoring.operating_hours += duration_hours
    state.total_operating_hours += duration_hours
    
    # Thermal control degradation
    if operation_type in ["deposition", "maintenance"]:
        thermal_stress = duration_hours * (1.0 + 0.3 * rng.random())
        state.thermal_control.heating_element_degradation += (
            config.heating_element_degradation_rate * thermal_stress * 
            (1.0 + state.thermal_control.heating_element_degradation * 0.5)
        )
        state.thermal_control.thermocouple_drift += (
            config.thermocouple_drift_rate * thermal_stress *
            (1.0 + 0.2 * rng.random())
        )
        state.thermal_control.thermal_uniformity_loss += (
            config.thermal_uniformity_loss_rate * thermal_stress
        )
    
    # Gas delivery degradation
    if operation_type in ["deposition", "purge"]:
        gas_stress = duration_hours * (1.0 + 0.2 * rng.random())
        state.gas_delivery.mfc_calibration_drift += (
            config.mfc_calibration_drift_rate * gas_stress *
            (1.0 + state.gas_delivery.mfc_calibration_drift * 0.3)
        )
        state.gas_delivery.valve_wear += (
            config.valve_wear_rate * gas_stress *
            (1.0 + 0.1 * rng.random())
        )
    
    if operation_type == "deposition":
        deposition_stress = duration_hours * (1.0 + 0.4 * rng.random())
        state.gas_delivery.deposition_hours += duration_hours
        state.process_tube.deposition_hours += duration_hours
        
        state.gas_delivery.gas_line_contamination += (
            config.gas_line_contamination_rate * deposition_stress
        )
        state.process_tube.tube_contamination += (
            config.tube_contamination_rate * deposition_stress *
            (1.0 + state.process_tube.tube_contamination * 0.2)
        )
        state.process_tube.quartz_devitrification += (
            config.quartz_devitrification_rate * deposition_stress
        )
    
    # Vacuum system degradation
    if operation_type in ["evacuation", "deposition"]:
        vacuum_stress = duration_hours * (1.0 + 0.3 * rng.random())
        state.vacuum_system.pump_throughput_loss += (
            config.pump_throughput_loss_rate * vacuum_stress *
            (1.0 + state.vacuum_system.pump_throughput_loss * 0.4)
        )
        state.vacuum_system.leak_accumulation += (
            config.leak_accumulation_rate * vacuum_stress
        )
    
    # Boat positioning degradation
    if operation_type == "boat_movement":
        mechanical_stress = duration_hours * (1.0 + 0.5 * rng.random())
        state.boat_positioning.mechanical_wear += (
            config.mechanical_wear_rate * mechanical_stress
        )
    
    # Process monitoring degradation (all operations)
    monitoring_stress = duration_hours * (1.0 + 0.1 * rng.random())
    state.process_monitoring.sensor_drift += (
        config.sensor_drift_rate * monitoring_stress
    )
    state.process_monitoring.alarm_threshold_drift += (
        config.alarm_threshold_drift_rate * monitoring_stress
    )

def check_maintenance_triggers(state: SystemState, config: Config) -> List[str]:
    maintenance_events = []
    
    # Scheduled maintenance
    hours_since_last = state.total_operating_hours - state.last_maintenance_check
    if hours_since_last >= config.scheduled_maintenance_interval_hours:
        maintenance_events.append("scheduled_maintenance")
    
    # Contamination-based maintenance
    if state.process_tube.tube_contamination >= config.contamination_maintenance_threshold:
        maintenance_events.append("tube_cleaning")
    
    if state.gas_delivery.gas_line_contamination >= config.contamination_maintenance_threshold:
        maintenance_events.append("gas_line_cleaning")
    
    # Calibration-based maintenance
    if state.gas_delivery.mfc_calibration_drift >= config.calibration_maintenance_threshold:
        maintenance_events.append("mfc_calibration")
    
    # Failure-based maintenance
    if state.thermal_control.heating_element_degradation >= config.heating_element_failure_threshold:
        maintenance_events.append("heating_element_replacement")
    
    if state.vacuum_system.pump_throughput_loss >= config.pump_failure_threshold:
        maintenance_events.append("pump_maintenance")
    
    return maintenance_events

def perform_maintenance(state: SystemState, config: Config, maintenance_type: str) -> None:
    if maintenance_type == "tube_cleaning":
        state.process_tube.tube_contamination *= (1.0 - config.tube_cleaning_restoration)
        state.process_tube.last_cleaning_hours = state.total_operating_hours
    
    elif maintenance_type == "gas_line_cleaning":
        state.gas_delivery.gas_line_contamination *= (1.0 - config.tube_cleaning_restoration)
    
    elif maintenance_type == "mfc_calibration":
        state.gas_delivery.mfc_calibration_drift *= (1.0 - config.mfc_calibration_restoration)
        state.gas_delivery.last_calibration_hours = state.total_operating_hours
    
    elif maintenance_type == "heating_element_replacement":
        state.thermal_control.heating_element_degradation *= (1.0 - config.heating_element_replacement_restoration)
        state.thermal_control.last_maintenance_hours = state.total_operating_hours
    
    elif maintenance_type == "pump_maintenance":
        state.vacuum_system.pump_throughput_loss *= (1.0 - config.pump_maintenance_restoration)
        state.vacuum_system.last_maintenance_hours = state.total_operating_hours
    
    elif maintenance_type == "scheduled_maintenance":
        state.process_tube.tube_contamination *= 0.8
        state.gas_delivery.mfc_calibration_drift *= 0.9
        state.thermal_control.thermocouple_drift *= 0.9
        state.last_maintenance_check = state.total_operating_hours

def generate_deposition_event(state: SystemState, config: Config, rng: np.random.Generator) -> Dict:
    # Recipe selection
    recipe_name = rng.choice(config.recipe_names)
    lot_id = f"LOT{rng.integers(1000, 9999)}"
    tube_name = f"TUBE{rng.integers(1, 5)}"
    
    # Process step
    process_step_number = rng.integers(10, 20)
    
    # Temperature generation with degradation effects
    temp_degradation_effect = (
        state.thermal_control.heating_element_degradation * 5.0 +
        state.thermal_control.thermocouple_drift * 3.0 +
        state.process_tube.tube_contamination * 2.0
    )
    
    temp_noise_multiplier = 1.0 + state.thermal_control.thermal_uniformity_loss * 2.0
    
    zone1_temperature = (
        config.base_zone_temp - temp_degradation_effect +
        rng.normal(0, config.temp_noise_base * temp_noise_multiplier)
    )
    zone2_temperature = (
        config.base_zone_temp - temp_degradation_effect * 0.9 +
        rng.normal(0, config.temp_noise_base * temp_noise_multiplier)
    )
    zone3_temperature = (
        config.base_zone_temp - temp_degradation_effect * 1.1 +
        rng.normal(0, config.temp_noise_base * temp_noise_multiplier)
    )
    
    safe_temp_2 = rng.uniform(config.safe_temp_range[0], config.safe_temp_range[1])
    internal_temperature = rng.uniform(20, 45)
    
    # Gas flow generation with degradation effects
    flow_degradation_effect = (
        state.gas_delivery.mfc_calibration_drift * 0.1 +
        state.gas_delivery.valve_wear * 0.05 +
        state.gas_delivery.gas_line_contamination * 0.03
    )
    
    flow_noise_multiplier = 1.0 + state.gas_delivery.mfc_calibration_drift * 3.0
    
    # Recipe-dependent gas flows
    if "LTO" in recipe_name:
        n2_flow_rate = 10.0 + rng.normal(0, config.flow_noise_base * flow_noise_multiplier)
        o2_flow_rate = 150.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 2.0 * flow_noise_multiplier)
        sih4_flow_rate = 50.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 1.0 * flow_noise_multiplier)
    elif "PolySi" in recipe_name:
        n2_flow_rate = 5.0 + rng.normal(0, config.flow_noise_base * flow_noise_multiplier)
        o2_flow_rate = 0.0
        sih4_flow_rate = 60.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 1.5 * flow_noise_multiplier)
    else:
        n2_flow_rate = 8.0 + rng.normal(0, config.flow_noise_base * flow_noise_multiplier)
        o2_flow_rate = 100.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 1.5 * flow_noise_multiplier)
        sih4_flow_rate = 40.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 1.0 * flow_noise_multiplier)
    
    thermal_mass_flow_rate = (sih4_flow_rate + o2_flow_rate * 0.1) * (1.0 - flow_degradation_effect * 0.5)
    flow_setpoint = thermal_mass_flow_rate * (1.0 + rng.normal(0, 0.02))
    
    # Valve control with wear effects
    valve_wear_effect = state.gas_delivery.valve_wear * 10.0
    valve_drive_level = min(100.0, max(0.0, 
        thermal_mass_flow_rate / 200.0 * 100.0 + valve_wear_effect +
        rng.normal(0, config.valve_drive_precision * (1.0 + valve_wear_effect * 0.1))
    ))
    
    # Pressure with vacuum system degradation
    pressure_degradation_effect = (
        state.vacuum_system.pump_throughput_loss * 20.0 +
        state.vacuum_system.leak_accumulation * 10.0
    )
    
    pressure_noise_multiplier = 1.0 + state.vacuum_system.pump_throughput_loss * 2.0
    
    process_pressure = (
        config.target_process_pressure + pressure_degradation_effect +
        rng.normal(0, config.pressure_noise_base * pressure_noise_multiplier)
    )
    
    # Boat position with mechanical wear
    position_error = state.boat_positioning.mechanical_wear * 5.0
    boat_position = (
        config.boat_in_position + position_error +
        rng.normal(0, config.boat_position_precision * (1.0 + position_error * 0.1))
    )
    
    # Process timing
    deposition_time = f"{int(config.deposition_time_base):03d}:{rng.integers(25, 35):02d}:00"
    step_time = f"{rng.integers(15, 25):03d}:{rng.integers(10, 50):02d}:{rng.integers(10, 50):02d}"
    
    # Valve states
    valve_override_state = rng.choice(["normal", "flow_off", "purge"], p=[0.85, 0.1, 0.05])
    main_vacuum_valve = True
    n2_purge_valve = False
    process_valve = True
    door_closed_status = True
    
    # Flow totalization
    flow_totalized = state.gas_delivery.deposition_hours * thermal_mass_flow_rate * 0.06
    
    # Alarms with threshold drift
    alarm_drift = state.process_monitoring.alarm_threshold_drift * 0.1
    flow_ratio = thermal_mass_flow_rate / flow_setpoint if flow_setpoint > 0 else 1.0
    
    high_limit_alarm = flow_ratio > (config.high_flow_alarm_threshold - alarm_drift)
    low_limit_alarm = flow_ratio < (config.low_flow_alarm_threshold + alarm_drift)
    
    temp_deviation = abs(zone1_temperature - config.base_zone_temp)
    temp_power_alarm = "set" if temp_deviation > (config.temp_alarm_threshold - alarm_drift * 5.0) else "reset"
    
    vacuum_fail_alarm = process_pressure > (config.target_process_pressure * 1.5 + pressure_degradation_effect)
    
    system_error = (
        state.thermal_control.heating_element_degradation > config.heating_element_failure_threshold or
        state.vacuum_system.pump_throughput_loss > config.pump_failure_threshold or
        state.gas_delivery.mfc_calibration_drift > config.mfc_failure_threshold
    )
    
    # Timing
    logging_start_time = datetime.now() - timedelta(hours=rng.uniform(0, 24))
    logging_end_time = logging_start_time + timedelta(minutes=config.deposition_time_base + rng.uniform(-5, 10))
    
    return {
        "logging_start_time": logging_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "logging_end_time": logging_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "zone1_temperature": round(zone1_temperature, 1),
        "zone2_temperature": round(zone2_temperature, 1),
        "zone3_temperature": round(zone3_temperature, 1),
        "safe_temp_2": round(safe_temp_2, 1),
        "internal_temperature": round(internal_temperature, 1),
        "process_pressure": round(process_pressure, 1),
        "n2_flow_rate": round(n2_flow_rate, 2),
        "o2_flow_rate": round(o2_flow_rate, 1),
        "sih4_flow_rate": round(sih4_flow_rate, 1),
        "thermal_mass_flow_rate": round(thermal_mass_flow_rate, 1),
        "flow_setpoint": round(flow_setpoint, 1),
        "valve_drive_level": round(valve_drive_level, 1),
        "valve_override_state": valve_override_state,
        "flow_totalized": round(flow_totalized, 2),
        "high_limit_alarm": high_limit_alarm,
        "low_limit_alarm": low_limit_alarm,
        "system_error": system_error,
        "temp_power_alarm": temp_power_alarm,
        "vacuum_fail_alarm": vacuum_fail_alarm,
        "boat_position": round(boat_position, 1),
        "main_vacuum_valve": main_vacuum_valve,
        "n2_purge_valve": n2_purge_valve,
        "process_valve": process_valve,
        "door_closed_status": door_closed_status,
        "process_step_number": process_step_number,
        "step_time": step_time,
        "recipe_name": recipe_name,
        "lot_id": lot_id,
        "tube_name": tube_name,
        "deposition_time": deposition_time,
        "evacuation_time": None,
        "purge_time": None
    }

def generate_evacuation_event(state: SystemState, config: Config, rng: np.random.Generator) -> Dict:
    # Basic parameters
    recipe_name = rng.choice(config.recipe_names)
    tube_name = f"TUBE{rng.integers(1, 5)}"
    process_step_number = rng.integers(4, 10)
    
    # Temperature (lower during evacuation)
    temp_degradation_effect = state.thermal_control.thermocouple_drift * 2.0
    zone1_temperature = 200.0 - temp_degradation_effect + rng.normal(0, 5.0)
    zone2_temperature = 200.0 - temp_degradation_effect + rng.normal(0, 5.0)
    zone3_temperature = 200.0 - temp_degradation_effect + rng.normal(0, 5.0)
    
    safe_temp_2 = rng.uniform(config.safe_temp_range[0], config.safe_temp_range[1])
    internal_temperature = rng.uniform(20, 35)
    
    # Pressure (very low during evacuation)
    pressure_degradation_effect = (
        state.vacuum_system.pump_throughput_loss * 5.0 +
        state.vacuum_system.leak_accumulation * 3.0
    )
    process_pressure = max(0.1, 2.0 + pressure_degradation_effect + rng.normal(0, 0.5))
    
    # Boat position
    position_error = state.boat_positioning.mechanical_wear * 3.0
    boat_position = config.boat_in_position + position_error + rng.normal(0, 1.0)
    
    # Valve states for evacuation
    main_vacuum_valve = True
    n2_purge_valve = False
    process_valve = False
    door_closed_status = True
    
    # Timing
    evacuation_time = f"000:{int(config.evacuation_time_base + rng.uniform(-1, 2)):02d}:00"
    step_time = f"000:{rng.integers(3, 8):02d}:{rng.integers(10, 50):02d}"
    
    # Alarms
    vacuum_fail_alarm = process_pressure > 10.0
    system_error = state.vacuum_system.pump_throughput_loss > config.pump_failure_threshold
    
    logging_start_time = datetime.now() - timedelta(hours=rng.uniform(0, 24))
    logging_end_time = logging_start_time + timedelta(minutes=config.evacuation_time_base)
    
    return {
        "logging_start_time": logging_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "logging_end_time": logging_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "zone1_temperature": round(zone1_temperature, 1),
        "zone2_temperature": round(zone2_temperature, 1),
        "zone3_temperature": round(zone3_temperature, 1),
        "safe_temp_2": round(safe_temp_2, 1),
        "internal_temperature": round(internal_temperature, 1),
        "process_pressure": round(process_pressure, 1),
        "n2_flow_rate": None,
        "o2_flow_rate": None,
        "sih4_flow_rate": None,
        "thermal_mass_flow_rate": 0.0,
        "flow_setpoint": 0.0,
        "valve_drive_level": 0.0,
        "valve_override_state": "flow_off",
        "flow_totalized": 0.0,
        "high_limit_alarm": False,
        "low_limit_alarm": False,
        "system_error": system_error,
        "temp_power_alarm": "reset",
        "vacuum_fail_alarm": vacuum_fail_alarm,
        "boat_position": round(boat_position, 1),
        "main_vacuum_valve": main_vacuum_valve,
        "n2_purge_valve": n2_purge_valve,
        "process_valve": process_valve,
        "door_closed_status": door_closed_status,
        "process_step_number": process_step_number,
        "step_time": step_time,
        "recipe_name": recipe_name,
        "lot_id": None,
        "tube_name": tube_name,
        "deposition_time": None,
        "evacuation_time": evacuation_time,
        "purge_time": None
    }

def generate_purge_event(state: SystemState, config: Config, rng: np.random.Generator) -> Dict:
    # Basic parameters
    recipe_name = rng.choice(config.recipe_names)
    tube_name = f"TUBE{rng.integers(1, 5)}"
    process_step_number = rng.integers(7, 16)
    
    # Temperature (moderate during purge)
    temp_degradation_effect = state.thermal_control.thermocouple_drift * 1.5
    zone1_temperature = 300.0 - temp_degradation_effect + rng.normal(0, 3.0)
    zone2_temperature = 300.0 - temp_degradation_effect + rng.normal(0, 3.0)
    zone3_temperature = 300.0 - temp_degradation_effect + rng.normal(0, 3.0)
    
    safe_temp_2 = rng.uniform(config.safe_temp_range[0], config.safe_temp_range[1])
    internal_temperature = rng.uniform(25, 40)
    
    # N2 flow for purging
    flow_degradation_effect = state.gas_delivery.mfc_calibration_drift * 0.05
    n2_flow_rate = 10.0 * (1.0 - flow_degradation_effect) + rng.normal(0, 0.5)
    thermal_mass_flow_rate = n2_flow_rate
    flow_setpoint = n2_flow_rate * 1.02
    
    # Valve control
    valve_drive_level = min(100.0, n2_flow_rate / 20.0 * 100.0 + rng.normal(0, 2.0))
    
    # Pressure (atmospheric during purge)
    process_pressure = 760.0 + rng.normal(0, 5.0)
    
    # Boat position
    position_error = state.boat_positioning.mechanical_wear * 2.0
    boat_position = config.boat_in_position + position_error + rng.normal(0, 1.0)
    
    # Valve states for purge
    main_vacuum_valve = False
    n2_purge_valve = True
    process_valve = True
    door_closed_status = True
    
    # Timing
    purge_time = f"000:{int(config.purge_time_base + rng.uniform(-1, 2)):02d}:00"
    step_time = f"000:{rng.integers(3, 8):02d}:{rng.integers(10, 50):02d}"
    
    # Flow totalization
    flow_totalized = state.gas_delivery.operating_hours * n2_flow_rate * 0.06
    
    # Alarms
    flow_ratio = thermal_mass_flow_rate / flow_setpoint if flow_setpoint > 0 else 1.0
    high_limit_alarm = flow_ratio > 1.1
    low_limit_alarm = flow_ratio < 0.9
    
    logging_start_time = datetime.now() - timedelta(hours=rng.uniform(0, 24))
    logging_end_time = logging_start_time + timedelta(minutes=config.purge_time_base)
    
    return {
        "logging_start_time": logging_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "logging_end_time": logging_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "zone1_temperature": round(zone1_temperature, 1),
        "zone2_temperature": round(zone2_temperature, 1),
        "zone3_temperature": round(zone3_temperature, 1),
        "safe_temp_2": round(safe_temp_2, 1),
        "internal_temperature": round(internal_temperature, 1),
        "process_pressure": round(process_pressure, 1),
        "n2_flow_rate": round(n2_flow_rate, 2),
        "o2_flow_rate": 0.0,
        "sih4_flow_rate": 0.0,
        "thermal_mass_flow_rate": round(thermal_mass_flow_rate, 1),
        "flow_setpoint": round(flow_setpoint, 1),
        "valve_drive_level": round(valve_drive_level, 1),
        "valve_override_state": "normal",
        "flow_totalized": round(flow_totalized, 2),
        "high_limit_alarm": high_limit_alarm,
        "low_limit_alarm": low_limit_alarm,
        "system_error": False,
        "temp_power_alarm": "reset",
        "vacuum_fail_alarm": False,
        "boat_position": round(boat_position, 1),
        "main_vacuum_valve": main_vacuum_valve,
        "n2_purge_valve": n2_purge_valve,
        "process_valve": process_valve,
        "door_closed_status": door_closed_status,
        "process_step_number": process_step_number,
        "step_time": step_time,
        "recipe_name": recipe_name,
        "lot_id": None,
        "tube_name": tube_name,
        "deposition_time": None,
        "evacuation_time": None,
        "purge_time": purge_time
    }

def generate_boat_movement_event(state: SystemState, config: Config, rng: np.random.Generator) -> Dict:
    # Basic parameters
    recipe_name = rng.choice(config.recipe_names)
    lot_id = f"LOT{rng.integers(1000, 9999)}" if rng.random() > 0.5 else None
    tube_name = f"TUBE{rng.integers(1, 5)}"
    process_step_number = rng.integers(1, 5)
    
    # Temperature (ambient during boat movement)
    zone1_temperature = 25.0 + rng.normal(0, 2.0)
    zone2_temperature = 25.0 + rng.normal(0, 2.0)
    zone3_temperature = 25.0 + rng.normal(0, 2.0)
    
    safe_temp_2 = rng.uniform(config.safe_temp_range[0], config.safe_temp_range[1])
    internal_temperature = rng.uniform(20, 30)
    
    # Boat position (either in or out)
    position_error = state.boat_positioning.mechanical_wear * 8.0
    if rng.random() > 0.5:
        boat_position = config.boat_out_position + position_error + rng.normal(0, 2.0)
    else:
        boat_position = config.boat_in_position + position_error + rng.normal(0, 2.0)
    
    # Valve states (atmospheric conditions)
    main_vacuum_valve = False
    n2_purge_valve = True
    process_valve = False
    door_closed_status = rng.random() > 0.3
    
    # Timing
    step_time = f"000:{rng.integers(8, 15):02d}:{rng.integers(10, 50):02d}"
    
    logging_start_time = datetime.now() - timedelta(hours=rng.uniform(0, 24))
    logging_end_time = logging_start_time + timedelta(minutes=10)
    
    return {
        "logging_start_time": logging_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "logging_end_time": logging_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "zone1_temperature": round(zone1_temperature, 1),
        "zone2_temperature": round(zone2_temperature, 1),
        "zone3_temperature": round(zone3_temperature, 1),
        "safe_temp_2": round(safe_temp_2, 1),
        "internal_temperature": round(internal_temperature, 1),
        "process_pressure": None,
        "n2_flow_rate": 5.0,
        "o2_flow_rate": 0.0,
        "sih4_flow_rate": 0.0,
        "thermal_mass_flow_rate": 5.0,
        "flow_setpoint": 5.0,
        "valve_drive_level": 25.0,
        "valve_override_state": "normal",
        "flow_totalized": 0.0,
        "high_limit_alarm": False,
        "low_limit_alarm": False,
        "system_error": False,
        "temp_power_alarm": "reset",
        "vacuum_fail_alarm": False,
        "boat_position": round(boat_position, 1),
        "main_vacuum_valve": main_vacuum_valve,
        "n2_purge_valve": n2_purge_valve,
        "process_valve": process_valve,
        "door_closed_status": door_closed_status,
        "process_step_number": process_step_number,
        "step_time": step_time,
        "recipe_name": recipe_name,
        "lot_id": lot_id,
        "tube_name": tube_name,
        "deposition_time": None,
        "evacuation_time": None,
        "purge_time": None
    }

def generate_maintenance_event(state: SystemState, config: Config, rng: np.random.Generator) -> Dict:
    # Basic parameters
    recipe_name = "MAINTENANCE"
    tube_name = f"TUBE{rng.integers(1, 5)}"
    process_step_number = 0
    
    # Temperature (elevated for some maintenance)
    if rng.random() > 0.5:
        temp_base = 400.0
    else:
        temp_base = 25.0
    
    zone1_temperature = temp_base + rng.normal(0, 5.0)
    zone2_temperature = temp_base + rng.normal(0, 5.0)
    zone3_temperature = temp_base + rng.normal(0, 5.0)
    
    safe_temp_2 = rng.uniform(config.safe_temp_range[0], config.safe_temp_range[1])
    internal_temperature = rng.uniform(20, 50)
    
    # Boat position (typically out for maintenance)
    position_error = state.boat_positioning.mechanical_wear * 5.0
    boat_position = config.boat_out_position + position_error + rng.normal(0, 3.0)
    
    # Gas flows (minimal during maintenance)
    n2_flow_rate = 2.0 + rng.normal(0, 0.5)
    thermal_mass_flow_rate = n2_flow_rate
    flow_setpoint = n2_flow_rate
    valve_drive_level = 10.0 + rng.normal(0, 2.0)
    
    # Valve states
    main_vacuum_valve = False
    n2_purge_valve = True
    process_valve = False
    door_closed_status = False
    
    # Timing
    step_time = f"001:{rng.integers(0, 59):02d}:{rng.integers(0, 59):02d}"
    
    logging_start_time = datetime.now() - timedelta(hours=rng.uniform(0, 24))
    logging_end_time = logging_start_time + timedelta(hours=2)
    
    return {
        "logging_start_time": logging_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "logging_end_time": logging_end_time.strftime("%Y-%m-%d %H:%M:%S"),
        "zone1_temperature": round(zone1_temperature, 1),
        "zone2_temperature": round(zone2_temperature, 1),
        "zone3_temperature": round(zone3_temperature, 1),
        "safe_temp_2": round(safe_temp_2, 1),
        "internal_temperature": round(internal_temperature, 1),
        "process_pressure": None,
        "n2_flow_rate": round(n2_flow_rate, 2),
        "o2_flow_rate": None,
        "sih4_flow_rate": None,
        "thermal_mass_flow_rate": round(thermal_mass_flow_rate, 1),
        "flow_setpoint": round(flow_setpoint, 1),
        "valve_drive_level": round(valve_drive_level, 1),
        "valve_override_state": "normal",
        "flow_totalized": 0.0,
        "high_limit_alarm": False,
        "low_limit_alarm": False,
        "system_error": False,
        "temp_power_alarm": "reset",
        "vacuum_fail_alarm": False,
        "boat_position": round(boat_position, 1),
        "main_vacuum_valve": main_vacuum_valve,
        "n2_purge_valve": n2_purge_valve,
        "process_valve": process_valve,
        "door_closed_status": door_closed_status,
        "process_step_number": process_step_number,
        "step_time": step_time,
        "recipe_name": recipe_name,
        "lot_id": None,
        "tube_name": tube_name,
        "deposition_time": None,
        "evacuation_time": None,
        "purge_time": None
    }

def generate_event(state: SystemState, config: Config, operation_type: str, rng: np.random.Generator) -> Dict:
    if operation_type == "deposition":
        return generate_deposition_event(state, config, rng)
    elif operation_type == "evacuation":
        return generate_evacuation_event(state, config, rng)
    elif operation_type == "purge":
        return generate_purge_event(state, config, rng)
    elif operation_type == "boat_movement":
        return generate_boat_movement_event(state, config, rng)
    elif operation_type == "maintenance":
        return generate_maintenance_event(state, config, rng)
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")

def simulate_furnace(usage_profile: UsageProfile, config: Config, simulation_days: int = 30, 
                    random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    state = SystemState()
    events = []
    maintenance_log = []
    
    # Calculate event rate (events per hour)
    events_per_day = usage_profile.operating_hours_per_day * usage_profile.intensity_multiplier
    events_per_hour = events_per_day / 24.0
    
    current_time = 0.0
    simulation_hours = simulation_days * 24.0
    
    while current_time < simulation_hours:
        # Generate next event time (Poisson process)
        inter_arrival_time = rng.exponential(1.0 / events_per_hour)
        current_time += inter_arrival_time
        
        if current_time >= simulation_hours:
            break
        
        # Check for maintenance triggers
        maintenance_events = check_maintenance_triggers(state, config)
        
        # Perform maintenance if needed
        for maintenance_type in maintenance_events:
            perform_maintenance(state, config, maintenance_type)
            maintenance_log.append({
                "time_hours": current_time,
                "maintenance_type": maintenance_type,
                "tube_contamination": state.process_tube.tube_contamination,
                "mfc_drift": state.gas_delivery.mfc_calibration_drift,
                "heating_degradation": state.thermal_control.heating_element_degradation,
                "pump_degradation": state.vacuum_system.pump_throughput_loss
            })
        
        # Select operation type based on usage profile
        operation_type = rng.choice(
            list(usage_profile.operation_mix.keys()),
            p=list(usage_profile.operation_mix.values())
        )
        
        # Generate event
        event_data = generate_event(state, config, operation_type, rng)
        
        # Determine event duration
        if operation_type == "deposition":
            duration_hours = config.deposition_time_base / 60.0
        elif operation_type == "evacuation":
            duration_hours = config.evacuation_time_base / 60.0
        elif operation_type == "purge":
            duration_hours = config.purge_time_base / 60.0
        elif operation_type == "boat_movement":
            duration_hours = 10.0 / 60.0
        elif operation_type == "maintenance":
            duration_hours = 2.0
        else:
            duration_hours = 0.5
        
        # Update degradation state
        update_degradation_state(state, config, operation_type, duration_hours, rng)
        
        # Add metadata to event
        event_data["operation_type"] = operation_type
        event_data["time_hours"] = current_time
        event_data["duration_hours"] = duration_hours
        
        events.append(event_data)
    
    events_df = pd.DataFrame(events)
    maintenance_df = pd.DataFrame(maintenance_log)
    
    return events_df, maintenance_df

if __name__ == "__main__":
    config = Config()
    
    # Define usage profiles
    usage_profiles = [
        UsageProfile(
            name="production",
            description="High-volume production with continuous operation",
            operating_hours_per_day=22.0,
            relative_intensity="high",
            operation_mix={
                "deposition": 0.7,
                "evacuation": 0.1,
                "purge": 0.1,
                "boat_movement": 0.1
            }
        ),
        UsageProfile(
            name="development",
            description="R&D operation with varied recipes and frequent changes",
            operating_hours_per_day=10.0,
            relative_intensity="medium",
            operation_mix={
                "deposition": 0.5,
                "evacuation": 0.2,
                "purge": 0.2,
                "boat_movement": 0.1
            }
        ),
        UsageProfile(
            name="maintenance",
            description="Periodic maintenance and calibration operations",
            operating_hours_per_day=3.0,
            relative_intensity="low",
            operation_mix={
                "maintenance": 0.8,
                "evacuation": 0.1,
                "purge": 0.1
            }
        )
    ]
    
    # Run simulations
    for profile in usage_profiles:
        print(f"\nSimulating {profile.name} usage profile...")
        
        events_df, maintenance_df = simulate_furnace(
            usage_profile=profile,
            config=config,
            simulation_days=30,
            random_seed=42
        )
        
        # Print summary
        print(f"Generated {len(events_df)} events")
        print(f"Generated {len(maintenance_df)} maintenance events")
        
        operation_counts = events_df['operation_type'].value_counts()
        print("Operation type distribution:")
        for op_type, count in operation_counts.items():
            print(f"  {op_type}: {count}")
        
        # Check for degradation trends
        if len(events_df) > 100:
            early_events = events_df.head(int(len(events_df) * 0.2))
            late_events = events_df.tail(int(len(events_df) * 0.2))
            
            deposition_events = events_df[events_df['operation_type'] == 'deposition']
            if len(deposition_events) > 50:
                early_dep = deposition_events.head(int(len(deposition_events) * 0.2))
                late_dep = deposition_events.tail(int(len(deposition_events) * 0.2))
                
                early_temp_std = early_dep['zone1_temperature'].std()
                late_temp_std = late_dep['zone1_temperature'].std()
                early_pressure_mean = early_dep['process_pressure'].mean()
                late_pressure_mean = late_dep['process_pressure'].mean()
                
                print(f"Temperature variability: early={early_temp_std:.2f}, late={late_temp_std:.2f}")
                print(f"Pressure drift: early={early_pressure_mean:.1f}, late={late_pressure_mean:.1f}")
        
        # Save to CSV
        events_filename = f"tempress_lpcvd_events_{profile.name}.csv"
        maintenance_filename = f"tempress_lpcvd_maintenance_{profile.name}.csv"
        
        events_df.to_csv(events_filename, index=False)
        maintenance_df.to_csv(maintenance_filename, index=False)
        
        print(f"Saved {events_filename} and {maintenance_filename}")
    
    print("\nSimulation complete!")