import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class OperationType(Enum):
    VELOCITY_CONTROL = "velocity_control"
    ORIENTATION = "orientation"
    RIGID_TAPPING = "rigid_tapping"
    SYNCHRONOUS_CONTROL = "synchronous_control"

@dataclass
class Config:
    # Operating parameters
    max_motor_speed: float = 8000.0  # rpm
    max_torque: float = 100.0  # %
    max_temperature: float = 150.0  # C
    overheat_threshold: float = 140.0  # C
    
    # Speed detection thresholds
    speed_arrival_threshold: float = 15.0  # % of command
    zero_speed_threshold: float = 0.75  # % of max speed
    
    # Position and velocity error limits
    max_position_error: float = 4096.0  # pulses
    max_velocity_error: float = 256.0  # rpm
    max_synchronous_error: float = 512.0  # pulses
    
    # Vibration severity thresholds (mm/s RMS)
    vibration_smooth_threshold: float = 1.8
    vibration_rough_threshold: float = 5.4
    vibration_critical_threshold: float = 10.7
    
    # Vibration acceleration thresholds (g RMS)
    acceleration_normal_threshold: float = 0.2
    acceleration_critical_threshold: float = 0.4
    
    # Degradation rates (per 1000 operating hours)
    bearing_wear_rate: float = 0.02
    preload_degradation_rate: float = 0.015
    grease_degradation_rate: float = 0.03
    contamination_buildup_rate: float = 0.01
    
    # Thermal effects
    thermal_expansion_coefficient: float = 0.1  # error per degree
    thermal_time_constant: float = 0.8  # hours
    
    # Noise parameters
    base_vibration_velocity: float = 0.5  # mm/s
    base_vibration_acceleration: float = 0.05  # g
    base_position_noise: float = 10.0  # pulses
    base_velocity_noise: float = 5.0  # rpm
    base_temperature_noise: float = 2.0  # C
    
    # Maintenance restoration factors
    lubrication_restoration: float = 0.9
    bearing_replacement_restoration: float = 1.0
    preload_adjustment_restoration: float = 0.7
    
    # Failure thresholds
    critical_bearing_wear: float = 0.8
    critical_temperature: float = 150.0
    critical_vibration: float = 10.7
    critical_position_error: float = 2000.0

@dataclass
class SpindleMotorState:
    temperature: float = 25.0
    rotor_magnetic_alignment: float = 1.0
    winding_insulation: float = 1.0
    thermal_history: float = 0.0

@dataclass
class BearingSystemState:
    wear_level: float = 0.0
    temperature: float = 25.0
    preload_degradation: float = 0.0
    lubrication_condition: float = 1.0

@dataclass
class ControlSystemState:
    accuracy: float = 1.0
    thermal_drift: float = 0.0

@dataclass
class PositionFeedbackState:
    encoder_accuracy: float = 1.0
    signal_quality: float = 1.0

@dataclass
class LubricationSystemState:
    grease_degradation: float = 0.0
    contamination_level: float = 0.0

@dataclass
class SystemState:
    spindle_motor: SpindleMotorState
    bearing_system: BearingSystemState
    control_system: ControlSystemState
    position_feedback: PositionFeedbackState
    lubrication_system: LubricationSystemState
    operating_hours: float = 0.0
    last_maintenance_hours: float = 0.0

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

def create_usage_profiles() -> List[UsageProfile]:
    return [
        UsageProfile(
            name="light_machining",
            description="Low-load finishing operations",
            operating_hours_per_day=10.0,
            relative_intensity="low",
            operation_mix={
                "velocity_control": 0.85,
                "orientation": 0.1,
                "rigid_tapping": 0.05,
                "synchronous_control": 0.0
            }
        ),
        UsageProfile(
            name="production_machining",
            description="Standard production operations",
            operating_hours_per_day=18.0,
            relative_intensity="medium",
            operation_mix={
                "velocity_control": 0.7,
                "orientation": 0.2,
                "rigid_tapping": 0.1,
                "synchronous_control": 0.0
            }
        ),
        UsageProfile(
            name="heavy_duty",
            description="High-load continuous operations",
            operating_hours_per_day=22.0,
            relative_intensity="high",
            operation_mix={
                "velocity_control": 0.8,
                "orientation": 0.15,
                "rigid_tapping": 0.05,
                "synchronous_control": 0.0
            }
        )
    ]

def update_degradation(state: SystemState, config: Config, hours_elapsed: float, 
                      intensity_multiplier: float) -> None:
    degradation_factor = hours_elapsed * intensity_multiplier / 1000.0
    
    # Bearing wear with nonlinear acceleration
    wear_acceleration = 1.0 + state.bearing_system.wear_level * 2.0
    state.bearing_system.wear_level += config.bearing_wear_rate * degradation_factor * wear_acceleration
    state.bearing_system.wear_level = min(state.bearing_system.wear_level, 1.0)
    
    # Preload degradation with thermal cycling effects
    thermal_cycles = state.spindle_motor.thermal_history * 0.1
    preload_factor = 1.0 + thermal_cycles
    state.bearing_system.preload_degradation += config.preload_degradation_rate * degradation_factor * preload_factor
    state.bearing_system.preload_degradation = min(state.bearing_system.preload_degradation, 1.0)
    
    # Grease degradation with temperature dependency
    temp_factor = 1.0 + max(0, (state.spindle_motor.temperature - 50) / 100)
    state.lubrication_system.grease_degradation += config.grease_degradation_rate * degradation_factor * temp_factor
    state.lubrication_system.grease_degradation = min(state.lubrication_system.grease_degradation, 1.0)
    
    # Contamination buildup with wear interaction
    contamination_factor = 1.0 + state.bearing_system.wear_level * 3.0
    state.lubrication_system.contamination_level += config.contamination_buildup_rate * degradation_factor * contamination_factor
    state.lubrication_system.contamination_level = min(state.lubrication_system.contamination_level, 1.0)
    
    # Update lubrication condition based on degradation and contamination
    state.bearing_system.lubrication_condition = max(0.1, 
        1.0 - state.lubrication_system.grease_degradation - state.lubrication_system.contamination_level * 0.5)

def check_maintenance_triggers(state: SystemState, config: Config) -> Optional[str]:
    hours_since_maintenance = state.operating_hours - state.last_maintenance_hours
    
    # Scheduled maintenance every 2000 hours
    if hours_since_maintenance > 2000:
        return "scheduled_maintenance"
    
    # Temperature-based maintenance
    if state.spindle_motor.temperature > 120:
        return "temperature_alarm"
    
    # Vibration-based maintenance (estimated from bearing condition)
    estimated_vibration = config.base_vibration_velocity * (1 + state.bearing_system.wear_level * 10)
    if estimated_vibration > config.vibration_rough_threshold:
        return "vibration_alarm"
    
    # Lubrication condition maintenance
    if state.bearing_system.lubrication_condition < 0.3:
        return "lubrication_degradation"
    
    return None

def perform_maintenance(state: SystemState, config: Config, maintenance_type: str) -> Dict:
    maintenance_record = {
        "timestamp": state.operating_hours,
        "maintenance_type": maintenance_type,
        "bearing_wear_before": state.bearing_system.wear_level,
        "lubrication_condition_before": state.bearing_system.lubrication_condition,
        "contamination_before": state.lubrication_system.contamination_level
    }
    
    if maintenance_type in ["scheduled_maintenance", "lubrication_degradation", "temperature_alarm"]:
        # Lubrication service
        state.lubrication_system.grease_degradation *= (1 - config.lubrication_restoration)
        state.lubrication_system.contamination_level *= 0.1
        state.bearing_system.lubrication_condition = min(1.0, 
            state.bearing_system.lubrication_condition + 0.6)
        maintenance_record["action"] = "lubrication_service"
        
    elif maintenance_type == "vibration_alarm":
        # Bearing replacement
        state.bearing_system.wear_level *= (1 - config.bearing_replacement_restoration)
        state.bearing_system.preload_degradation *= 0.2
        state.lubrication_system.grease_degradation = 0.0
        state.lubrication_system.contamination_level = 0.0
        state.bearing_system.lubrication_condition = 1.0
        maintenance_record["action"] = "bearing_replacement"
    
    state.last_maintenance_hours = state.operating_hours
    
    maintenance_record.update({
        "bearing_wear_after": state.bearing_system.wear_level,
        "lubrication_condition_after": state.bearing_system.lubrication_condition,
        "contamination_after": state.lubrication_system.contamination_level
    })
    
    return maintenance_record

def select_operation_type(operation_mix: Dict[str, float], rng: np.random.Generator) -> OperationType:
    operations = list(operation_mix.keys())
    probabilities = list(operation_mix.values())
    selected = rng.choice(operations, p=probabilities)
    return OperationType(selected)

def generate_event(state: SystemState, config: Config, operation_type: OperationType,
                  rng: np.random.Generator) -> Dict:
    
    # Update thermal state
    ambient_temp = 25.0
    thermal_load = 20.0 + state.bearing_system.wear_level * 30.0
    state.spindle_motor.temperature = (state.spindle_motor.temperature * 0.9 + 
                                     (ambient_temp + thermal_load) * 0.1)
    state.bearing_system.temperature = state.spindle_motor.temperature + 5.0
    state.spindle_motor.thermal_history += abs(state.spindle_motor.temperature - ambient_temp) * 0.01
    
    # Generate control inputs based on operation type
    if operation_type == OperationType.VELOCITY_CONTROL:
        spindle_speed_command = rng.uniform(1000, config.max_motor_speed * 0.8)
        cutting_load = rng.uniform(0.2, 0.8)
    elif operation_type == OperationType.ORIENTATION:
        spindle_speed_command = rng.uniform(100, 500)
        cutting_load = 0.1
    elif operation_type == OperationType.RIGID_TAPPING:
        spindle_speed_command = rng.uniform(200, 2000)
        cutting_load = rng.uniform(0.3, 0.6)
    else:  # SYNCHRONOUS_CONTROL
        spindle_speed_command = rng.uniform(500, 3000)
        cutting_load = rng.uniform(0.4, 0.7)
    
    maximum_motor_speed = config.max_motor_speed
    spindle_speed_command = min(spindle_speed_command, maximum_motor_speed)
    
    # Calculate velocity error with degradation effects
    base_velocity_error = config.base_velocity_noise * rng.normal(0, 1)
    wear_velocity_error = state.bearing_system.wear_level * 50.0 * rng.normal(0, 1)
    lubrication_velocity_error = (1 - state.bearing_system.lubrication_condition) * 30.0 * rng.normal(0, 1)
    velocity_error = base_velocity_error + wear_velocity_error + lubrication_velocity_error
    velocity_error = np.clip(velocity_error, -config.max_velocity_error, config.max_velocity_error)
    
    # Calculate actual spindle speed
    actual_spindle_speed = max(0, spindle_speed_command - abs(velocity_error))
    
    # Calculate torque command
    base_torque = cutting_load * 60.0
    friction_torque = state.bearing_system.wear_level * 20.0 + (1 - state.bearing_system.lubrication_condition) * 15.0
    torque_command = min(config.max_torque, base_torque + friction_torque)
    
    # Calculate position error with degradation effects
    base_position_error = config.base_position_noise * rng.normal(0, 1)
    wear_position_error = state.bearing_system.wear_level * 200.0 * rng.normal(0, 1)
    preload_position_error = state.bearing_system.preload_degradation * 150.0 * rng.normal(0, 1)
    thermal_position_error = (state.spindle_motor.temperature - 25.0) * config.thermal_expansion_coefficient * rng.normal(0, 1)
    position_error = base_position_error + wear_position_error + preload_position_error + thermal_position_error
    position_error = np.clip(position_error, -config.max_position_error, config.max_position_error)
    
    # Calculate synchronous error
    synchronous_error = None
    if operation_type == OperationType.SYNCHRONOUS_CONTROL:
        base_sync_error = 20.0 * rng.normal(0, 1)
        wear_sync_error = state.bearing_system.wear_level * 100.0 * rng.normal(0, 1)
        thermal_sync_error = state.control_system.thermal_drift * 50.0 * rng.normal(0, 1)
        synchronous_error = base_sync_error + wear_sync_error + thermal_sync_error
        synchronous_error = np.clip(synchronous_error, -config.max_synchronous_error, config.max_synchronous_error)
    
    # Calculate vibration with nonlinear degradation effects
    base_vib_vel = config.base_vibration_velocity
    wear_multiplier = 1.0 + state.bearing_system.wear_level * 15.0 * (1 + state.bearing_system.wear_level)
    lubrication_multiplier = 1.0 + (1 - state.bearing_system.lubrication_condition) * 8.0
    contamination_multiplier = 1.0 + state.lubrication_system.contamination_level * 12.0
    
    vibration_velocity = base_vib_vel * wear_multiplier * lubrication_multiplier * (1 + rng.normal(0, 0.3))
    vibration_velocity = max(0.1, vibration_velocity)
    
    base_vib_acc = config.base_vibration_acceleration
    acc_wear_multiplier = 1.0 + state.bearing_system.wear_level * 6.0
    acc_contamination_multiplier = 1.0 + state.lubrication_system.contamination_level * 8.0
    
    vibration_acceleration = base_vib_acc * acc_wear_multiplier * acc_contamination_multiplier * (1 + rng.normal(0, 0.4))
    vibration_acceleration = max(0.01, vibration_acceleration)
    
    # Calculate motor voltage
    motor_voltage = min(100.0, 30.0 + torque_command * 0.6 + state.bearing_system.wear_level * 10.0)
    
    # Calculate motor current
    base_current = torque_command * 0.8
    friction_current = state.bearing_system.wear_level * 15.0 + (1 - state.bearing_system.lubrication_condition) * 10.0
    motor_current = base_current + friction_current
    
    # Calculate timing parameters
    base_accel_time = 2.0
    wear_accel_penalty = state.bearing_system.wear_level * 3.0
    lubrication_accel_penalty = (1 - state.bearing_system.lubrication_condition) * 2.0
    acceleration_deceleration_time = int(base_accel_time + wear_accel_penalty + lubrication_accel_penalty)
    
    orientation_time = None
    if operation_type == OperationType.ORIENTATION:
        base_orient_time = 500.0
        wear_orient_penalty = state.bearing_system.wear_level * 2000.0
        preload_orient_penalty = state.bearing_system.preload_degradation * 1500.0
        orientation_time = int(base_orient_time + wear_orient_penalty + preload_orient_penalty)
    
    # Calculate motor power off delay
    motor_power_off_delay = int(200 + state.bearing_system.wear_level * 100)
    
    # Calculate detection and outcome signals
    speed_arrival_threshold_rpm = spindle_speed_command * config.speed_arrival_threshold / 100.0
    speed_arrival_detection = abs(actual_spindle_speed - spindle_speed_command) <= speed_arrival_threshold_rpm
    speed_arrival_signal = speed_arrival_detection
    
    zero_speed_threshold_rpm = maximum_motor_speed * config.zero_speed_threshold / 100.0
    zero_speed_detection = actual_spindle_speed <= zero_speed_threshold_rpm
    
    orientation_completed = None
    if operation_type == OperationType.ORIENTATION:
        orientation_completed = abs(position_error) <= 100.0
    
    # Calculate load detection levels
    load_detection_level_1 = min(100.0, (motor_current / 80.0) * 100.0)
    
    # Determine spindle sequence state
    spindle_sequence_state = "a"
    if operation_type == OperationType.ORIENTATION:
        if orientation_time and orientation_time < 1000:
            spindle_sequence_state = "e"
        elif abs(velocity_error) < 10:
            spindle_sequence_state = "d"
        else:
            spindle_sequence_state = "c"
    
    # Calculate current command
    current_command = motor_current
    
    # Determine operation mode
    operation_mode_map = {
        OperationType.VELOCITY_CONTROL: "velocity_control",
        OperationType.ORIENTATION: "orientation",
        OperationType.RIGID_TAPPING: "rigid_tapping",
        OperationType.SYNCHRONOUS_CONTROL: "synchronous_control"
    }
    operation_mode = operation_mode_map[operation_type]
    
    # Calculate acceleration value
    acceleration_value = max(100.0, 1000.0 - state.bearing_system.wear_level * 500.0)
    
    # Determine orientation stop position
    orientation_stop_position = int(rng.uniform(0, 4095)) if operation_type == OperationType.ORIENTATION else 0
    
    # Calculate torque limitation
    torque_limitation_active = torque_command >= config.max_torque * 0.9
    
    # Calculate alarm conditions
    temperature_alarm = state.spindle_motor.temperature > config.overheat_threshold
    vibration_alarm = vibration_velocity > config.vibration_critical_threshold
    position_alarm = abs(position_error) > config.critical_position_error
    spindle_alarm = temperature_alarm or vibration_alarm or position_alarm
    
    alarm_detection_status = spindle_alarm
    
    # Calculate deceleration time constant
    deceleration_time_constant = acceleration_deceleration_time * 0.8
    
    # Build event record
    event = {
        "spindle_speed_command": spindle_speed_command,
        "actual_spindle_speed": actual_spindle_speed,
        "torque_command": torque_command,
        "position_error": position_error,
        "velocity_error": velocity_error,
        "synchronous_error": synchronous_error,
        "motor_voltage": motor_voltage,
        "acceleration_deceleration_time": acceleration_deceleration_time,
        "orientation_time": orientation_time,
        "speed_arrival_signal": speed_arrival_signal,
        "zero_speed_detection": zero_speed_detection,
        "load_detection_level_1": load_detection_level_1,
        "vibration_velocity": vibration_velocity,
        "vibration_acceleration": vibration_acceleration,
        "current_command": current_command,
        "spindle_sequence_state": spindle_sequence_state,
        "speed_arrival_detection": speed_arrival_detection,
        "orientation_completed": orientation_completed,
        "maximum_motor_speed": maximum_motor_speed,
        "motor_power_off_delay": motor_power_off_delay,
        "torque_limitation_active": torque_limitation_active,
        "spindle_alarm": spindle_alarm,
        "operation_mode": operation_mode,
        "acceleration_value": acceleration_value,
        "motor_current": motor_current,
        "motor_temperature": state.spindle_motor.temperature,
        "orientation_stop_position": orientation_stop_position,
        "alarm_detection_status": alarm_detection_status,
        "deceleration_time_constant": deceleration_time_constant
    }
    
    return event

def simulate_spindle(usage_profile: UsageProfile, config: Config, 
                    simulation_days: int = 30, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    rng = np.random.default_rng(seed)
    
    state = SystemState(
        spindle_motor=SpindleMotorState(),
        bearing_system=BearingSystemState(),
        control_system=ControlSystemState(),
        position_feedback=PositionFeedbackState(),
        lubrication_system=LubricationSystemState()
    )
    
    events = []
    maintenance_log = []
    
    # Calculate event arrival rate (Poisson process)
    events_per_hour = 12.0 * usage_profile.intensity_multiplier
    total_hours = simulation_days * 24
    
    current_time = 0.0
    
    while current_time < total_hours:
        # Generate next event time (Poisson process)
        inter_arrival_time = rng.exponential(1.0 / events_per_hour)
        current_time += inter_arrival_time
        
        if current_time >= total_hours:
            break
        
        # Update operating hours only during operating periods
        operating_fraction = usage_profile.operating_hours_per_day / 24.0
        if rng.random() < operating_fraction:
            state.operating_hours += inter_arrival_time
            
            # Update degradation
            update_degradation(state, config, inter_arrival_time, usage_profile.intensity_multiplier)
            
            # Check for maintenance
            maintenance_type = check_maintenance_triggers(state, config)
            if maintenance_type:
                maintenance_record = perform_maintenance(state, config, maintenance_type)
                maintenance_log.append(maintenance_record)
            
            # Select operation type
            operation_type = select_operation_type(usage_profile.operation_mix, rng)
            
            # Generate event
            event = generate_event(state, config, operation_type, rng)
            event["timestamp"] = current_time
            event["operating_hours"] = state.operating_hours
            events.append(event)
    
    events_df = pd.DataFrame(events)
    maintenance_df = pd.DataFrame(maintenance_log)
    
    return events_df, maintenance_df

if __name__ == "__main__":
    config = Config()
    usage_profiles = create_usage_profiles()
    
    print("CNC Spindle Unit Simulator")
    print("=" * 50)
    
    for profile in usage_profiles:
        print(f"\nSimulating {profile.name} profile...")
        
        events_df, maintenance_df = simulate_spindle(profile, config, simulation_days=30)
        
        print(f"Generated {len(events_df)} events")
        print(f"Maintenance events: {len(maintenance_df)}")
        
        if len(events_df) > 0:
            print(f"Vibration velocity range: {events_df['vibration_velocity'].min():.2f} - {events_df['vibration_velocity'].max():.2f} mm/s")
            print(f"Temperature range: {events_df['motor_temperature'].min():.1f} - {events_df['motor_temperature'].max():.1f} °C")
            print(f"Alarm rate: {events_df['spindle_alarm'].mean()*100:.1f}%")
            
            # Save files
            events_filename = f"cnc_spindle_events_{profile.name}.csv"
            maintenance_filename = f"cnc_spindle_maintenance_{profile.name}.csv"
            
            events_df.to_csv(events_filename, index=False)
            print(f"Saved events to {events_filename}")
            
            if len(maintenance_df) > 0:
                maintenance_df.to_csv(maintenance_filename, index=False)
                print(f"Saved maintenance log to {maintenance_filename}")
    
    print("\nSimulation complete!")