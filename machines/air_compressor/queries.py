manuals = [
    # OEM manuals
    {"filename": "atlas_copco_air_compressor.pdf", "source_type": "oem_manual"},
    # Component datasheets
    {"filename": "compressed_air_manual.pdf", "source_type": "component_datasheet"},
]

domain_context = """
Atlas Copco GA series oil-injected rotary screw air compressor used in industrial
compressed air supply.
Key output: dry, filtered compressed air at regulated pressure.
Key failure consequences:
- Oil carryover: contaminated downstream air, fouled dryer/filters, process contamination
- Overtemperature shutdown: unplanned downtime, potential screw element seizure
- Unloader/pressure regulation failure: pressure instability, inability to meet demand
- Air dryer failure (Full-feature): high dewpoint, moisture in air lines
- Oil separator degradation: excessive oil consumption, high differential pressure
- Belt/coupling wear: vibration, power loss, eventual drive failure
Primary degradation story: gradual increase in oil separator differential pressure,
air filter restriction, cooler fouling, oil degradation over service hours, and
Elektronikon sensor drift accumulating over thousands of operating hours.
"""

manual_exclusions = []
manual_inclusions = []

# Phase 2 — Schema extraction queries
schema_queries = [
    # Pressure
    "outlet pressure setpoint actual reading bar psi regulator unloader",
    "differential pressure oil separator air filter element restriction",
    # Temperature
    "element outlet temperature thermometer sensor actual limit shutdown",
    "coolant temperature intercooler aftercooler oil cooler air cooler",
    # Flow / load
    "capacity flow rate FAD free air delivery unload load duty cycle",
    "inlet valve throttle modulation blow-off control",
    # Drive
    "motor current power consumption speed rpm drive belt coupling",
    # Dryer (Full-feature)
    "dryer dewpoint refrigerant pressure evaporator condenser bypass",
]

process_queries = [
    "start sequence load unload idle automatic standby",
    "shutdown stop normal emergency cooldown blowdown",
    "pressure build regulation setpoint band load unload switch",
    "oil circuit lubrication injection separation scavenging",
    "cooling circuit air cooled water cooled thermostatic valve bypass",
    "condensate drain automatic timed solenoid separator drain",
    "air dryer cycle refrigerant compressor fan dewpoint control",
    "filter service replacement interval restriction indicator",
]

# Phase 3 — Simulation queries (topic-tagged tuples)
simulation_queries = [
    ("operating_context",
     "operating hours run time load ratio duty cycle air demand site conditions ambient temperature altitude"),
    ("degradation_maintenance_and_failures",
     "oil separator differential pressure air filter restriction oil change interval cooler fouling belt tension coupling wear Elektronikon alarm fault code service"),
    ("process_and_ranges",
     "pressure range temperature limit flow capacity FAD setpoint band tolerance specification operating envelope"),
]

# Phase 4 — Twin comprehension queries
twin_comprehension_queries = [
    # --- Compressor overall ---
    "compressor overview description purpose rotary screw oil-injected air system",
    "system architecture components subsystems assemblies drive train",

    # --- Compression element ---
    "screw element rotor male female lobe profile oil injection compression ratio",
    "element outlet temperature limit thermostatic bypass oil temperature control",

    # --- Oil system ---
    "oil circuit injection separation scavenging oil separator tank sump",
    "oil filter element bypass valve oil cooler thermostatic valve",
    "oil specification grade change interval top-up capacity level check",
    "oil separator differential pressure service replacement indicator",

    # --- Air system ---
    "inlet filter element air intake restriction service interval replacement",
    "inlet valve unloader throttle modulation capacity control",
    "minimum pressure check valve outlet non-return blowdown",
    "aftercooler moisture separator condensate drain automatic",

    # --- Cooling system ---
    "cooler air cooled fan motor drive belt tension adjustment",
    "thermostatic valve bypass temperature regulation coolant flow",
    "cooler fouling cleaning inspection differential temperature",

    # --- Drive system ---
    "motor starter direct star delta frequency drive power consumption current",
    "belt drive V-belt tension adjustment pulley alignment wear replacement",
    "coupling flexible element alignment torque drive inspection",

    # --- Pressure regulation / control ---
    "pressure setpoint band load unload regulator pilot valve adjustment",
    "unloader valve blow-off idle stop start pressure switch",
    "safety valve relief pressure setting inspection test",

    # --- Air dryer (Full-feature) ---
    "dryer refrigerant circuit compressor condenser evaporator heat exchanger",
    "dewpoint sensor measurement bypass valve dryer fault alarm",
    "condensate drain dryer separator automatic timed solenoid",

    # --- Elektronikon controller ---
    "Elektronikon controller display setpoint parameter menu navigation",
    "sensor input analog temperature pressure 4-20mA PT100 measurement",
    "alarm warning shutdown condition code description cause remedy",
    "service timer interval reset maintenance schedule programmed",
    "communication Modbus remote monitoring analog output relay",

    # --- Faults / maintenance ---
    "fault alarm error warning condition description trigger cause action",
    "maintenance service interval procedure inspection replacement torque",
    "specifications limits operating conditions performance parameters",
    "lubrication greasing bearing motor fan shaft interval quantity",
]

# Phase 7 -— State estimation and RUL prediction parameters
measurement_noise = {
    'outlet_pressure':          0.05,   # bar
    'element_outlet_temp':      1.0,    # degC
    'oil_separator_dp':         0.02,   # bar
    'air_filter_dp':            0.005,  # bar
    'motor_current':            0.5,    # A
    'dewpoint':                 1.0,    # degC (Full-feature only)
    'coolant_temp':             1.0,    # degC
}

process_noise_default = 1e-4  # applied to all states