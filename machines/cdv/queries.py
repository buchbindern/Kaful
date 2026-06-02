manuals = [
    {"filename": "m220_383_02_process_manual.pdf",   "source_type": "oem_manual"},
    {"filename": "m320_01_furnace.pdf",               "source_type": "oem_manual"},
    {"filename": "m330_01_gas_cabinet.pdf",           "source_type": "oem_manual"},
    {"filename": "m410_01_dpc.pdf",                   "source_type": "oem_manual"},
    {"filename": "olt-6036_h.pdf",                    "source_type": "oem_manual"},
    {"filename": "m210_01_operator.pdf",              "source_type": "oem_manual"},
    # Component datasheets
    {"filename": "mks_mfc_1179b.pdf",                 "source_type": "component_datasheet"},
    {"filename": "brooks_gf_series_mfc.pdf",          "source_type": "component_datasheet"},
    {"filename": "smc_zse30_pressure_switch.pdf",     "source_type": "component_datasheet"},
    {"filename": "edwards_e2m_vacuum_pump.pdf",       "source_type": "component_datasheet"},
    {"filename": "kanthal_fibrothal_heating.pdf",     "source_type": "component_datasheet"},
]

domain_context = """
Tempress Systems horizontal LPCVD (Low Pressure Chemical Vapor Deposition) furnace
used in semiconductor wafer fabrication.
Key deposited films: silicon nitride (Si3N4), polysilicon, TEOS oxide (SiO2).
Key failure consequences:
- Tube contamination: scrapped wafer batch, extended downtime for tube cleaning/bake
- Temperature non-uniformity: film thickness variation across wafer, out-of-spec deposition
- MFC drift or failure: incorrect gas ratios, wrong film stoichiometry, recipe deviation
- Vacuum pump failure: inability to reach process pressure, run abort
- Heating element failure: loss of thermal zone control, process abort
Primary degradation story: gradual drift in MFC calibration, heating element resistance
increase, vacuum pump throughput loss, and quartz tube devitrification accumulating
over hundreds of process runs and thermal cycles.
"""

manual_exclusions = ["boat_loader"]
manual_inclusions = []

# Phase 2 — Schema extraction queries
schema_queries = [
    # Original 4 — keep these, they worked
    "temperature setpoint zone thermocouple measurement actual process",
    "pressure process vacuum Torr mTorr gauge reading setpoint",
    "gas flow rate sccm MFC mass flow controller setpoint actual",
    "recipe step time duration ramp soak deposition rate",
    # Only add one new targeted query for missing channels
    "nitrogen N2 ammonia NH3 dichlorosilane DCS gas species flow channel",
]

process_queries = [
    "recipe load sequence wafer boat push pull cassette",
    "pump down evacuation base pressure leak rate crossover",
    "temperature ramp rate heat soak stabilize flat zone",
    "gas flow sequence purge stabilize deposit vent",
    "interlock abort status alarm condition trigger response",
    "vent atmospheric pressure nitrogen purge cooldown",
]

# Phase 3 — Simulation queries (topic-tagged tuples)
simulation_queries = [
    ("operating_context",
     "process run frequency wafers per run batch size throughput utilization tube capacity"),
    ("degradation_maintenance_and_failures",
     "tube cleaning devitrification MFC calibration drift pump maintenance heating element replacement thermocouple replacement alarm fault code"),
    ("process_and_ranges",
     "temperature range zone uniformity pressure range flow range deposition rate film thickness uniformity specification limit tolerance"),
]

# Phase 4 — Twin comprehension queries
twin_comprehension_queries = [
    # --- Furnace overall ---
    "furnace overview description purpose horizontal LPCVD tube system",
    "system architecture components subsystems assemblies configuration",

    # --- Thermal / heating ---
    "heating zone layout element arrangement flat zone length temperature profile",
    "thermocouple type location position control spike profile measurement",
    "SCR power controller heating element resistance winding zone",

    # --- Gas delivery ---
    "gas cabinet manifold MFC mass flow controller gas species silane ammonia",
    "gas flow sequence valve pneumatic solenoid purge stabilize deposit",
    "bubbler source precursor gas line pressure regulation",

    # --- Vacuum / pressure ---
    "vacuum pump rotary vane type capacity ultimate pressure throughput",
    "pressure gauge transducer Pirani capacitance manometer location",
    "pump down evacuation base pressure leak rate crossover valve",

    # --- Process tube ---
    "process tube quartz liner boat paddle wafer carrier material",
    "tube installation removal cleaning devitrification inspection",

    # --- Controllers ---
    "DPC digital process controller PID setpoint output channel recipe step",
    "DTC digital temperature controller zone ramp rate soak setpoint",

    # --- Recipe / process ---
    "recipe structure step type parameter ramp soak flow pressure time",
    "process sequence load push pull boat position wafer cassette",

    # --- Faults / maintenance ---
    "fault alarm error code warning condition description trigger",
    "maintenance service interval procedure inspection replacement cleaning",
    "specifications limits operating conditions parameters performance",
]

measurement_noise = {
    'zone1_temperature': 0.25,
    'zone2_temperature': 0.25,
    'zone3_temperature': 0.25,
    'process_pressure':  4.0,
    'sih4_flow_rate':    0.25,
    'n2_flow_rate':      0.01,
    'boat_position':     1.0,
}

process_noise_default = 1e-4  # applied to all states