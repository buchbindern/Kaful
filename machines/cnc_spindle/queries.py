manuals = [
    {"filename": "nsk_bearing_selection_mounting.pdf",   "source_type": "oem_manual"},
    {"filename": "mtconnect_part1.pdf",                  "source_type": "interface_standard"},
    {"filename": "mtconnect_part2.pdf",                  "source_type": "interface_standard"},
    {"filename": "iso_10816_vibration_severity.pdf",     "source_type": "condition_monitoring"},
    {"filename": "fanuc_ac_spindle_motor.pdf",           "source_type": "oem_manual"},
]

domain_context = """
CNC spindle unit used in high-precision machining operations.
Key failure consequences:
- Bearing failure: unplanned downtime, scrap parts, potential workpiece/tool damage
- Thermal growth: dimensional inaccuracy, out-of-tolerance parts
- Imbalance/runout: surface finish degradation, tool wear acceleration
Primary degradation story: bearing wear accumulating through load cycles,
accelerated by inadequate lubrication, thermal stress, and imbalance.
"""

manual_exclusions = []
manual_inclusions = []

# Phase 2 — Schema extraction queries
schema_queries = [
    "spindle speed RPM load current power torque cutting force",
    "vibration acceleration amplitude frequency bearing temperature thermal",
    "tool change cycle count ATC position tool number offset",
    "coolant flow pressure temperature lubrication oil level",
]

process_queries = [
    "spindle run-up acceleration deceleration ramp time orientation",
    "cutting operation feed rate depth of cut material removal rate",
    "tool change sequence clamp unclamp drawbar force retention knob",
    "thermal compensation growth offset axis correction temperature gradient",
]

# Phase 3 — Simulation queries (topic-tagged tuples)
simulation_queries = [
    ("operating_context",
     "operating hours shifts per day duty cycle spindle utilization workpiece material cutting conditions"),

    ("degradation_maintenance_and_failures",
     "bearing wear fatigue spalling preload loss lubrication interval grease replacement vibration threshold fault code alarm"),

    ("process_and_ranges",
     "speed range RPM torque curve power rating thermal limits runout tolerance vibration limits out of spec alarm"),
]

# Phase 4 — Twin comprehension queries
twin_comprehension_queries = [
    "spindle overview description purpose type configuration",
    "bearing arrangement preload front rear angular contact",
    "motor rotor stator windings cooling thermal management",
    "drawbar tool clamping unclamping retention force mechanism",
    "lubrication grease oil mist injection system",
    "vibration sensor accelerometer position location measurement",
    "thermal sensor temperature measurement location compensation",
    "fault alarm error code warning condition trigger",
    "maintenance service interval procedure inspection replacement",
    "specifications limits operating conditions parameters performance",
]