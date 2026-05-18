manuals = [
    {"filename": "eversys_legacy_user_manual.pdf", "source_type": "oem_manual"},
]

# Optional domain context — injected into simulator and triage prompts
# Leave as None if not needed
domain_context = None

schema_queries = [
    "product settings recipe parameters drink settings beverage quantity milk quantity milk temperature",
    "product and keys menu water quantity milk quantity hot water quantity learn quantity",
    "milk sequence popup milk quantity milk temperature foam setting beverage parameters",
]

process_queries = [
    "bean grinder menu grind size coffee quantity grinder adjustment",
    "coffee extraction recipe brew quantity shot time water quantity flowmeter",
    "dispense beverage product configuration recipe configuration",
]

# Phase 3 — Simulation queries (topic-tagged tuples)
simulation_queries = [
    ("operating_context",
     "operating environment usage intensity hourly volume daily volume product mix workflow"),

    ("degradation_maintenance_and_failures",
     "wear degradation maintenance cleaning descaling service interval failure warning error codes"),

    ("process_and_ranges",
     "process sequence operation steps timing temperature pressure flow ranges limits out of spec"),
]

twin_comprehension_queries = [
    "machine overview description purpose type",
    "components subsystems assemblies architecture structure",
    "process sequence operation steps cycle phases",
    "flow path transfer connection inlet outlet",
    "specifications limits operating conditions parameters",
    "fault error warning alarm code",
    "maintenance service cleaning interval procedure",
]

manual_exclusions = ["touch_screen"]
manual_inclusions = []  # force these into full_component regardless of triage