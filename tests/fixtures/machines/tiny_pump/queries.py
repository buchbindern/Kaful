# tiny_pump: fictional 2-component fixture machine for the offline harness (M2).
# Any change here changes prompts, so recorded fixtures must be re-recorded.

manuals = [
    {"filename": "tp200_manual.pdf", "source_type": "oem_manual"},
]

domain_context = None

schema_queries = [
    "monitoring channels sensors pressure flow current temperature vibration units range",
    "rated operating conditions flow inlet pressure discharge pressure",
]

process_queries = [
    "flow path inlet filter booster pump discharge",
    "differential pressure filter cartridge clogging",
]

simulation_queries = [
    ("operating_context",
     "duty operating hours per day rated flow water temperature"),

    ("degradation_maintenance_and_failures",
     "wear clogging service interval replace inspect life alarm warning"),

    ("process_and_ranges",
     "permitted range limits discharge pressure inlet pressure motor current vibration"),
]

twin_comprehension_queries = [
    "overview components flow path",
    "inlet cartridge filter differential pressure limits service",
    "booster pump impeller wear limits service troubleshooting",
]

manual_exclusions = []
manual_inclusions = []
