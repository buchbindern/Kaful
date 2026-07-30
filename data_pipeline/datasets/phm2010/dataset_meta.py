"""
datasets/phm2010/dataset_meta.py
----------------------------------
Optional but first-class metadata for the PHM Society 2010 Data Challenge.

Source:
    https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge
    "Each training record contains one wear file that lists wear after each cut
    in 10^-3 mm, and a folder with approximately 300 individual data acquisition
    files (one per cut). Data files are in CSV format with seven columns."

When this file is present, phases consume it directly instead of inferring.
If it is absent, phases fall back to dataset defaults (hardcoded in the loader)
or LLM inference (phase 2).
"""

# ── Signal columns ─────────────────────────────────────────────────────────────
# Exact mapping from the official dataset description.
# "index" matches the 0-based CSV column position in each per-cut file.

SIGNAL_COLUMNS = [
    {"index": 0, "name": "force_x",    "unit": "N", "type": "force",            "axis": "x"},
    {"index": 1, "name": "force_y",    "unit": "N", "type": "force",            "axis": "y"},
    {"index": 2, "name": "force_z",    "unit": "N", "type": "force",            "axis": "z"},
    {"index": 3, "name": "vibration_x","unit": "g", "type": "vibration",        "axis": "x"},
    {"index": 4, "name": "vibration_y","unit": "g", "type": "vibration",        "axis": "y"},
    {"index": 5, "name": "vibration_z","unit": "g", "type": "vibration",        "axis": "z"},
    {"index": 6, "name": "ae_rms",     "unit": "V", "type": "acoustic_emission","axis": None},
]

# ── Wear label format ──────────────────────────────────────────────────────────
WEAR_UNIT  = "1e-3 mm"   # raw values in the wear CSV are in 10^-3 mm
WEAR_SCALE = 1e-3        # multiply raw wear values by this to get mm

# ── Cutting parameters ─────────────────────────────────────────────────────────
# Used by phase 3 to inform model form selection (Taylor tool life, power-law).
CUTTING_PARAMS = {
    "spindle_speed_rpm":  10400,
    "feed_rate_mm_min":   1555,
    "depth_radial_mm":    0.125,   # Y depth of cut
    "depth_axial_mm":     0.200,   # Z depth of cut
    "sampling_rate_hz":   50_000,  # 50 KHz per channel
}

# ── Machine and tool context ───────────────────────────────────────────────────
# Used by phase 2 to enrich LLM component mapping prompts.
MACHINE_TYPE       = "CNC high-speed milling machine"
WORKPIECE_MATERIAL = "stainless steel (HRC52)"
TOOL_MATERIAL      = "tungsten carbide"
TOOL_GEOMETRY      = "2-flute ball nose cutter"
PROCESS_TYPE       = "dry milling"

# ── Wear threshold ─────────────────────────────────────────────────────────────
# ISO 8688 flank wear failure criterion for milling.
# This is the RUL target — the cut at which wear_mm crosses this value is "end of life".
WEAR_THRESHOLD_MM = 0.2

# ── Record registry ────────────────────────────────────────────────────────────
# Tells phase 1 which records exist and which have ground-truth wear labels.
# c1, c4, c6 are the labeled training records released by PHM Society.
# c2, c3, c5 were held out for the competition and have no wear labels.
RECORDS = {
    "c1": {"labeled": True,  "notes": "Primary training record"},
    "c4": {"labeled": True,  "notes": "Primary training record"},
    "c6": {"labeled": True,  "notes": "Validation record"},
    "c2": {"labeled": False, "notes": "Competition holdout — no wear labels"},
    "c3": {"labeled": False, "notes": "Competition holdout — no wear labels"},
    "c5": {"labeled": False, "notes": "Competition holdout — no wear labels"},
}