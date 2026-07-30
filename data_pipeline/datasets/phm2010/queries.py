"""
datasets/phm2010/queries.py
----------------------------
Optional LLM prompt customizations for PHM 2010.

Phase 2 (component mapping) uses these to give the LLM additional context
about this specific dataset beyond what it can infer from column names alone.
If this file is absent, phase 2 uses its generic prompts.

These are not the prompts themselves — they're context strings and hints
that get injected into the phase 2 prompts at runtime.
"""

# ── Additional context for phase 2 component mapping ─────────────────────────
# Injected into the system prompt for the column → component mapping step.
DATASET_CONTEXT = """
This is the PHM Society 2010 Data Challenge dataset for CNC milling tool wear.

Setup:
- Machine: high-speed CNC milling center, 2-flute ball nose tungsten carbide cutter
- Workpiece: stainless steel (HRC52)
- Process: dry milling at 10400 RPM, 1555 mm/min feed rate
- External sensors: 3-axis dynamometer (force), 3 piezoelectric accelerometers (vibration),
  1 AE sensor (acoustic emission)
- All sensors are external (attached to workpiece/fixture), not internal CNC signals

Key degradation mechanism:
- Flank wear (VB) on the tool cutting edges
- Wear accumulates cut by cut due to abrasive contact with the workpiece
- ISO 8688 failure criterion: VB >= 0.2 mm
- Wear is non-reversible and monotonically increasing under normal conditions

Signal-to-physics mapping:
- Force signals (Fx, Fy, Fz): cutting force increases as tool dulls (more friction,
  less sharp edge). Z-axis force (thrust) is particularly sensitive to flank wear.
- Vibration signals (Ax, Ay, Az): vibration amplitude and frequency content change
  as tool geometry degrades. Kurtosis is especially diagnostic of chatter onset.
- AE-RMS: acoustic emission energy from chip formation and tool-workpiece contact.
  AE-RMS tends to increase with wear and is sensitive to micro-fractures.
"""

# ── Phase 2 hint: primary degradation state variable ─────────────────────────
# If provided, phase 2 skips the LLM inference step for state variable selection
# and uses this directly. Must match a column in wear.parquet.
PRIMARY_STATE_VARIABLE = "wear_mm"
PRIMARY_STATE_UNIT     = "mm"

# ── Phase 3 hint: preferred model form ───────────────────────────────────────
# Tells phase 3 which degradation model form to fit first.
# Options: "power_law", "linear", "exponential", "paris_law"
# If None, phase 3 uses LLM to select based on dataset context.
PREFERRED_MODEL_FORM = "power_law"   # VB(t) = C * t^n

# Override step_d filter feature selection
# The LLM tends to pick force_x/y over thrust (force_z) and drops vibration.
# These are the features backed by PHM 2010 literature for VB estimation.
FILTER_FEATURES = [
    "force_z_rms",       # thrust force — primary VB indicator
    "force_z_std",       # thrust variability — sensitive to wear onset
    "ae_rms_rms",        # acoustic emission — independent modality, early detection
    "vibration_z_rms",   # axial vibration — corroborates thrust channel
]