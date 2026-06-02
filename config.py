"""
config.py
---------
Global configuration for the digital twin pipeline.
Add settings here as each phase needs them.

Usage:
    from config import get_machine_config, DEFAULT_MODEL
    cfg = get_machine_config("eversys_coffee_machine")
"""

from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent
MACHINES_DIR = ROOT_DIR / "machines"

# ── Models ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "claude-sonnet-4-20250514"
CODEGEN_MODEL   = "claude-opus-4-6"
HAIKU_MODEL     = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL = "text-embedding-3-large"

# ── Framework RAGs ────────────────────────────────────────────────────────────
PROGPY_RAG_DIR = ROOT_DIR / "rag_frameworks" / "progpy" / "progpy_rag"
PROGPY_CHROMA_DIR = PROGPY_RAG_DIR / "data" / "chroma"

# Phase 7 — State estimation
NUM_PARTICLES  = 100
ESTIMATE_FREQ  = 50
MC_SAMPLES     = 100
PREDICTION_HORIZON = 50000

def get_machine_config(machine_id: str) -> dict:
    """
    Build all paths for a given machine.
    All phases use this — never hardcode paths in a phase file.

    Args:
        machine_id: folder name under machines/, e.g. "eversys_coffee_machine"

    Returns:
        dict with all path and identity values for this machine.
    """
    machine_dir = MACHINES_DIR / machine_id

    # Load machine-specific queries config
    queries_path = machine_dir / "queries.py"
    queries_cfg = {}
    if queries_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("queries", queries_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        queries_cfg = vars(mod)

    return {
        # Identity
        "machine_id":      machine_id,
        "machine_dir":     machine_dir,

        # Phase 1 — ingest
        "manual_dir":      machine_dir / "manual",
        "rag_dir":         machine_dir / "rag" / "vector_db",
        "chunks_dir":      machine_dir / "rag" / "chunks",
        "collection_name": f"{machine_id}_manual",

        # Phase 2 - schema
        "schema_dir":        machine_dir / "outputs" / "schema",
        "step_a_chunks": machine_dir / "outputs" / "schema" / "step_a_chunks.json",
        "step_b_context": machine_dir / "outputs" / "schema" / "step_b_context.json",
        "step_c_runs": machine_dir / "outputs" / "schema" / "step_c_runs.json",
        "step_d_normalized": machine_dir / "outputs" / "schema" / "step_d_normalized.json",
        "step_e_merged": machine_dir / "outputs" / "schema" / "step_e_merged.json",
        "final_schema_path": machine_dir / "outputs" / "schema" / "final_schema.json",
        "ddl_path": machine_dir / "outputs" / "schema" / "brewing_events.sql",

        # Phase 3 — Simulate
        "simulate_dir":      machine_dir / "outputs" / "simulate",
        "sim_step_a_context": machine_dir / "outputs" / "simulate" / "step_a_context.json",
        "sim_step_b_plan": machine_dir / "outputs" / "simulate" / "step_b_plan.json",
        "sim_step_c_code": machine_dir / "outputs" / "simulate" / "simulator.py",
        "sim_step_d_validation": machine_dir / "outputs" / "simulate" / "step_d_validation.json",
        "events_csv_path":       machine_dir / "outputs" / "simulate" / "events.csv",
        "maintenance_csv_path":  machine_dir / "outputs" / "simulate" / "maintenance_log.csv",

        # Phase 4 — Twin
        "twin_dir":               machine_dir / "outputs" / "twin",
        "twin_step_a_comprehension": machine_dir / "outputs" / "twin" / "step_a_comprehension.json",
        "twin_components_dir": machine_dir / "outputs" / "twin" / "components",
        "twin_step_c_model": machine_dir / "outputs" / "twin" / "step_c_machine_model.json",
        "twin_step_d_validation": machine_dir / "outputs" / "twin" / "step_d_validation.json",

        # Phase 5 — Triage
        "triage_dir":                  machine_dir / "outputs" / "triage",
        "triage_step_a_assignments":   machine_dir / "outputs" / "triage" / "step_a_field_assignments.json",
        "triage_step_b_triage":        machine_dir / "outputs" / "triage" / "step_b_triage.json",
        "triage_step_c_full_components": machine_dir / "outputs" / "triage" / "step_c_full_components.json",

        # Phase 6 — Codegen
        "codegen_dir":       machine_dir / "outputs" / "codegen",
        "codegen_specs_dir": machine_dir / "outputs" / "codegen" / "specs",
        "codegen_code_dir":  machine_dir / "outputs" / "codegen" / "components",
        "codegen_step_c_validation": machine_dir / "outputs" / "codegen" / "step_c_validation.json",
        "codegen_step_d_connections":     machine_dir / "outputs" / "codegen" / "step_d_connections.json",
        "codegen_step_d_external_inputs": machine_dir / "outputs" / "codegen" / "step_d_external_inputs.json",
        "codegen_composite_path":         machine_dir / "outputs" / "codegen" / "composite_model.py",
        "codegen_step_d_validation":      machine_dir / "outputs" / "codegen" / "step_d_validation.json",

        # Phase 7 — Estimate
        "estimate_dir": machine_dir / "outputs" / "estimate",
        # Phase 7 — noise config from queries.py
        "measurement_noise":    queries_cfg.get("measurement_noise", {}),
        "process_noise_default": queries_cfg.get("process_noise_default", 1e-4),
    }


def ensure_dirs(cfg: dict) -> None:
    """
    Create all directories for a machine if they don't exist.
    Call once at the start of a pipeline run.
    """
    dirs = [
        cfg["manual_dir"],
        cfg["rag_dir"],
        cfg["chunks_dir"],
        cfg["schema_dir"],
        cfg["twin_dir"],
        cfg["triage_dir"],
        cfg["codegen_dir"],
        cfg["estimate_dir"]
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)