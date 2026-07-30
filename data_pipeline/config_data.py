"""
config_data.py
--------------
Global configuration for the data-first digital twin pipeline.

Mirrors config.py from the doc pipeline — same patterns, different paths.

Usage:
    from config_data import get_record_config, ensure_dirs
    cfg = get_record_config("phm2010", "c1")
"""

import importlib
import importlib.util
from pathlib import Path

# ── Repo root ─────────────────────────────────────────────────────────────────
# config_data.py lives at:  kaful/data_pipeline/config_data.py
# PIPELINE_DIR  →           kaful/data_pipeline/
# ROOT_DIR      →           kaful/               (shared with doc pipeline)
PIPELINE_DIR = Path(__file__).parent
ROOT_DIR     = PIPELINE_DIR.parent
DATASETS_DIR = PIPELINE_DIR / "datasets"

# ── Models ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "claude-sonnet-4-6"
CODEGEN_MODEL = "claude-opus-4-6"

# ── Phase 5 — State estimation ────────────────────────────────────────────────
PARTICLE_COUNT     = 500    # number of particles in the filter
RESAMPLE_THRESHOLD = 0.5    # effective sample size ratio below which we resample

# ── Phase 6 — RUL prediction ──────────────────────────────────────────────────
MC_SAMPLES         = 200    # Monte Carlo forward trajectories per cut
MAX_PREDICTION_CUTS = 500   # hard cap on forward horizon


def get_record_config(dataset_id: str, record_id: str) -> dict:
    """
    Build all paths for a given dataset + record.
    All phases use this — never hardcode paths in a phase file.

    Args:
        dataset_id: folder name under datasets/, e.g. "phm2010"
        record_id:  specific record within that dataset, e.g. "c1"

    Returns:
        dict with all path and identity values for this run.
    """
    dataset_dir = DATASETS_DIR / dataset_id
    record_dir  = dataset_dir  / "outputs" / record_id

    # ── Load optional dataset metadata ────────────────────────────────────────
    meta = _load_meta(dataset_dir)

    # ── Load optional LLM queries config ──────────────────────────────────────
    queries_cfg = _load_queries(dataset_dir)

    return {
        # Identity
        "dataset_id":   dataset_id,
        "record_id":    record_id,
        "dataset_dir":  dataset_dir,
        "record_dir":   record_dir,
        "meta":         meta,           # None if dataset_meta.py absent

        # Phase 1 — Ingest
        "raw_dir":       dataset_dir / "raw",
        "ingest_dir":    record_dir  / "ingest",
        "cuts_path":     record_dir  / "ingest" / "cuts.parquet",
        "wear_path":     record_dir  / "ingest" / "wear.parquet",
        "ingest_meta_path": record_dir / "ingest" / "ingest_meta.json",

        # Phase 2 — Component mapping
        "component_map_dir":  record_dir / "component_map",
        "step_a_schema_path": record_dir / "component_map" / "step_a_signal_schema.json",
        "step_b_map_path":    record_dir / "component_map" / "step_b_component_map.json",
        "step_c_pathway_path":record_dir / "component_map" / "step_c_degradation_pathway.json",
        "step_d_progpy_path": record_dir / "component_map" / "step_d_progpy_roles.json",

        # Phase 3 — Parameterization
        "params_dir":         record_dir / "params",
        "step_a_wear_fit_path":  record_dir / "params" / "step_a_wear_fit.json",
        "step_b_model_form_path":record_dir / "params" / "step_b_model_form.json",
        "step_c_noise_q_path":   record_dir / "params" / "step_c_noise_q.json",
        "step_d_noise_r_path":   record_dir / "params" / "step_d_noise_r.json",
        "model_params_path":     record_dir / "params" / "model_params.json",

        # Phase 4 — ProgPy codegen
        "codegen_dir":           record_dir / "codegen",
        "step_a_component_path": record_dir / "codegen" / "step_a_component.py",
        "step_b_obs_model_path": record_dir / "codegen" / "step_b_obs_model.py",
        "step_c_composite_path": record_dir / "codegen" / "composite_model.py",
        "step_d_validation_path":record_dir / "codegen" / "step_d_validation.json",

        # Phase 5 — State estimation
        "estimate_dir":          record_dir / "estimate",
        "state_estimates_path":  record_dir / "estimate" / "state_estimates.parquet",
        "estimate_plot_path":    record_dir / "estimate" / "state_vs_wear.png",

        # Phase 6 — RUL prediction
        "rul_dir":               record_dir / "rul",
        "rul_predictions_path":  record_dir / "rul" / "rul_predictions.parquet",
        "rul_plot_path":         record_dir / "rul" / "rul_vs_actual.png",

        # Phase 7 — Validation
        "validation_dir":        record_dir / "validation",
        "validation_report_path":record_dir / "validation" / "validation_report.json",
        "validation_plot_path":  record_dir / "validation" / "validation_summary.png",

        # From queries.py (optional overrides)
        "queries": queries_cfg,
    }


def ensure_dirs(cfg: dict) -> None:
    """
    Create all output directories for a record if they don't exist.
    Call once at the start of a pipeline run.
    """
    dirs = [
        cfg["ingest_dir"],
        cfg["component_map_dir"],
        cfg["params_dir"],
        cfg["codegen_dir"],
        cfg["estimate_dir"],
        cfg["rul_dir"],
        cfg["validation_dir"],
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def _load_meta(dataset_dir: Path):
    """
    Load dataset_meta.py from the dataset directory if it exists.
    Returns the module object, or None if absent.
    """
    meta_path = dataset_dir / "dataset_meta.py"
    if not meta_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("dataset_meta", meta_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_queries(dataset_dir: Path) -> dict:
    """
    Load queries.py from the dataset directory if it exists.
    Returns vars(module) dict, or {} if absent.
    """
    queries_path = dataset_dir / "queries.py"
    if not queries_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("queries", queries_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod)