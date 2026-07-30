"""
run_data.py
-----------
Main entry point for the data-first digital twin pipeline.

Usage:
    python run_data.py phm2010 c1
    python run_data.py phm2010 c1 --from phase3
    python run_data.py phm2010 c1 --only phase1
    python run_data.py phm2010 c1 --only phase1 --force
    python run_data.py mu_tcm exp_001 --from phase5

Phases:
    phase1 — ingest raw data → normalized cuts.parquet + wear.parquet
    phase2 — component mapping (columns → physical components + ProgPy roles)
    phase3 — model parameterization (fit wear model, estimate Q and R)
    phase4 — ProgPy codegen (state equation + obs model + composite)
    phase5 — state estimation (particle filter with real observation updates)
    phase6 — RUL prediction (Monte Carlo forward propagation)
    phase7 — validation (metrics + plots against ground truth)

Each phase saves all outputs to disk and skips steps already done.
To rerun a phase from scratch, delete its output directory or use --force.

Adding a new dataset:
    1. Create datasets/<dataset_id>/raw/  and drop data there
    2. Write datasets/<dataset_id>/dataset_meta.py  (optional but recommended)
    3. Write phases/phase1_ingest/<dataset_id>.py   (the only dataset-specific code)
    4. Run: python run_data.py <dataset_id> <record_id>
"""

import sys
import argparse
from pathlib import Path

# ── Path bootstrap ─────────────────────────────────────────────────────────────
# Ensures `data_pipeline/` is on sys.path regardless of where the script is
# invoked from. Supports all three invocation styles:
#   python data_pipeline/run_data.py phm2010 c1          (from Kaful root)
#   python run_data.py phm2010 c1                         (from data_pipeline/)
#   python -m data_pipeline.run_data phm2010 c1           (as module)
_HERE = Path(__file__).resolve().parent   # always data_pipeline/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
load_dotenv()

from config_data import get_record_config, ensure_dirs
from state_data  import PipelineState


PHASE_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Data-first digital twin pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_data.py phm2010 c1
  python run_data.py phm2010 c1 --from phase3
  python run_data.py phm2010 c1 --only phase1 --force
  python run_data.py mu_tcm exp_001
        """,
    )
    parser.add_argument("dataset_id", help="Dataset ID (folder name under datasets/)")
    parser.add_argument("record_id",  help="Record ID within the dataset (e.g. c1, exp_001)")
    parser.add_argument(
        "--from",  dest="from_phase",  default=None,
        help="Start from this phase (e.g. phase3) — runs this phase and all after",
    )
    parser.add_argument(
        "--only",  dest="only_phase",  default=None,
        help="Run only this phase (e.g. phase2)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-run of the targeted phase(s) even if outputs exist",
    )
    return parser.parse_args()


def should_run(phase: str, from_phase: str, only_phase: str) -> bool:
    if only_phase:
        return phase == only_phase
    if from_phase:
        return PHASE_ORDER.index(phase) >= PHASE_ORDER.index(from_phase)
    return True


def main():
    args = parse_args()

    dataset_id = args.dataset_id
    record_id  = args.record_id
    cfg        = get_record_config(dataset_id, record_id)

    ensure_dirs(cfg)
    state = PipelineState(cfg)

    print(f"\n{'='*60}")
    print(f"Data-first Pipeline — {dataset_id} / {record_id}")
    print(f"{'='*60}")

    if cfg["meta"] is not None:
        print(f"  ✓ dataset_meta.py loaded")
    else:
        print(f"  · No dataset_meta.py — phases will infer from data")

    # ── Phase 1 — Ingest ──────────────────────────────────────────────────────
    if should_run("phase1", args.from_phase, args.only_phase):
        from phases.phase1_ingest import run as phase1
        phase1(cfg, force=args.force)
        state.mark_done("phase1")

    # ── Phase 2 — Component mapping ───────────────────────────────────────────
    if should_run("phase2", args.from_phase, args.only_phase):
        from phases.phase2_component_map import run as phase2

        _ensure_phase("phase1", cfg, state)
        component_map = phase2(cfg, force=args.force)
        state.mark_done("phase2")

    # ── Phase 3 — Parameterization ────────────────────────────────────────────
    if should_run("phase3", args.from_phase, args.only_phase):
        from phases.phase3_parameterize import run as phase3

        _ensure_phase("phase1", cfg, state)
        model_params = phase3(cfg, force=args.force)
        state.mark_done("phase3")

    # ── Phase 4 — ProgPy codegen ──────────────────────────────────────────────
    if should_run("phase4", args.from_phase, args.only_phase):
        from phases.phase4_codegen import run as phase4

        _ensure_phase("phase2", cfg, state)
        _ensure_phase("phase3", cfg, state)
        phase4(cfg, force=args.force)
        state.mark_done("phase4")

    # ── Phase 5 — State estimation ────────────────────────────────────────────
    if should_run("phase5", args.from_phase, args.only_phase):
        from phases.phase5_estimate import run as phase5

        _ensure_phase("phase4", cfg, state)
        phase5(cfg, force=args.force)
        state.mark_done("phase5")

    # ── Phase 6 — RUL prediction ──────────────────────────────────────────────
    if should_run("phase6", args.from_phase, args.only_phase):
        from phases.phase6_rul import run as phase6

        _ensure_phase("phase5", cfg, state)
        phase6(cfg, force=args.force)
        state.mark_done("phase6")

    # ── Phase 7 — Validation ──────────────────────────────────────────────────
    if should_run("phase7", args.from_phase, args.only_phase):
        from phases.phase7_validate import run as phase7

        _ensure_phase("phase6", cfg, state)
        phase7(cfg, force=args.force)
        state.mark_done("phase7")

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete — {dataset_id} / {record_id}")
    print(f"{'='*60}")
    print(f"\nPhase summary:")
    print(state.summary())
    print()


def _ensure_phase(required_phase: str, cfg: dict, state: PipelineState) -> None:
    """
    Check that a required upstream phase has been run.
    If not, run it now with default settings.

    This matches the doc pipeline's lazy-load pattern: if you jump into
    --from phase5, phase1-4 outputs must already exist on disk. If they
    don't, we run them rather than crashing.
    """
    if state.is_done(required_phase):
        return

    # Check if outputs exist even if state file doesn't know about them
    # (e.g. user deleted state.json but kept outputs)
    output_exists = _check_output_exists(required_phase, cfg)
    if output_exists:
        state.mark_done(required_phase)
        return

    print(f"\n  ↳ {required_phase} not yet run — running it now...")

    if required_phase == "phase1":
        from phases.phase1_ingest import run as phase
        phase(cfg)
    elif required_phase == "phase2":
        from phases.phase2_component_map import run as phase
        phase(cfg)
    elif required_phase == "phase3":
        from phases.phase3_parameterize import run as phase
        phase(cfg)
    elif required_phase == "phase4":
        from phases.phase4_codegen import run as phase
        phase(cfg)
    elif required_phase == "phase5":
        from phases.phase5_estimate import run as phase
        phase(cfg)
    elif required_phase == "phase6":
        from phases.phase6_rul import run as phase
        phase(cfg)

    state.mark_done(required_phase)


def _check_output_exists(phase: str, cfg: dict) -> bool:
    """Check if a phase's primary output file exists on disk."""
    sentinel = {
        "phase1": cfg["cuts_path"],
        "phase2": cfg["step_d_progpy_path"],
        "phase3": cfg["model_params_path"],
        "phase4": cfg["step_c_composite_path"],
        "phase5": cfg["state_estimates_path"],
        "phase6": cfg["rul_predictions_path"],
        "phase7": cfg["validation_report_path"],
    }
    path = sentinel.get(phase)
    return path is not None and path.exists()


if __name__ == "__main__":
    main()