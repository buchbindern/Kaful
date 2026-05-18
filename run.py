"""
run.py
------
Main entry point for the digital twin pipeline.

Usage:
    python run.py eversys_coffee_machine
    python run.py eversys_coffee_machine --from phase3
    python run.py eversys_coffee_machine --only phase2
    python run.py eversys_coffee_machine --only phase1 --force

Phases:
    phase1 — ingest manual into RAG
    phase2 — generate event schema
    phase3 — simulate data
    phase4 — build digital twin machine model
    phase5 — triage components
    phase6 — generate ProgPy component code and composite model
    phase7 — state estimation and RUL prediction

Each phase saves all outputs to disk and skips steps already done.
To rerun a phase from scratch, delete its output files.
"""

import sys
import argparse
from dotenv import load_dotenv
load_dotenv()

from config import get_machine_config, ensure_dirs
from utils.rag import ManualRAG


PHASE_ORDER = ["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7"]


def parse_args():
    parser = argparse.ArgumentParser(description="Digital twin pipeline")
    parser.add_argument("machine_id", help="Machine ID (folder name under machines/)")
    parser.add_argument("--from",  dest="from_phase",  default=None,
                        help="Start from this phase (e.g. phase3)")
    parser.add_argument("--only",  dest="only_phase",  default=None,
                        help="Run only this phase (e.g. phase2)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if outputs exist (phase1 only)")
    return parser.parse_args()


def should_run(phase: str, from_phase: str, only_phase: str) -> bool:
    if only_phase:
        return phase == only_phase
    if from_phase:
        return PHASE_ORDER.index(phase) >= PHASE_ORDER.index(from_phase)
    return True


def main():
    args = parse_args()

    machine_id = args.machine_id
    cfg        = get_machine_config(machine_id)
    ensure_dirs(cfg)

    print(f"\n{'='*60}")
    print(f"Digital Twin Pipeline — {machine_id}")
    print(f"{'='*60}")

    # ── Shared state ──────────────────────────────────────────────────────────
    # Each phase lazily loads what it needs from disk if not already in memory
    rag             = ManualRAG(cfg)
    schema          = None
    machine_model   = None
    full_components = None

    # ── Phase 1 — Ingest ──────────────────────────────────────────────────────
    if should_run("phase1", args.from_phase, args.only_phase):
        from phases.phase1_ingest import run as phase1
        phase1(cfg, force=args.force)

    # ── Phase 2 — Schema ──────────────────────────────────────────────────────
    if should_run("phase2", args.from_phase, args.only_phase):
        from phases.phase2_schema import run as phase2
        schema = phase2(cfg, rag)

    # ── Phase 3 — Simulate ────────────────────────────────────────────────────
    if should_run("phase3", args.from_phase, args.only_phase):
        from phases.phase3_simulate import run as phase3

        if schema is None:
            from phases.phase2_schema import run as phase2
            schema = phase2(cfg, rag)

        phase3(cfg, rag, schema)

    # ── Phase 4 — Digital Twin ────────────────────────────────────────────────
    if should_run("phase4", args.from_phase, args.only_phase):
        from phases.phase4_twin import run as phase4

        if schema is None:
            from phases.phase2_schema import run as phase2
            schema = phase2(cfg, rag)

        machine_model, issues, warnings = phase4(cfg, rag)

        if issues:
            print(f"\n⚠ Phase 4 has {len(issues)} issues — review before continuing")

    # ── Phase 5 — Triage ──────────────────────────────────────────────────────
    if should_run("phase5", args.from_phase, args.only_phase):
        from phases.phase5_triage import run as phase5

        if schema is None:
            from phases.phase2_schema import run as phase2
            schema = phase2(cfg, rag)

        if machine_model is None:
            from phases.phase4_twin import run as phase4
            machine_model, _, _ = phase4(cfg, rag)

        full_components = phase5(cfg, schema, machine_model)

    # ── Phase 6 — Codegen ─────────────────────────────────────────────────────
    if should_run("phase6", args.from_phase, args.only_phase):
        from phases.phase6_codegen import run as phase6

        if schema is None:
            from phases.phase2_schema import run as phase2
            schema = phase2(cfg, rag)

        if machine_model is None:
            from phases.phase4_twin import run as phase4
            machine_model, _, _ = phase4(cfg, rag)

        if full_components is None:
            from phases.phase5_triage import run as phase5
            full_components = phase5(cfg, schema, machine_model)

        phase6(cfg, full_components, machine_model)

    # ── Phase 7 — Estimate ────────────────────────────────────────────────────
    if should_run("phase7", args.from_phase, args.only_phase):
        from phases.phase7_estimate import run as phase7
        phase7(cfg)

    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete — {machine_id}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()