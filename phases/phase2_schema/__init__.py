"""
phases/phase2_schema/__init__.py
---------------------------------
Phase 2: Generate the event schema from the machine manual.

Steps:
    a — retrieve relevant chunks from RAG using machine-specific queries
    b — infer extraction context (event type, measurement categories)
    c — extract candidate fields N times
    d — normalize and dedupe fields within each run
    e — merge duplicate fields across runs
    f — build and save final schema
    g — generate and test DDL (coming next)

Each step saves its output to disk and skips if already done.
To re-run a step, delete its output file and re-run.

Run directly:
    python -m phases.phase2_schema eversys_coffee_machine

Or call from run.py:
    from phases.phase2_schema import run
    schema = run(cfg, rag)
"""

import sys
from pathlib import Path

from phases.phase2_schema.step_a_retrieve import run as step_a
from phases.phase2_schema.step_b_context import run as step_b
from phases.phase2_schema.step_c_extract import run as step_c
from phases.phase2_schema.step_d_normalize import run as step_d
from phases.phase2_schema.step_e_merge import run as step_e
from phases.phase2_schema.step_f_finalize import run as step_f


def run(cfg: dict, rag) -> list[dict]:
    """
    Run all phase 2 steps in order.

    Args:
        cfg: result of get_machine_config()
        rag: ManualRAG instance (already indexed from phase 1)

    Returns:
        final schema as a list of field dicts
    """
    print(f"\n{'='*52}")
    print(f"Phase 2 — Schema: {cfg['machine_id']}")
    print(f"{'='*52}")

    chunks, context = step_a(cfg, rag)
    context_info    = step_b(cfg, context)
    runs            = step_c(cfg, context, context_info)
    normalized      = step_d(cfg, runs)
    merged          = step_e(cfg, normalized)
    schema          = step_f(cfg, merged)

    print(f"\n✓ Phase 2 complete — {len(schema)} fields in final schema.")
    return schema


if __name__ == "__main__":
    # Allow running directly: python -m phases.phase2_schema eversys_coffee_machine
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    # Allow running directly from phases/ directory
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from config import get_machine_config, ensure_dirs
    from utils.rag import ManualRAG

    if len(sys.argv) < 2:
        print("Usage: python -m phases.phase2_schema <machine_id>")
        print("Example: python -m phases.phase2_schema eversys_coffee_machine")
        sys.exit(1)

    machine_id = sys.argv[1]
    cfg        = get_machine_config(machine_id)
    ensure_dirs(cfg)

    rag    = ManualRAG(cfg)
    schema = run(cfg, rag)