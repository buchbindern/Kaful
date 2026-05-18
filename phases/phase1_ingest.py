"""
phases/phase1_ingest.py
-----------------------
Phase 1: Ingest and index the machine manual into ChromaDB.

Run directly:
    python phases/phase1_ingest.py eversys_coffee_machine

Or import and call from run.py:
    from phases.phase1_ingest import run
    run(cfg)
"""

import sys
from pathlib import Path

# Allow running directly from phases/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_machine_config, ensure_dirs, DEFAULT_MODEL
from utils.rag import ManualRAG

from dotenv import load_dotenv
load_dotenv()


def run(cfg: dict, force: bool = False) -> ManualRAG:
    """
    Run phase 1 — ingest the manual and index into ChromaDB.

    Args:
        cfg:   result of get_machine_config()
        force: re-index even if collection already exists

    Returns:
        ManualRAG instance ready for querying in phase 2
    """
    print(f"\n{'='*52}")
    print(f"Phase 1 — Ingest: {cfg['machine_id']}")
    print(f"{'='*52}")

    ensure_dirs(cfg)

    rag = ManualRAG(cfg, model=DEFAULT_MODEL)

    if rag.collection_exists() and not force:
        print(f"✓ Already indexed — skipping.")
        print(f"  Run with force=True to re-index.")
    else:
        rag.index_manual(force=force)

    # Verify
    collection = rag.chroma_client.get_collection(name=cfg["collection_name"])
    print(f"\n✓ Collection '{cfg['collection_name']}' has {collection.count()} chunks.")

    return rag


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phases/phase1_ingest.py <machine_id> [--force]")
        print("Example: python phases/phase1_ingest.py eversys_coffee_machine")
        sys.exit(1)

    machine_id = sys.argv[1]
    force      = "--force" in sys.argv

    cfg = get_machine_config(machine_id)
    run(cfg, force=force)