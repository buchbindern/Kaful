"""
phases/phase1_ingest/__init__.py
---------------------------------
Phase 1: Ingest raw dataset files into a normalized internal format.

Steps:
    a — detect dataset format and validate raw directory
    b — load and segment signal files into cut indexed windows
    c — load wear labels and align to cut index
    d — extract per-cut feature vectors
    e — write normalized outputs: cuts.parquet + wear.parquet + ingest_meta.json

The only dataset-specific code lives in the loader files:
    phases/phase1_ingest/phm2010.py
    phases/phase1_ingest/mu_tcm.py
    phases/phase1_ingest/<new_dataset>.py   ← adding a new dataset = one new file here

All loaders must return the same interface so phases 2–7 never see raw data.
See normalize.py for the shared feature extraction and output writing.

Call from run_data.py:
    from phases.phase1_ingest import run
    run(cfg, force=False)

Or run directly:
    python -m phases.phase1_ingest phm2010 c1
"""

import sys
import importlib
from pathlib import Path


def run(cfg: dict, force: bool = False) -> None:
    """
    Run all phase 1 steps for the given dataset + record.

    Dispatches to the dataset-specific loader, which handles steps a–c
    (format detection, file loading, wear alignment). Then calls the
    shared normalize.py for steps d–e (feature extraction, output writing).

    Args:
        cfg:   result of get_record_config()
        force: if True, re-run even if outputs already exist
    """
    dataset_id = cfg["dataset_id"]
    record_id  = cfg["record_id"]

    print(f"\n{'='*52}")
    print(f"Phase 1 — Ingest: {dataset_id} / {record_id}")
    print(f"{'='*52}")

    # Skip if already done and not forced
    if not force and cfg["cuts_path"].exists() and cfg["wear_path"].exists():
        print(f"  ✓ phase1 already done — cuts.parquet and wear.parquet exist")
        print(f"    Use --force to re-run")
        return

    if force:
        print(f"  --force: re-running phase1")

    # Dispatch to dataset-specific loader
    loader = _load_dataset_loader(dataset_id)
    cuts_df, wear_df, raw_meta = loader.run(cfg)

    # Shared feature extraction + output writing (steps d–e)
    from phases.phase1_ingest.normalize import run as normalize
    normalize(cfg, cuts_df, wear_df, raw_meta)

    print(f"\n✓ Phase 1 complete — {dataset_id} / {record_id}")
    print(f"    cuts.parquet : {len(cuts_df)} cuts × {len(cuts_df.columns)} features")
    print(f"    wear.parquet : {len(wear_df)} wear measurements")


def _load_dataset_loader(dataset_id: str):
    """
    Import the dataset-specific loader module.

    Loader must be at: phases/phase1_ingest/<dataset_id>.py
    and must expose a run(cfg) function returning (cuts_df, wear_df, raw_meta).
    """
    module_path = Path(__file__).parent / f"{dataset_id}.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"No loader found for dataset '{dataset_id}'.\n"
            f"Expected: phases/phase1_ingest/{dataset_id}.py\n"
            f"Available loaders: {_list_available_loaders()}"
        )

    spec = importlib.util.spec_from_file_location(f"phase1_{dataset_id}", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _list_available_loaders() -> list[str]:
    """Return names of all dataset loader files in this directory."""
    here = Path(__file__).parent
    skip = {"__init__.py", "normalize.py"}
    return [p.stem for p in here.glob("*.py") if p.name not in skip]


# ── Allow running directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    # data_pipeline/ needs to be on sys.path
    _pipeline_dir = Path(__file__).resolve().parent.parent.parent  # data_pipeline/
    if str(_pipeline_dir) not in sys.path:
        sys.path.insert(0, str(_pipeline_dir))

    from config_data import get_record_config, ensure_dirs

    if len(sys.argv) < 3:
        print("Usage: python -m phases.phase1_ingest <dataset_id> <record_id> [--force]")
        print("Example: python -m phases.phase1_ingest phm2010 c1")
        sys.exit(1)

    _dataset_id = sys.argv[1]
    _record_id  = sys.argv[2]
    _force      = "--force" in sys.argv

    _cfg = get_record_config(_dataset_id, _record_id)
    ensure_dirs(_cfg)
    run(_cfg, force=_force)