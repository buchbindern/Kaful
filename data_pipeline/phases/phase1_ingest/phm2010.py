"""
phases/phase1_ingest/phm2010.py
---------------------------------
PHM 2010 dataset-specific loader — steps a–c of phase 1.

Handles the PHM 2010 format:
    datasets/phm2010/raw/<record_id>/
        <record_id>_wear.csv      — one row per cut: cut number, VB1, VB2, VB3 (in 10^-3 mm)
        <record_id>/              — subfolder with one CSV per cut

Signal columns (from official dataset description, confirmed by dataset_meta.py):
    0: Force X   (N)
    1: Force Y   (N)
    2: Force Z   (N)
    3: Vibration X (g)
    4: Vibration Y (g)
    5: Vibration Z (g)
    6: AE-RMS    (V)

Cut file naming — all known PHM 2010 distribution variants are handled:
    "1.csv", "001.csv"        plain numeric
    "c_1_001.csv"             c_<record>_<cut>  (Kaggle distribution)
    "c1_001.csv"              c<record>_<cut>   (some mirrors)

Wear files: values are in 10^-3 mm — converted to mm on load.
Wear column used: mean of VB1, VB2, VB3 (average flank wear across tool flutes).

Returns:
    cuts_df   — DataFrame: index=cut_index, columns=channel names,
                            each cell is a 1D np.ndarray of raw samples
    wear_df   — DataFrame: index=cut_index, columns=[wear_mm, vb1_mm, vb2_mm, vb3_mm]
    raw_meta  — dict with channel info, sampling rate, n_samples_per_cut
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Load and segment PHM 2010 data for a single record (e.g. c1, c4, c6).

    Args:
        cfg: result of get_record_config()

    Returns:
        (cuts_df, wear_df, raw_meta)
    """
    record_id = cfg["record_id"]
    raw_dir   = cfg["raw_dir"] / record_id
    meta      = cfg["meta"]     # dataset_meta.py module, or None

    print(f"\n  Step a — detecting format and validating raw directory...")
    _validate_raw_dir(raw_dir, record_id)

    # ── Resolve column names and units ────────────────────────────────────────
    channels   = _resolve_channels(meta)
    wear_scale = getattr(meta, "WEAR_SCALE", 1e-3) if meta else 1e-3

    print(f"    raw_dir  : {raw_dir}")
    print(f"    channels : {[c['name'] for c in channels]}")
    print(f"    wear_scale: {wear_scale} (raw → mm)")

    # ── Step b — load signal files ────────────────────────────────────────────
    print(f"\n  Step b — loading signal files...")
    cuts_df, n_samples_per_cut = _load_signal_files(raw_dir, record_id, channels)
    print(f"    Loaded {len(cuts_df)} cuts, {n_samples_per_cut} samples/cut")

    # ── Step c — load wear file and align to cut index ────────────────────────
    print(f"\n  Step c — loading wear labels...")
    wear_df = _load_wear_file(raw_dir, record_id, cuts_df.index, wear_scale)
    print(f"    Wear range: {wear_df['wear_mm'].min():.4f} – {wear_df['wear_mm'].max():.4f} mm")

    # ── Raw metadata ──────────────────────────────────────────────────────────
    sampling_rate = (
        getattr(meta, "CUTTING_PARAMS", {}).get("sampling_rate_hz", 50_000)
        if meta else 50_000
    )
    raw_meta = {
        "channels":           channels,
        "n_samples_per_cut":  n_samples_per_cut,
        "sampling_rate_hz":   sampling_rate,
        "wear_scale_applied": wear_scale,
        "source_format":      "phm2010_per_cut_csv",
    }

    return cuts_df, wear_df, raw_meta


# ── Cut file naming ───────────────────────────────────────────────────────────

def _parse_cut_number(stem: str) -> int | None:
    """
    Extract the integer cut number from a filename stem.

    Handles all known PHM 2010 naming conventions:
        "1"         → 1    plain numeric
        "001"       → 1    zero-padded numeric
        "c_1_001"   → 1    c_<record>_<cut>  (Kaggle)
        "c1_001"    → 1    c<record>_<cut>
    Returns None if the stem doesn't match any known pattern.
    """
    if stem.isdigit():
        return int(stem)
    m = re.fullmatch(r"c_\d+_(\d+)", stem)
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"c\d+_(\d+)", stem)
    if m:
        return int(m.group(1))
    return None


def _find_cut_files(directory: Path) -> list[tuple[Path, int]]:
    """
    Find all cut CSVs in a directory, returning (path, cut_number) sorted by cut number.
    Skips any CSV that doesn't match a recognised cut filename pattern.
    """
    results = []
    for f in directory.glob("*.csv"):
        n = _parse_cut_number(f.stem)
        if n is not None:
            results.append((f, n))
    return sorted(results, key=lambda x: x[1])


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_raw_dir(raw_dir: Path, record_id: str) -> None:
    """
    Check that the raw directory exists and contains recognisable files.
    Reports what it found so naming issues are immediately obvious.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Expected layout:\n"
            f"    datasets/phm2010/raw/{record_id}/\n"
            f"        {record_id}_wear.csv\n"
            f"        {record_id}/\n"
            f"            c_1_001.csv, c_1_002.csv, ...  (one per cut)\n"
            f"Download: https://ieee-dataport.org/documents/2010-phm-society-conference-data-challenge"
        )

    # Wear file
    wear_file = raw_dir / f"{record_id}_wear.csv"
    if not wear_file.exists():
        alt_names = ["wear.csv", f"{record_id}wear.csv", f"{record_id}_wear"]
        found = next((raw_dir / n for n in alt_names if (raw_dir / n).exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"Wear file not found in {raw_dir}\n"
                f"Looked for: {wear_file.name} (and fallbacks: {alt_names})\n"
                f"Check your raw data layout."
            )

    # Cut files — check flat level then nested
    cut_files_flat   = _find_cut_files(raw_dir)
    cut_files_nested = _find_cut_files(raw_dir / record_id)
    cut_files = cut_files_flat or cut_files_nested

    if len(cut_files) == 0:
        # Show what IS there so the user can diagnose naming issues
        all_csvs = list(raw_dir.glob("*.csv")) + list((raw_dir / record_id).glob("*.csv"))
        examples = [f.name for f in all_csvs[:8]]
        raise FileNotFoundError(
            f"No recognisable cut files found in {raw_dir} or {raw_dir / record_id}\n"
            f"Files present: {examples}\n"
            f"Recognised naming formats: 1.csv, 001.csv, c_1_001.csv, c1_001.csv"
        )

    layout  = "nested" if (cut_files_nested and not cut_files_flat) else "flat"
    example = cut_files[0][0].name
    print(f"    ✓ Found wear file + {len(cut_files)} cut files  (layout: {layout}, e.g. '{example}')")


def _resolve_channels(meta) -> list[dict]:
    """
    Return channel definitions. Uses dataset_meta.py if available,
    else falls back to PHM 2010 dataset description defaults.
    """
    if meta is not None and hasattr(meta, "SIGNAL_COLUMNS"):
        return meta.SIGNAL_COLUMNS

    return [
        {"index": 0, "name": "force_x",    "unit": "N", "type": "force",            "axis": "x"},
        {"index": 1, "name": "force_y",    "unit": "N", "type": "force",            "axis": "y"},
        {"index": 2, "name": "force_z",    "unit": "N", "type": "force",            "axis": "z"},
        {"index": 3, "name": "vibration_x","unit": "g", "type": "vibration",        "axis": "x"},
        {"index": 4, "name": "vibration_y","unit": "g", "type": "vibration",        "axis": "y"},
        {"index": 5, "name": "vibration_z","unit": "g", "type": "vibration",        "axis": "z"},
        {"index": 6, "name": "ae_rms",     "unit": "V", "type": "acoustic_emission","axis": None},
    ]


def _load_signal_files(
    raw_dir: Path,
    record_id: str,
    channels: list[dict],
) -> tuple[pd.DataFrame, int]:
    """
    Load all per-cut CSV files.

    PHM 2010 per-cut files have no header — 7 columns of raw float values,
    one row per sample. Handles flat and nested layouts, all naming formats.

    Returns:
        cuts_df           — DataFrame: index=cut_index (0-based), each cell is
                            a 1D np.ndarray of float64 samples for that channel
        n_samples_per_cut — sample count from the first successfully loaded file
    """
    # Resolve directory
    nested_dir = raw_dir / record_id
    if nested_dir.exists() and _find_cut_files(nested_dir):
        cuts_dir = nested_dir
    else:
        cuts_dir = raw_dir

    cut_files = _find_cut_files(cuts_dir)

    rows = []
    n_samples_per_cut = None
    n_loaded  = 0
    n_skipped = 0

    for cut_path, cut_number in cut_files:
        cut_index = cut_number - 1   # convert to 0-based

        try:
            data = pd.read_csv(cut_path, header=None).values.astype(np.float64)

            if data.shape[1] != len(channels):
                print(f"    ⚠ cut {cut_number} ({cut_path.name}): "
                      f"expected {len(channels)} cols, got {data.shape[1]} — skipping")
                n_skipped += 1
                continue

            if n_samples_per_cut is None:
                n_samples_per_cut = data.shape[0]

            row = {"cut_index": cut_index}
            for ch in channels:
                row[ch["name"]] = data[:, ch["index"]]

            rows.append(row)
            n_loaded += 1

        except Exception as e:
            print(f"    ⚠ cut {cut_number} ({cut_path.name}): failed ({e}) — skipping")
            n_skipped += 1

    if n_loaded == 0:
        raise ValueError(f"No cut files could be loaded from {cuts_dir}")

    if n_skipped > 0:
        print(f"    ⚠ Skipped {n_skipped}/{n_loaded + n_skipped} cuts")

    cuts_df = pd.DataFrame(rows).set_index("cut_index")
    return cuts_df, n_samples_per_cut or 0


def _load_wear_file(
    raw_dir: Path,
    record_id: str,
    cut_index: pd.Index,
    wear_scale: float,
) -> pd.DataFrame:
    """
    Load the wear CSV and align to the cut index from the signal files.

    PHM 2010 wear file format (no header):
        col 0: cut number (1-indexed)
        col 1: VB1  (flute 1 flank wear, in 10^-3 mm)
        col 2: VB2  (flute 2 flank wear, in 10^-3 mm)
        col 3: VB3  (flute 3 flank wear, in 10^-3 mm)

    Applies wear_scale to convert raw values to mm.
    Primary wear_mm = mean(VB1, VB2, VB3).
    """
    candidates = [
        raw_dir / f"{record_id}_wear.csv",
        raw_dir / "wear.csv",
        raw_dir / f"{record_id}wear.csv",
    ]
    wear_file = next((p for p in candidates if p.exists()), None)
    if wear_file is None:
        raise FileNotFoundError(f"Wear file not found in {raw_dir}")

    # Auto-detect header: peek at the first cell — if it's not numeric, there's a header row
    _peek = pd.read_csv(wear_file, header=None, nrows=1)
    _first_cell = str(_peek.iloc[0, 0]).strip()
    _has_header = not _first_cell.lstrip("-").isdigit()
    raw = pd.read_csv(wear_file, header=0 if _has_header else None)

    if raw.shape[1] == 4:
        raw.columns = ["cut_number", "vb1_raw", "vb2_raw", "vb3_raw"]
        raw["cut_index"] = raw["cut_number"].astype(int) - 1
        raw = raw.set_index("cut_index")
        raw["vb1_mm"]  = raw["vb1_raw"] * wear_scale
        raw["vb2_mm"]  = raw["vb2_raw"] * wear_scale
        raw["vb3_mm"]  = raw["vb3_raw"] * wear_scale
        raw["wear_mm"] = raw[["vb1_mm", "vb2_mm", "vb3_mm"]].mean(axis=1)
    elif raw.shape[1] == 2:
        raw.columns = ["cut_number", "wear_raw"]
        raw["cut_index"] = raw["cut_number"].astype(int) - 1
        raw = raw.set_index("cut_index")
        raw["wear_mm"] = raw["wear_raw"] * wear_scale
    else:
        raise ValueError(
            f"Unexpected wear file format: {raw.shape[1]} columns in {wear_file}\n"
            f"Expected 4 columns (cut, vb1, vb2, vb3) or 2 columns (cut, wear)"
        )

    wear_cols = [c for c in ["wear_mm", "vb1_mm", "vb2_mm", "vb3_mm"] if c in raw.columns]
    wear_df = raw[wear_cols].reindex(cut_index)

    n_missing = wear_df["wear_mm"].isna().sum()
    if n_missing > 0:
        print(f"    ⚠ {n_missing} cuts have no wear label — forward-filling")
        wear_df = wear_df.ffill()

    return wear_df