"""
phases/phase1_ingest/normalize.py
----------------------------------
Steps d–e of phase 1: shared across all datasets.

Step d — extract per-cut feature vectors from raw signal windows.
Step e — write normalized outputs to disk.

Dataset-specific loaders handle steps a–c (loading, segmentation, wear
alignment) and hand off a cuts_df and wear_df here. This file never
touches raw data formats — it only sees pandas DataFrames.

Output contract (what phases 2–7 see):
    cuts.parquet
        index : cut_index (int, 0-based)
        cols  : one column per feature per channel
                naming: <channel>_<stat>
                e.g. force_x_rms, vibration_z_kurtosis, ae_rms_peak
                All values are float64.

    wear.parquet
        index : cut_index (int, 0-based)
        cols  : wear_mm (float64)  — always in mm, regardless of source units
                Optional: wear_vb1_mm, wear_vb2_mm, wear_vb3_mm (per-flute if available)

    ingest_meta.json
        dataset_id, record_id, n_cuts, n_features, channels, feature_stats,
        cutting_params (from dataset_meta if available), wear_threshold_mm
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


# ── Feature functions applied to each channel's time-series window ────────────
# Each returns a scalar. Add more here and they'll apply to every channel.

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def _mean_abs(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))

def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))

def _kurtosis(x: np.ndarray) -> float:
    mu  = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 4))

def _crest_factor(x: np.ndarray) -> float:
    r = _rms(x)
    if r == 0:
        return 0.0
    return float(np.max(np.abs(x)) / r)

def _std(x: np.ndarray) -> float:
    return float(np.std(x))

FEATURE_FUNCS = {
    "rms":          _rms,
    "mean_abs":     _mean_abs,
    "peak":         _peak,
    "kurtosis":     _kurtosis,
    "crest_factor": _crest_factor,
    "std":          _std,
}


def run(
    cfg:     dict,
    cuts_df: pd.DataFrame,   # rows=cuts, cols=channel names, values=np.ndarray windows
    wear_df: pd.DataFrame,   # rows=cuts, cols include "wear_mm"
    raw_meta: dict,          # loader-provided metadata (channel names, units, etc.)
) -> None:
    """
    Steps d–e: extract features from raw signal windows and write outputs.

    Args:
        cfg:      result of get_record_config()
        cuts_df:  DataFrame where each cell is a 1D np.ndarray (the raw signal
                  window for that cut × channel). Columns are channel names.
        wear_df:  DataFrame with at minimum a "wear_mm" column, indexed by cut.
        raw_meta: dict from the loader with at minimum:
                    "channels": list of channel dicts (name, unit, type)
                    "n_samples_per_cut": int
                    "sampling_rate_hz": float (if known)
    """
    print(f"\n  Step d — extracting features...")

    channels = [col for col in cuts_df.columns]
    feature_rows = []

    for cut_idx, row in cuts_df.iterrows():
        feat = {"cut_index": cut_idx}
        for ch in channels:
            window = row[ch]
            if window is None or len(window) == 0:
                # Fill with NaN if window is missing (e.g. bad cut)
                for stat in FEATURE_FUNCS:
                    feat[f"{ch}_{stat}"] = float("nan")
            else:
                arr = np.asarray(window, dtype=np.float64)
                for stat, fn in FEATURE_FUNCS.items():
                    feat[f"{ch}_{stat}"] = fn(arr)
        feature_rows.append(feat)

    features_df = pd.DataFrame(feature_rows).set_index("cut_index")

    n_cuts     = len(features_df)
    n_features = len(features_df.columns)
    print(f"    {n_cuts} cuts × {n_features} features ({len(channels)} channels × {len(FEATURE_FUNCS)} stats)")

    # ── Step e — write outputs ─────────────────────────────────────────────
    print(f"\n  Step e — writing outputs...")

    cfg["ingest_dir"].mkdir(parents=True, exist_ok=True)

    # cuts.parquet
    features_df.to_parquet(cfg["cuts_path"])
    print(f"    ✓ cuts.parquet  → {cfg['cuts_path'].name}")

    # wear.parquet — ensure index is named cut_index
    wear_out = wear_df.copy()
    wear_out.index.name = "cut_index"
    wear_out.to_parquet(cfg["wear_path"])
    print(f"    ✓ wear.parquet  → {cfg['wear_path'].name}")

    # ingest_meta.json
    meta = cfg["meta"]
    wear_threshold = (
        getattr(meta, "WEAR_THRESHOLD_MM", None)
        if meta is not None else None
    )
    cutting_params = (
        getattr(meta, "CUTTING_PARAMS", None)
        if meta is not None else None
    )

    ingest_meta = {
        "dataset_id":        cfg["dataset_id"],
        "record_id":         cfg["record_id"],
        "n_cuts":            n_cuts,
        "n_features":        n_features,
        "channels":          raw_meta.get("channels", []),
        "feature_names":     list(features_df.columns),
        "n_samples_per_cut": raw_meta.get("n_samples_per_cut"),
        "sampling_rate_hz":  raw_meta.get("sampling_rate_hz"),
        "wear_threshold_mm": wear_threshold,
        "cutting_params":    cutting_params,
        "wear_range_mm": {
            "min": float(wear_df["wear_mm"].min()),
            "max": float(wear_df["wear_mm"].max()),
        },
    }

    with open(cfg["ingest_meta_path"], "w") as f:
        json.dump(ingest_meta, f, indent=2)
    print(f"    ✓ ingest_meta.json → {cfg['ingest_meta_path'].name}")