"""
pipeline/preprocessor.py

Converts raw CSV data into a structured DataSummary JSON that feeds
every downstream Claude call (decomposition, generation, integration).

Supports two CSV types:
  - events    : time-series operational data (sensor readings per event)
  - maintenance: scheduled/unscheduled maintenance logs with wear readings

The richer the DataSummary, the better the decomposition and generation
quality will be — this is the foundation everything else builds on.

Usage:
    from pipeline.preprocessor import build_data_summary, save_summary

    summary = build_data_summary(
        machine_name="coffee_machine",
        events_csv="data/coffee_machine/events.csv",
        maintenance_csv="data/coffee_machine/maintenance.csv",  # optional
    )
    save_summary(summary, "data/coffee_machine/summary.json")
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class SignalStats:
    """Statistics for a single signal/column."""
    name: str
    dtype: str
    missing_pct: float

    # Numeric signals
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    unit_hint: Optional[str] = None        # inferred from column name

    # Categorical signals
    categories: Optional[list] = None
    n_unique: Optional[int] = None


@dataclass
class MaintenanceSignal:
    """A wear/degradation signal from the maintenance CSV."""
    name: str
    min: float
    max: float
    mean: float
    maintenance_types: list[str]           # which events reset/affect this signal


@dataclass
class DataSummary:
    """
    Complete structured summary of a machine's data.
    This is the single input that flows into every Claude call.
    """
    # Identity
    machine_name: str
    machine_type: str                      # inferred or provided

    # Events data
    n_events: int
    date_range: dict                       # {"start": ..., "end": ..., "days": ...}
    product_types: list[str]              # e.g. ["latte", "espresso"]
    signals: list[SignalStats]            # one per column

    # Maintenance data (optional)
    has_maintenance: bool
    n_maintenance_records: int
    maintenance_types: list[str]
    wear_signals: list[MaintenanceSignal]  # degradation indicators

    # Inferred characteristics (used by decomposition step)
    temporal_columns: list[str]           # timestamp-like columns
    id_columns: list[str]                 # machine_id, serial_no etc
    target_columns: list[str]             # event_success, error_code etc
    degradation_indicators: list[str]     # columns that suggest wear/aging


# ------------------------------------------------------------------
# Unit hint inference
# ------------------------------------------------------------------

UNIT_HINTS = {
    "temperature": "°C",
    "temp":        "°C",
    "pressure":    "bar",
    "weight":      "g",
    "volume":      "ml",
    "time":        "s",
    "duration":    "s",
    "speed":       "rpm",
    "current":     "A",
    "voltage":     "V",
    "power":       "W",
    "consumption": "W",
    "level":       "%",
    "pct":         "%",
    "count":       "count",
    "height":      "mm",
    "size":        "mm",
}

def infer_unit(col_name: str) -> Optional[str]:
    col_lower = col_name.lower()
    for keyword, unit in UNIT_HINTS.items():
        if keyword in col_lower:
            return unit
    return None


# ------------------------------------------------------------------
# Column classification
# ------------------------------------------------------------------

def classify_columns(df: pd.DataFrame) -> dict:
    """Classify columns into temporal, id, target, degradation, and signal."""
    temporal, ids, targets, degradation = [], [], [], []

    for col in df.columns:
        col_lower = col.lower()
        if any(col_lower.endswith(k) or col_lower.startswith(k) for k in ["timestamp", "datetime", "date"]) or col_lower in ["time", "week"]:
            temporal.append(col)
        elif any(k in col_lower for k in ["machine_id", "_id", "serial", "device"]):
            ids.append(col)
        elif any(k in col_lower for k in ["success", "error", "fault", "alarm", "status", "mode"]):
            targets.append(col)
        elif any(k in col_lower for k in ["wear", "buildup", "degradation", "aging", "fatigue", "erosion"]):
            degradation.append(col)

    return {
        "temporal": temporal,
        "ids": ids,
        "targets": targets,
        "degradation": degradation,
    }


# ------------------------------------------------------------------
# Signal stats extraction
# ------------------------------------------------------------------

def extract_signal_stats(df: pd.DataFrame, skip_cols: list[str]) -> list[SignalStats]:
    """Compute stats for every column not in skip_cols."""
    stats = []

    for col in df.columns:
        if col in skip_cols:
            continue

        series = df[col]
        missing_pct = round(series.isna().mean() * 100, 2)
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            stats.append(SignalStats(
                name=col,
                dtype=dtype,
                missing_pct=missing_pct,
                min=round(float(clean.min()), 4) if len(clean) else None,
                max=round(float(clean.max()), 4) if len(clean) else None,
                mean=round(float(clean.mean()), 4) if len(clean) else None,
                std=round(float(clean.std()), 4) if len(clean) else None,
                median=round(float(clean.median()), 4) if len(clean) else None,
                unit_hint=infer_unit(col),
            ))
        else:
            unique_vals = series.dropna().unique().tolist()
            stats.append(SignalStats(
                name=col,
                dtype=dtype,
                missing_pct=missing_pct,
                categories=unique_vals[:20],   # cap at 20 for readability
                n_unique=series.nunique(),
            ))

    return stats


# ------------------------------------------------------------------
# Maintenance processing
# ------------------------------------------------------------------

def extract_wear_signals(df: pd.DataFrame) -> list[MaintenanceSignal]:
    """Extract wear/degradation signals from maintenance CSV."""
    wear_cols = [c for c in df.columns if any(
        k in c.lower() for k in ["wear", "buildup", "degradation"]
    )]

    maintenance_types = df["maintenance_type"].unique().tolist() if "maintenance_type" in df.columns else []
    signals = []

    for col in wear_cols:
        clean = df[col].dropna()
        if len(clean) == 0:
            continue

        # Find which maintenance types are associated with this signal
        if "maintenance_type" in df.columns:
            associated = df[df[col].notna()]["maintenance_type"].unique().tolist()
        else:
            associated = maintenance_types

        signals.append(MaintenanceSignal(
            name=col,
            min=round(float(clean.min()), 4),
            max=round(float(clean.max()), 4),
            mean=round(float(clean.mean()), 4),
            maintenance_types=associated,
        ))

    return signals


# ------------------------------------------------------------------
# Machine type inference
# ------------------------------------------------------------------

MACHINE_TYPE_HINTS = {
    "coffee":     ["espresso", "grinder", "brew", "milk", "steam", "foam"],
    "vehicle":    ["rpm", "throttle", "fuel", "transmission", "brake", "tire"],
    "industrial": ["pump", "valve", "compressor", "conveyor", "actuator"],
    "battery":    ["soc", "voltage", "current", "cell", "charge", "discharge"],
    "hvac":       ["hvac", "cooling", "heating", "airflow", "refrigerant"],
    "rotorcraft": ["rotor", "propeller", "thrust", "altitude", "pitch"],
}

def infer_machine_type(machine_name: str, columns: list[str]) -> str:
    combined = (machine_name + " " + " ".join(columns)).lower()
    for machine_type, keywords in MACHINE_TYPE_HINTS.items():
        if any(k in combined for k in keywords):
            return machine_type
    return "industrial"  # safe default


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def build_data_summary(
    machine_name: str,
    events_csv: str,
    maintenance_csv: Optional[str] = None,
    machine_type: Optional[str] = None,
) -> DataSummary:
    """
    Build a complete DataSummary from raw CSV files.

    Args:
        machine_name    : human readable name e.g. "coffee_machine"
        events_csv      : path to operational events CSV
        maintenance_csv : path to maintenance log CSV (optional but recommended)
        machine_type    : override auto-detection if needed
    """
    print(f"[preprocessor] Loading events: {events_csv}")
    events_df = pd.read_csv(events_csv)
    print(f"  Shape: {events_df.shape}")

    # Classify columns
    col_classes = classify_columns(events_df)

    # Date range
    date_range = {"start": None, "end": None, "days": None}
    if col_classes["temporal"]:
        ts_col = col_classes["temporal"][0]
        try:
            ts = pd.to_datetime(events_df[ts_col])
            date_range = {
                "start": str(ts.min()),
                "end":   str(ts.max()),
                "days":  int((ts.max() - ts.min()).days),
            }
        except Exception:
            pass

    # Product types
    product_types = []
    for col in ["product_type", "product", "type", "category"]:
        if col in events_df.columns:
            product_types = events_df[col].dropna().unique().tolist()
            break

    # Skip metadata columns for signal stats
    skip = col_classes["temporal"] + col_classes["ids"]
    signals = extract_signal_stats(events_df, skip_cols=skip)

    # Maintenance
    has_maintenance = False
    n_maintenance = 0
    maintenance_types = []
    wear_signals = []

    if maintenance_csv and Path(maintenance_csv).exists():
        print(f"[preprocessor] Loading maintenance: {maintenance_csv}")
        maint_df = pd.read_csv(maintenance_csv)
        print(f"  Shape: {maint_df.shape}")
        has_maintenance = True
        n_maintenance = len(maint_df)
        if "maintenance_type" in maint_df.columns:
            maintenance_types = maint_df["maintenance_type"].dropna().unique().tolist()
        wear_signals = extract_wear_signals(maint_df)

    # Infer machine type
    if not machine_type:
        machine_type = infer_machine_type(machine_name, events_df.columns.tolist())

    summary = DataSummary(
        machine_name=machine_name,
        machine_type=machine_type,
        n_events=len(events_df),
        date_range=date_range,
        product_types=product_types,
        signals=signals,
        has_maintenance=has_maintenance,
        n_maintenance_records=n_maintenance,
        maintenance_types=maintenance_types,
        wear_signals=wear_signals,
        temporal_columns=col_classes["temporal"],
        id_columns=col_classes["ids"],
        target_columns=col_classes["targets"],
        degradation_indicators=col_classes["degradation"],
    )

    print(f"\n[preprocessor] Summary complete:")
    print(f"  Machine type   : {summary.machine_type}")
    print(f"  Events         : {summary.n_events}")
    print(f"  Signals        : {len(summary.signals)}")
    print(f"  Wear signals   : {len(summary.wear_signals)}")
    print(f"  Degradation    : {summary.degradation_indicators}")
    print(f"  Date range     : {summary.date_range['days']} days")

    return summary


def save_summary(summary: DataSummary, output_path: str) -> None:
    """Save DataSummary to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict, handle numpy types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=convert)

    print(f"\n[preprocessor] Saved to: {output_path}")


def load_summary(path: str) -> DataSummary:
    """Load a saved DataSummary from JSON."""
    with open(path) as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
    data["signals"] = [SignalStats(**s) for s in data["signals"]]
    data["wear_signals"] = [MaintenanceSignal(**w) for w in data["wear_signals"]]
    return DataSummary(**data)