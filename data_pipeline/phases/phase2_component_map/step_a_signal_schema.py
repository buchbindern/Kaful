"""
phases/phase2_component_map/step_a_signal_schema.py
-----------------------------------------------------
Step a: Build a structured signal schema from phase 1 outputs.

No LLM call here — this is pure data. We read ingest_meta.json and
cuts.parquet and produce a list of signal descriptors that step b
will pass to the LLM for component mapping.

Channel/stat splitting uses a known-suffix registry rather than naive
rsplit, so multi-word stats like 'mean_abs' and 'crest_factor' are
correctly identified. The registry is derived from normalize.py's
FEATURE_FUNCS keys and can be extended there without touching this file.

When dataset_meta.py is present, SIGNAL_COLUMNS entries are merged in
so the LLM gets units and physical type without having to guess.

Output: step_a_signal_schema.json
[
  {
    "name":         "force_x_rms",
    "channel":      "force_x",
    "stat":         "rms",
    "unit":         "N",
    "channel_type": "force",
    "axis":         "x",
    "mean":         12.4,
    "std":          3.1,
    "min":          4.2,
    "max":          31.7
  },
  ...
]
"""

import json
import pandas as pd


# ── Known stat suffixes ───────────────────────────────────────────────────────
# Must match the keys in normalize.py's FEATURE_FUNCS exactly.
# Sorted longest-first so 'mean_abs' is tried before 'abs', 'crest_factor'
# before 'factor', etc. — prevents partial matches swallowing the wrong suffix.
KNOWN_STATS = sorted(
    ["rms", "mean_abs", "peak", "kurtosis", "crest_factor", "std"],
    key=len,
    reverse=True,
)


def _split_channel_stat(col: str) -> tuple[str, str]:
    """
    Split a feature column name into (channel, stat).

    Tries each known stat suffix in longest-first order. Matches on
    '<channel>_<stat>' where stat is in KNOWN_STATS.

    Examples:
        'force_x_rms'         → ('force_x',    'rms')
        'force_x_mean_abs'    → ('force_x',    'mean_abs')
        'vibration_z_crest_factor' → ('vibration_z', 'crest_factor')
        'ae_rms_rms'          → ('ae_rms',     'rms')
        'ae_rms_mean_abs'     → ('ae_rms',     'mean_abs')

    Falls back to rsplit on the last underscore if no known stat matches
    (handles future stats added to normalize.py without breaking this file).
    """
    for stat in KNOWN_STATS:
        suffix = f"_{stat}"
        if col.endswith(suffix):
            channel = col[: -len(suffix)]
            if channel:          # guard against col == suffix itself
                return channel, stat

    # Fallback: rsplit on last underscore
    parts = col.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return col, "value"


def run(cfg: dict) -> list[dict]:
    output_path = cfg["step_a_schema_path"]

    if output_path.exists():
        print("  ✓ step_a already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_a — building signal schema...")

    # ── Load phase 1 outputs ──────────────────────────────────────────────────
    with open(cfg["ingest_meta_path"]) as f:
        ingest_meta = json.load(f)

    cuts_df = pd.read_parquet(cfg["cuts_path"])

    # ── Build channel lookup from dataset_meta if available ───────────────────
    meta = cfg["meta"]
    channel_info: dict[str, dict] = {}
    if meta is not None and hasattr(meta, "SIGNAL_COLUMNS"):
        for col in meta.SIGNAL_COLUMNS:
            channel_info[col["name"]] = {
                "unit":         col.get("unit", "unknown"),
                "channel_type": col.get("type", "unknown"),
                "axis":         col.get("axis"),
            }

    # Fallback: ingest_meta channels list
    for ch in ingest_meta.get("channels", []):
        name = ch.get("name", "")
        if name not in channel_info:
            channel_info[name] = {
                "unit":         ch.get("unit", "unknown"),
                "channel_type": ch.get("type", "unknown"),
                "axis":         ch.get("axis"),
            }

    # ── Build signal descriptors ──────────────────────────────────────────────
    signals  = []
    for col in cuts_df.columns:
        channel, stat = _split_channel_stat(col)
        ch_meta       = channel_info.get(channel, {})
        series        = cuts_df[col].dropna()

        signals.append({
            "name":         col,
            "channel":      channel,
            "stat":         stat,
            "unit":         ch_meta.get("unit", "unknown"),
            "channel_type": ch_meta.get("channel_type", "unknown"),
            "axis":         ch_meta.get("axis"),
            "mean":         round(float(series.mean()), 6) if len(series) else None,
            "std":          round(float(series.std()),  6) if len(series) else None,
            "min":          round(float(series.min()),  6) if len(series) else None,
            "max":          round(float(series.max()),  6) if len(series) else None,
        })

    # ── Verify channel extraction is clean ───────────────────────────────────
    unique_channels = sorted({s["channel"] for s in signals})
    known_channels  = set(channel_info.keys())
    unknown = [ch for ch in unique_channels if ch not in known_channels]
    if unknown:
        print(f"    ⚠ channels not in dataset_meta: {unknown}")
        print(f"      (check KNOWN_STATS covers all stat suffixes in normalize.py)")

    print(f"    {len(signals)} features across {len(unique_channels)} channels "
          f"({len(unique_channels)} unique)")
    print(f"    channels: {unique_channels}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(signals, f, indent=2)
    print(f"    Saved → {output_path.name}")

    return signals