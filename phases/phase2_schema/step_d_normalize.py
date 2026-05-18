"""
phases/phase2_schema/step_d_normalize.py
-----------------------------------------
Step D: Parse and normalize raw extraction runs.

Takes the raw JSON strings from step_c and:
1. Parses each run (handles markdown fences)
2. Normalizes field names to lowercase_with_underscores
3. Dedupes fields within each run

Output saved to: outputs/schema/step_d_normalized.json
"""

import json
import re

from utils.parsing import parse_json


def run(cfg: dict, all_runs: list[str]) -> list[list[dict]]:
    """
    Parse and normalize all extraction runs.

    Args:
        cfg:      result of get_machine_config()
        all_runs: list of raw JSON strings from step_c

    Returns:
        list of normalized, deduped runs (list of lists of field dicts)
    """
    output_path = cfg["step_d_normalized"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_d already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print("  Running step_d — parsing and normalizing runs...")

    # Parse
    parsed_runs = _parse_all_runs(all_runs)

    # Normalize
    normalized_runs = [
        [_normalize_field(field) for field in run]
        for run in parsed_runs
    ]

    # Dedupe within each run
    deduped_runs = [_dedupe_run(run) for run in normalized_runs]

    # Print summary
    for i, run in enumerate(deduped_runs):
        print(f"    Run {i+1}: {len(run)} fields after normalize + dedup")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(deduped_runs, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return deduped_runs


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_all_runs(all_runs: list[str]) -> list[list[dict]]:
    """Parse raw model outputs into Python objects."""
    parsed_runs = []

    for i, run in enumerate(all_runs, start=1):
        data = parse_json(run)
        if data and isinstance(data, list):
            parsed_runs.append(data)
            print(f"    Run {i}: parsed {len(data)} fields")
        else:
            print(f"    Run {i}: ✗ failed to parse — skipping")

    return parsed_runs


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize_field_name(name: str) -> str:
    """Normalize a field name to lowercase_with_underscores."""
    if not isinstance(name, str):
        return name

    name = name.lower().strip()
    name = re.sub(r"[\s\-]+", "_", name)   # spaces and hyphens → underscore
    name = re.sub(r"[^a-z0-9_]", "", name) # remove non-alphanumeric
    name = re.sub(r"_+", "_", name)         # collapse multiple underscores
    name = name.strip("_")                  # remove leading/trailing underscores

    return name


def _normalize_text_value(value) -> str:
    """Light normalization for non-field-name string values."""
    if not isinstance(value, str):
        return value

    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def _normalize_field(field: dict) -> dict:
    """
    Normalize one field dict into a consistent schema.
    Keeps original field_name and adds normalized_name.
    """
    return {
        "field_name":      field.get("field_name"),
        "normalized_name": _normalize_field_name(field.get("field_name", "")),
        "category":        _normalize_field_name(field.get("category", "")) if field.get("category") else None,
        "data_type":       _normalize_field_name(field.get("data_type", "")) if field.get("data_type") else None,
        "unit":            _normalize_field_name(field.get("unit", "")) if field.get("unit") else None,
        "typical_range":   _normalize_text_value(field.get("typical_range")),
        "description":     _normalize_text_value(field.get("description")),
        "source_pages":    field.get("source_pages"),
        "evidence":        field.get("evidence"),
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedupe_run(run: list[dict]) -> list[dict]:
    """Remove duplicate fields within a single run based on normalized_name."""
    seen    = set()
    deduped = []

    for field in run:
        key = field["normalized_name"]
        if key and key not in seen:
            seen.add(key)
            deduped.append(field)

    return deduped