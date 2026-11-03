"""
Reference copy of data_pipeline/phases/phase2_component_map/reconcile.py.

Kept in M3 (sha256 fb5d4e9f48bc9df09b7e732cf4202cfcb6e3f171338f4cff117473c8490b4806,
identical to tag `pre-revamp`). Reference only: not imported by the pipeline, and
reimplemented cleanly in M10 (composite wiring). Changes from the original: type
annotations only, so strict mypy passes. Behaviour is unchanged.

Known issues to fix in the M10 reimplementation (hard rule 14):
- the placeholder loop iterates `valid_channels` (a set), so warning and
  placeholder order depend on PYTHONHASHSEED;
- `difflib.get_close_matches` over a set breaks score ties in hash order.

Original docstring:

phases/phase2_component_map/reconcile.py
-----------------------------------------
Post-LLM reconciliation utilities for phase 2.

LLMs hallucinate names. These functions snap LLM output back to ground
truth — the actual channel names and feature names that exist in the data.
Nothing downstream of phase 2 should ever see a name that isn't in
cuts.parquet or the real channel list.

Used by step_b and step_d. Not a pipeline step itself.
"""

from __future__ import annotations

import difflib
from typing import Any


def reconcile_channels(
    result: dict[str, Any],
    valid_channels: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """
    Snap all channel references in a step_b result back to valid_channels.

    Fixes:
      - components[].channels: remove any name not in valid_channels,
        with fuzzy-match suggestion in the warning
      - channel_assignments keys: remove phantom keys, warn

    Returns:
        (cleaned_result, list_of_warnings)
    """
    warnings: list[str] = []

    # ── Fix components[].channels ─────────────────────────────────────────────
    for comp in result.get("components", []):
        original = comp.get("channels", [])
        cleaned: list[str] = []
        for ch in original:
            if ch in valid_channels:
                cleaned.append(ch)
            else:
                suggestion = _closest(ch, valid_channels)
                warnings.append(
                    f"step_b: component '{comp['name']}' referenced phantom channel "
                    f"'{ch}' — removed"
                    + (f" (closest valid: '{suggestion}')" if suggestion else "")
                )
        comp["channels"] = cleaned

    # ── Fix channel_assignments keys ──────────────────────────────────────────
    phantom_keys = [k for k in result.get("channel_assignments", {}) if k not in valid_channels]
    for k in phantom_keys:
        suggestion = _closest(k, valid_channels)
        warnings.append(
            f"step_b: channel_assignments had phantom key '{k}' — removed"
            + (f" (closest valid: '{suggestion}')" if suggestion else "")
        )
        del result["channel_assignments"][k]

    # ── Ensure every valid channel has an assignment ──────────────────────────
    # If the LLM dropped a real channel entirely, add a placeholder so
    # downstream phases don't have to handle missing keys.
    assignments = result.setdefault("channel_assignments", {})
    for ch in valid_channels:
        if ch not in assignments:
            warnings.append(
                f"step_b: channel '{ch}' had no assignment — adding 'low' placeholder"
            )
            assignments[ch] = {
                "component":             "unknown",
                "observable":            ch,
                "degradation_relevance": "low",
                "reasoning":             "Auto-filled: LLM did not assign this channel",
            }

    return result, warnings


def reconcile_output_variables(
    result: dict[str, Any],
    signals: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """
    Snap all output_variables in a step_d result back to real feature names.

    For each output variable:
      1. Check that 'name' exists in the signals list
      2. If not, try fuzzy-match against valid feature names
      3. If a match is found within edit distance, remap to it
      4. If no match found, drop the output and warn
      5. Re-derive 'channel' and 'stat' from the actual signal record
         (don't trust what the LLM wrote for these)

    Returns:
        (cleaned_result, list_of_warnings)
    """
    warnings: list[str] = []
    sig_index = {s["name"]: s for s in signals}
    valid_features = set(sig_index.keys())

    cleaned_outputs: list[dict[str, Any]] = []
    for out in result.get("output_variables", []):
        name = out.get("name", "")

        if name in sig_index:
            # Name is valid — re-derive channel/stat from ground truth
            sig = sig_index[name]
            out["channel"] = sig["channel"]
            out["stat"]    = sig["stat"]
            out["unit"]    = sig["unit"]
            cleaned_outputs.append(out)

        else:
            # Try fuzzy match
            match = _closest(name, valid_features, cutoff=0.75)
            if match:
                sig = sig_index[match]
                warnings.append(
                    f"step_d: output '{name}' doesn't exist — remapped to '{match}'"
                )
                out["name"]    = match
                out["channel"] = sig["channel"]
                out["stat"]    = sig["stat"]
                out["unit"]    = sig["unit"]
                cleaned_outputs.append(out)
            else:
                warnings.append(
                    f"step_d: output '{name}' doesn't exist and no close match found — dropped"
                )

    # Deduplicate by name (LLM sometimes returns the same feature twice)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for out in cleaned_outputs:
        if out["name"] not in seen:
            seen.add(out["name"])
            deduped.append(out)
        else:
            warnings.append(f"step_d: duplicate output '{out['name']}' — kept first occurrence")

    result["output_variables"] = deduped
    return result, warnings


def _closest(name: str, candidates: set[str], cutoff: float = 0.6) -> str | None:
    """
    Return the closest match from candidates using SequenceMatcher,
    or None if nothing scores above cutoff.
    """
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None