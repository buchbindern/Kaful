"""Characterization tests pinning the reference reconcile.py behaviour for M10.

Warnings are compared sorted: the original iterates sets, so their order depends
on PYTHONHASHSEED (logged as an M10 eval case).
"""

from typing import Any

from kaful.reference.reconcile import reconcile_channels, reconcile_output_variables

CHANNELS = {"spindle_speed", "vibration_x", "vibration_y"}


def test_reconcile_channels_drops_phantoms_and_fills_missing() -> None:
    result: dict[str, Any] = {
        "components": [{"name": "bearing", "channels": ["vibration_x", "vibraton_y", "pressure"]}],
        "channel_assignments": {"vibration_x": {"component": "bearing"}, "temp": {"component": "x"}},
    }
    cleaned, warnings = reconcile_channels(result, CHANNELS)
    assert cleaned["components"][0]["channels"] == ["vibration_x"]
    assert set(cleaned["channel_assignments"]) == CHANNELS
    assert cleaned["channel_assignments"]["vibration_y"]["degradation_relevance"] == "low"
    assert sorted(warnings) == sorted([
        "step_b: component 'bearing' referenced phantom channel 'vibraton_y' — removed "
        "(closest valid: 'vibration_y')",
        "step_b: component 'bearing' referenced phantom channel 'pressure' — removed",
        "step_b: channel_assignments had phantom key 'temp' — removed",
        "step_b: channel 'spindle_speed' had no assignment — adding 'low' placeholder",
        "step_b: channel 'vibration_y' had no assignment — adding 'low' placeholder",
    ])


def test_reconcile_output_variables_remaps_drops_and_dedupes() -> None:
    signals = [
        {"name": "vib_x_rms", "channel": "vibration_x", "stat": "rms", "unit": "g"},
        {"name": "vib_y_rms", "channel": "vibration_y", "stat": "rms", "unit": "g"},
    ]
    result: dict[str, Any] = {"output_variables": [
        {"name": "vib_x_rms", "channel": "wrong", "stat": "max", "unit": "mm"},
        {"name": "vib_y_rmss"},
        {"name": "acoustic_level"},
        {"name": "vib_x_rms"},
    ]}
    cleaned, warnings = reconcile_output_variables(result, signals)
    assert cleaned["output_variables"] == [
        {"name": "vib_x_rms", "channel": "vibration_x", "stat": "rms", "unit": "g"},
        {"name": "vib_y_rms", "channel": "vibration_y", "stat": "rms", "unit": "g"},
    ]
    assert warnings == [
        "step_d: output 'vib_y_rmss' doesn't exist — remapped to 'vib_y_rms'",
        "step_d: output 'acoustic_level' doesn't exist and no close match found — dropped",
        "step_d: duplicate output 'vib_x_rms' — kept first occurrence",
    ]
