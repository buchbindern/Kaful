"""Phase 3 simulator CSV naming contract (M2 bug: 3c never stated what 3d/7a glob for). Offline."""

from pathlib import Path

import pytest

from phases.phase3_simulate import step_c_codegen, step_d_validate


def _cfg(tmp_path: Path, simulator: str) -> dict:
    sim_dir = tmp_path / "simulate"
    sim_dir.mkdir()
    (sim_dir / "simulator.py").write_text(simulator)
    return {"simulate_dir": sim_dir, "sim_step_c_code": sim_dir / "simulator.py",
            "sim_step_d_validation": sim_dir / "step_d_validation.json"}


def test_prompt_states_the_csv_naming_contract() -> None:
    prompt = step_c_codegen.DATA_SIMULATOR_PROMPT
    assert step_c_codegen.EVENTS_CSV_NAME in prompt
    assert step_c_codegen.MAINTENANCE_CSV_NAME in prompt
    assert "combined" in prompt
    prompt.format(machine_type="m", sensor_measurements="[]", manual_context="", simulator_plan="{}")


def test_contract_names_match_the_glob_used_downstream() -> None:
    name = step_c_codegen.EVENTS_CSV_NAME.replace("<prefix>", "pump").replace("<profile>", "light")
    assert Path(name).match(step_d_validate.EVENTS_CSV_GLOB)
    assert Path(name).stem.split("events_", 1)[-1] == "light"  # phase 7 step_a profile name


def test_wrong_csv_names_fail_with_the_expected_pattern(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "open('pump_light_events.csv', 'w').write('a\\n1\\n')\n")
    with pytest.raises(FileNotFoundError) as err:
        step_d_validate.run(cfg, schema=[])
    msg = str(err.value)
    assert step_d_validate.EVENTS_CSV_GLOB in msg
    assert step_c_codegen.EVENTS_CSV_NAME in msg
    assert "pump_light_events.csv" in msg
