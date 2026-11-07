"""The understand prompts' eval sets, offline: good scripted outputs pass every threshold,
legacy-style outputs (sensors as components, alarm limits as channels) fail the
over-extraction metrics, and the eval input is exactly what the phase sends for tiny_pump."""

import json
from pathlib import Path
from typing import Any

import pytest

from kaful.config import KafulSettings, load_machine_config
from kaful.evals.models import EvalResult
from kaful.evals.runner import run_eval
from kaful.phases.p1_ingest.f_chunk_pages import chunk_document
from kaful.phases.p1_ingest.a_check_documents import check_documents
from kaful.phases.p1_ingest.tools.read_document import read_document
from kaful.phases.p2_understand.a_select_context import render
from kaful.prompts.registry import PromptRegistry
from tests.harness.kaful_llm import FIXED_NOW, ScriptedTransport, make_gateway
from tests.harness.kaful_understand import TINY_PUMP, channels, components, fault_modes, q

PROMPTS = ("understand.components", "understand.fault_modes", "understand.sensor_channels")


def _clean_components() -> dict[str, Any]:
    """The phase script without its deliberate flaw (wear_ring cites a paraphrase)."""
    data = components(0)
    for item in data["components"]:
        if item["id"] == "wear_ring":
            item["evidence"] = [q(3, "Its impeller and wear ring wear slowly in service")]
    return data


def _clean_fault_modes() -> dict[str, Any]:
    """The phase script without its deliberate flaw (a channel that does not exist)."""
    data = fault_modes(0)
    for mode in data["fault_modes"]:
        mode["observable_by"] = [c for c in mode["observable_by"] if c != "cavitation_index"]
    return data


def _tp300(line: str) -> dict[str, Any]:
    return {"doc_id": "tp300_manual", "page": 1, "quote": line}


def _tp300_channel(name: str, unit: str, line: str, span: tuple[float, float] | None,
                   role: str | None) -> dict[str, Any]:
    return {"name": name, "printed_name": name, "description": line, "role": "observation", "dtype": "float",
            "unit": unit, "range_min": span[0] if span else None, "range_max": span[1] if span else None,
            "range_evidence": _tp300(line) if span else None, "range_role": role, "categories": None,
            "sample_period": {"value": 10, "unit": "seconds", "evidence": _tp300("channels once every 10 seconds:"),
                              "role": "spec"},
            "evidence": [_tp300(line)], "secondary_role": None, "secondary_reason": None, "secondary_evidence": None}


def tp300_channels(bearing_scale_as_spec: bool = False) -> dict[str, Any]:
    """The value-roles case: a spec range, a variant range, and a screenshot-only scale."""
    bearing = ((0.0, 80.0), "spec") if bearing_scale_as_spec else (None, None)
    return {"channels": [
        _tp300_channel("suction_pressure", "bar", "suction_pressure (bar), measuring range -1 to 5", (-1.0, 5.0), "spec"),
        _tp300_channel("motor_current", "A",
                       "motor_current (A), measuring range 0 to 16 for the TP-300-15 and 0 to 25 for the TP-300-22",
                       (0.0, 16.0), "variant"),
        _tp300_channel("bearing_temperature", "degC", "bearing_temperature (degC)", *bearing)]}


GOOD = {"understand.components": _clean_components(), "understand.sensor_channels": channels(0),
        "understand.fault_modes": _clean_fault_modes()}


def _eval(tmp: Path, prompt_id: str, reply: dict[str, Any], tp300: dict[str, Any] | None = None) -> EvalResult:
    prompt = PromptRegistry().load(prompt_id)
    tp300 = tp300 if tp300 is not None else tp300_channels()

    def answer(request: dict[str, Any]) -> str:
        return json.dumps(tp300 if "TP-300" in json.dumps(request["messages"]) else reply)

    gateway = make_gateway(tmp, ScriptedTransport(answer))
    return run_eval(prompt, gateway, model="claude-haiku-5-5", effort=None, judge_prompt=None, judge_model=None,
                    source="offline", git_commit="test", now=lambda: FIXED_NOW)


@pytest.mark.parametrize("prompt_id", PROMPTS)
def test_good_outputs_pass_every_threshold(tmp_path: Path, prompt_id: str) -> None:
    result = _eval(tmp_path, prompt_id, GOOD[prompt_id])
    assert result.passed, [t for t in result.thresholds if not t.passed]
    assert all(t.value is not None for t in result.thresholds)  # every threshold has a metric behind it


def test_sensors_as_components_fail(tmp_path: Path) -> None:
    legacy = _clean_components()
    legacy["components"].append({"id": "filter_dp_sensor", "name": "filter dp sensor", "description": "sensor",
                                 "category": "sensor", "parent": None, "evidence": [q(1, "filter_dp (bar)")]})
    legacy["components"].append({"id": "pump_motor", "name": "pump motor", "description": "motor",
                                 "category": "motor", "parent": None, "evidence": [q(1, "driven by a 1.5 kW motor")]})
    result = _eval(tmp_path, "understand.components", legacy)
    failed = {t.metric for t in result.thresholds if not t.passed}
    assert failed == {"no_sensor_components", "top_level_only_main"}


def test_alarm_limits_as_channels_fail(tmp_path: Path) -> None:
    legacy = channels(0)
    legacy["channels"].append({**legacy["channels"][2], "name": "filter_dp_alarm", "range_min": 0.8, "range_max": 0.8,
                               "evidence": [q(2, 'Alarm A-22 "Filter blocked": filter_dp above 0.80 bar.')]})
    result = _eval(tmp_path, "understand.sensor_channels", legacy)
    failed = {t.metric for t in result.thresholds if not t.passed}
    assert failed == {"no_extra_channels"}


def test_eval_context_is_what_the_phase_sends() -> None:
    settings = KafulSettings()
    record = check_documents(TINY_PUMP, load_machine_config(TINY_PUMP)).records[0]
    text = render(chunk_document(read_document(TINY_PUMP, record, settings.ingest), settings.ingest))
    for prompt_id in PROMPTS:
        source = PromptRegistry().load(prompt_id).directory / "evals" / "sources" / "tp200_context.txt"
        assert source.read_text(encoding="utf-8") == text, prompt_id


def test_an_example_returned_as_a_spec_fails(tmp_path: Path) -> None:
    result = _eval(tmp_path, "understand.sensor_channels", channels(0), tp300_channels(bearing_scale_as_spec=True))
    failed = {t.metric for t in result.thresholds if not t.passed}
    assert failed == {"examples_not_specs"}


def test_value_role_metrics_score_the_value_roles_case(tmp_path: Path) -> None:
    result = _eval(tmp_path, "understand.sensor_channels", GOOD["understand.sensor_channels"])
    assert (result.metric("spec_recall"), result.metric("variant_found"), result.metric("examples_not_specs")) == (
        1.0, 1.0, 1.0)
