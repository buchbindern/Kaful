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

EVERSYS = "eversys_legacy_user_manual"


def _ev(page: int, quote: str) -> dict[str, Any]:
    return {"doc_id": EVERSYS, "page": page, "quote": quote}


def _mode(component: str, mechanism: str, page: int, quote: str) -> dict[str, Any]:
    return {"component": component, "id": f"{component}_{mechanism}", "name": f"{component} {mechanism}",
            "mechanism": mechanism, "effect": "e", "driven_by": [], "observable_by": [], "evidence": [_ev(page, quote)]}


def eversys_fault_modes(grinder_jam: bool = False) -> dict[str, Any]:
    """The eversys gradual case: W-014 states filter exhaustion and boiler scaling; p14 names milk and coffee
    residue; p127 the steam wand blocking slowly."""
    w014, residue = "Predefined water amount has", "residue may accumulate in the machine, clog the outlets or get into drinks."
    modes = [_mode("water_filter", "exhaustion", 114, w014), _mode("coffee_boiler", "scaling", 114, w014),
             _mode("steam_boiler", "scaling", 114, w014), _mode("milk_system", "residue", 14, residue),
             _mode("coffee_outlet", "residue", 14, residue),
             _mode("steam_wand", "clogging", 127, "Not using the purge will slowly block the pipe inside.")]
    if grinder_jam:
        modes.append(_mode("ceramic_grinders", "blockage", 54, "blocked or damaged."))
    return {"carriers": [], "fault_modes": modes}


def _carrier(id_: str, quote: dict[str, Any]) -> dict[str, Any]:
    return {"id": id_, "name": id_.replace("_", " "), "description": "d", "category": "part", "parent": None,
            "evidence": [quote]}


def eversys_carrier_modes(carriers: bool = True) -> dict[str, Any]:
    """The carriers case (M7b): milk_system and water_filter are not in the given list, so their modes come with
    carriers quoted from p114 and p14; without carriers the modes cannot be given."""
    data = eversys_fault_modes()
    if carriers:
        data["carriers"] = [_carrier("water_filter", _ev(114, "the water filter.")),
                            _carrier("milk_system", _ev(114, "Refill the milk tank [55]."))]
    else:
        data["fault_modes"] = [m for m in data["fault_modes"] if m["component"] not in ("water_filter", "milk_system")]
    return data


def tp200_carrier_modes() -> dict[str, Any]:
    """The tiny_pump carrier case: the inlet filter F1 is not in the given list."""
    data = _clean_fault_modes()
    data["carriers"] = [_carrier("inlet_cartridge_filter_f1", q(2, "F1 is a 20 micron pleated cartridge."))]
    return data


def eversys_components(water_filter: bool = True) -> dict[str, Any]:
    def part(id_: str, page: int, quote: str) -> dict[str, Any]:
        return {"id": id_, "name": id_.replace("_", " "), "description": "d", "category": "part", "parent": None,
                "evidence": [_ev(page, quote)]}
    parts = [part("coffee_boiler", 114, "Coffee boiler")] + ([part("water_filter", 114, "the water filter.")]
                                                             if water_filter else [])

    return {"components": parts, "boundaries": [], "flow_paths": []}


def eversys_channels(counters_role: str = "operating_input") -> dict[str, Any]:
    def counter(name: str, role: str, quote: str) -> dict[str, Any]:
        return {"name": name, "printed_name": name, "description": "d", "role": role, "dtype": "int", "unit": "count",
                "range_min": None, "range_max": None, "range_evidence": None, "range_role": None, "categories": None,
                "sample_period": None, "evidence": [_ev(71, quote)], "secondary_role": None, "secondary_reason": None,
                "secondary_evidence": None}
    return {"channels": [counter("coffee_counter", counters_role, "Coffee Counter"),
                         counter("milk_counter", counters_role, "Milk Counter"),
                         counter("total_counter", "usage_clock", "Total Counter")]}


def eversys_modules(milk: bool = True) -> dict[str, Any]:
    """The module column of the p102-103 error list, as top-level components."""
    names = [m for m in ("hydraulic", "milk", "brewing", "grinder", "steam") if milk or m != "milk"]
    return {"components": [{"id": f"{m}_module", "name": f"{m} module", "description": "d", "category": "module",
                            "parent": None, "evidence": [_ev(102 if m in ("hydraulic", "milk", "brewing") else 103,
                                                             f"{m.capitalize()} module")]} for m in names],
            "boundaries": [], "flow_paths": []}


EVERSYS_GOOD = {"understand.components": eversys_components(), "understand.sensor_channels": eversys_channels(),
                "understand.fault_modes": eversys_fault_modes()}


def _eval(tmp: Path, prompt_id: str, reply: dict[str, Any], tp300: dict[str, Any] | None = None,
          eversys: dict[str, Any] | None = None, modules: dict[str, Any] | None = None,
          carried: dict[str, Any] | None = None) -> EvalResult:
    prompt = PromptRegistry().load(prompt_id)
    tp300 = tp300 if tp300 is not None else tp300_channels()
    eversys = eversys if eversys is not None else EVERSYS_GOOD[prompt_id]
    modules = modules if modules is not None else eversys_modules()
    carried = carried if carried is not None else eversys_carrier_modes()

    def answer(request: dict[str, Any]) -> str:
        sent = json.dumps(request["messages"])
        if "Error messages (E-XXX)" in sent and prompt_id == "understand.components":
            return json.dumps(modules)
        if prompt_id == "understand.fault_modes" and "- wear_ring" in sent and "- inlet_cartridge_filter_f1" not in sent:
            return json.dumps(tp200_carrier_modes())
        if prompt_id == "understand.fault_modes" and EVERSYS in sent and "- milk_system" not in sent:
            return json.dumps(carried)
        return json.dumps(tp300 if "TP-300" in sent else eversys if EVERSYS in sent else reply)

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


@pytest.mark.parametrize(("prompt_id", "reply", "metric"), [
    ("understand.fault_modes", eversys_fault_modes(grinder_jam=True), "no_sudden_faults"),
    ("understand.components", eversys_components(water_filter=False), "consumable_found"),
    ("understand.sensor_channels", eversys_channels(counters_role="observation"), "counter_roles"),
])
def test_the_eversys_misses_fail_their_metric(tmp_path: Path, prompt_id: str, reply: dict[str, Any],
                                              metric: str) -> None:
    """2026-11-08: what the eversys whole-manual run did (a sudden jam as a fault mode, no water filter,
    counters as observations) fails the metric written for it, and only that one."""
    result = _eval(tmp_path, prompt_id, GOOD[prompt_id], eversys=reply)
    assert {t.metric for t in result.thresholds if not t.passed} == {metric}


def test_a_module_of_the_error_list_left_out_fails_module_components(tmp_path: Path) -> None:
    """eversys run 20261108T192349Z-21c0ec never extracted the milk system (p102 (label 102) names the Milk module)."""
    result = _eval(tmp_path, "understand.components", GOOD["understand.components"],
                   modules=eversys_modules(milk=False))
    assert {t.metric for t in result.thresholds if not t.passed} == {"module_components"}


def test_carrier_modes_without_their_carriers_fail(tmp_path: Path) -> None:
    """M7b: a components step that missed milk_system and water_filter leaves their modes nowhere to sit; a reply
    that drops them (as fault_modes v3 had to) fails the carrier metrics."""
    result = _eval(tmp_path, "understand.fault_modes", GOOD["understand.fault_modes"],
                   carried=eversys_carrier_modes(carriers=False))
    failed = {t.metric for t in result.thresholds if not t.passed}
    assert {"carrier_found", "carrier_modes"} <= failed
