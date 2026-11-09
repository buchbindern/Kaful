"""Offline understand helpers: the tiny_pump fixture machine and scripted extractions.

The scripted samples differ on purpose, to exercise merging and verification:
- components: sample 2 adds `pump_bearings` (agreement 1/3); `wear_ring` cites a
  paraphrase that is not on page 3 (downgraded);
- channels: the 7 documented channels with the stated 10 s period; sample 1 also
  returns `rated_flow` (an over-extraction, agreement 1/3) and sample 2 states the
  period in 's';
- fault modes: cavitation names a channel that does not exist (dropped reference).
"""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from kaful.config import REPO_ROOT, KafulSettings
from kaful.prompts.registry import LockEntry, PromptLock, PromptRegistry

TINY_PUMP = REPO_ROOT / "tests" / "fixtures" / "machines" / "tiny_pump"
MACHINE_ID = "tiny_pump"
DOC = "tp200_manual"
PROMPTS = ("ingest.extract_page", "understand.components", "understand.fault_modes", "understand.sensor_channels")


def make_tiny_pump(machines_dir: Path, instrumentation: str = "") -> Path:
    machine_dir = machines_dir / MACHINE_ID
    shutil.copytree(TINY_PUMP / "manual", machine_dir / "manual")
    (machine_dir / "machine.toml").write_text((TINY_PUMP / "machine.toml").read_text() + instrumentation)
    return machine_dir


def understand_settings(machines_dir: Path, n_samples: int = 3) -> KafulSettings:
    settings = KafulSettings(machines_dir=machines_dir)
    llm = settings.llm.model_copy(update={"token_estimate": "chars", "default_model": "claude-haiku-5-5"})
    ingest = settings.ingest.model_copy(update={"embedding_dimensions": 8})
    understand = settings.understand.model_copy(update={"n_samples": n_samples})
    return settings.model_copy(update={"llm": llm, "ingest": ingest, "understand": understand})


def locked_prompts() -> PromptLock:
    registry = PromptRegistry()
    prompts = [registry.load(pid) for pid in PROMPTS]
    return PromptLock(prompts={p.id: LockEntry(version=p.version, content_sha256=p.content_sha256) for p in prompts})


def q(page: int, quote: str) -> dict[str, Any]:
    return {"doc_id": DOC, "page": page, "quote": quote}


def _component(id: str, name: str, category: str, parent: str | None, page: int, quote: str) -> dict[str, Any]:
    return {"id": id, "name": name, "description": name, "category": category, "parent": parent,
            "evidence": [q(page, quote)]}


def components(sample: int) -> dict[str, Any]:
    items = [
        _component("inlet_cartridge_filter_f1", "inlet cartridge filter (F1)", "filter", None, 1,
                   "the inlet cartridge filter (F1), which removes particles before the pump;"),
        _component("booster_pump_p1", "booster pump (P1)", "centrifugal pump", None, 1,
                   "the booster pump (P1), a single-stage centrifugal pump driven by a 1.5 kW motor."),
        _component("impeller", "impeller", "impeller", "booster_pump_p1", 3,
                   "Its impeller and wear ring wear slowly in service"),
        _component("wear_ring", "wear ring", "seal", "booster_pump_p1", 3, "The wear ring is replaced at overhaul."),
    ]
    if sample == 2:
        items.append(_component("pump_bearings", "pump bearings", "bearing", "booster_pump_p1", 3,
                                "The pump bearings are sealed for life."))
    path = "Flow path: supply inlet -> inlet filter F1 -> booster pump P1 -> discharge outlet."
    flows = [{"from_node": a, "to_node": b, "medium": "fluid", "quantity": "water", "evidence": [q(1, path)]}
             for a, b in [("supply_inlet", "inlet_cartridge_filter_f1"),
                          ("inlet_cartridge_filter_f1", "booster_pump_p1"), ("booster_pump_p1", "discharge_outlet")]]
    return {"components": items, "boundaries": ["discharge_outlet", "supply_inlet"], "flow_paths": flows}


CHANNELS = [  # name, unit, high, role, printed line
    ("inlet_pressure", "bar", 10, "operating_input", "inlet_pressure (bar), range 0 to 10"),
    ("discharge_pressure", "bar", 10, "observation", "discharge_pressure (bar), range 0 to 10"),
    ("filter_dp", "bar", 2, "observation", "filter_dp (bar), differential pressure across F1, range 0 to 2"),
    ("flow_rate", "m3/h", 12, "operating_input", "flow_rate (m3/h), range 0 to 12"),
    ("motor_current", "A", 10, "observation", "motor_current (A), range 0 to 10"),
    ("motor_temperature", "degC", 120, "observation", "motor_temperature (degC), range 0 to 120"),
    ("vibration_rms", "mm/s", 20, "observation", "vibration_rms (mm/s), measured on the pump bearing housing, range 0 to 20"),
]
PERIOD_LINE = "The controller reports the following channels once every 10 seconds:"


NO_SECOND = {"secondary_role": None, "secondary_reason": None, "secondary_evidence": None}


def channels(sample: int) -> dict[str, Any]:
    period = {"value": 10, "unit": "s" if sample == 2 else "seconds", "evidence": q(1, PERIOD_LINE), "role": "spec"}
    items = [{"name": name, "printed_name": name, "description": line, "role": role, "dtype": "float", "unit": unit,
              "range_min": 0, "range_max": high, "range_evidence": q(1, line), "range_role": "spec",
              "categories": None, "sample_period": period, "evidence": [q(1, line)], **NO_SECOND}
             for name, unit, high, role, line in CHANNELS]
    if sample == 1:
        items.append({"name": "rated_flow", "printed_name": "Rated flow", "description": "Rated flow", "role":
                      "operating_input", "dtype": "float", "unit": "m3/h", "range_min": 2.0, "range_max": 8.0,
                      "range_evidence": q(1, "Rated flow: 6.0 m3/h (permitted range 2.0 to 8.0 m3/h)"),
                      "range_role": "spec", "categories": None, "sample_period": None,
                      "evidence": [q(1, "Rated flow: 6.0 m3/h (permitted range 2.0 to 8.0 m3/h)")], **NO_SECOND})
    return {"channels": items}


def fault_modes(_sample: int) -> dict[str, Any]:
    return {"fault_modes": [
        {"id": "inlet_cartridge_filter_f1_clogging", "component": "inlet_cartridge_filter_f1", "name": "Filter clogging",
         "mechanism": "clogging", "effect": "filter_dp rises with the volume filtered", "driven_by": ["flow_rate"],
         "observable_by": ["filter_dp", "inlet_pressure"], "evidence": [q(2, (
             "Particles collect on the cartridge during operation, so the differential pressure filter_dp rises "
             "with the volume filtered."))]},
        {"id": "impeller_wear", "component": "impeller", "name": "Impeller wear", "mechanism": "wear",
         "effect": "lowers discharge pressure and raises motor current", "driven_by": [],
         "observable_by": ["discharge_pressure", "motor_current"], "evidence": [q(3, (
             "Its impeller and wear ring wear slowly in service, which lowers discharge pressure and raises motor "
             "current at a given flow."))]},
        {"id": "impeller_cavitation", "component": "impeller", "name": "Cavitation damage",
         "mechanism": "cavitation", "effect": "raises vibration and accelerates impeller wear",
         "driven_by": ["inlet_pressure"], "observable_by": ["cavitation_index", "vibration_rms"], "evidence": [q(2, (
             "Below 0.5 bar inlet pressure the pump may cavitate, which raises vibration_rms and accelerates "
             "impeller wear."))]},
    ]}


SCRIPTS = {"list the machine's physical components": components,
           "list the channels the machine reports": channels,
           "list the ways the machine's components degrade": fault_modes}


class ScriptedUnderstanding:
    """A responder for ScriptedTransport: picks the script by the system text, and the
    sample by how many times that prompt was asked (samples are sent in order)."""

    def __init__(self) -> None:
        self.asked: Counter[str] = Counter()

    def __call__(self, request: dict[str, Any]) -> str:
        system = request["system"][0]["text"]
        for marker, script in SCRIPTS.items():
            if marker in system:
                sample = self.asked[marker]
                self.asked[marker] += 1
                return json.dumps(script(sample))
        raise AssertionError("unexpected prompt")
