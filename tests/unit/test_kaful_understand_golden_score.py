"""Scoring an understand run against golden labels (offline, deterministic): matching by name or
alias, unmatched outputs counted against precision, fault modes matched by page or words."""

from pathlib import Path
from typing import Any

from kaful.artifacts.base import Evidence
from kaful.artifacts.machine_model import Component, FaultMode, MachineModel
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.artifacts.understand_report import ContextSelection, ItemCheck, UnderstandReport
from kaful.config import UnderstandSettings
from kaful.phases.p2_understand.tools.golden_score import score

GOLDEN = """schema_version = 1
machine_id = "spindle"
status = "reviewed"

[[legacy_fields]]
name = "spindle_load"
aliases = ["load_meter"]
verdict = "channel"
role = "operating_input"
secondary_role = "observation"
note = ""

[[legacy_fields]]
name = "motor_temperature"
verdict = "channel"
role = "observation"
secondary_role = ""
note = ""

[[legacy_fields]]
name = "max_speed"
verdict = "not_channel"
role = ""
secondary_role = ""
note = ""

[[legacy_fields]]
name = "sync_error"
verdict = "unsure"
role = ""
secondary_role = ""
note = ""

[[candidate_channels]]
name = "operating_hours"
verdict = "channel"
role = "usage_clock"
secondary_role = ""
note = ""

[[components]]
id = "spindle_motor"
verdict = "top_level"
parent = ""
note = ""

[[components]]
id = "bearings"
verdict = "child"
parent = "spindle_motor"
note = ""

[[components]]
id = "housing"
verdict = "not_a_component"
parent = ""
note = ""

[[fault_modes]]
component = "bearings"
what_degrades = "grease"
verdict = "yes"
doc_id = "nsk"
page = "53"
page_label = "52"
note = ""

[[fault_modes]]
component = "bearings"
what_degrades = "cage material"
verdict = "yes"
doc_id = ""
page = ""
page_label = ""
note = ""

[[fault_modes]]
component = "spindle_motor"
what_degrades = "position accuracy"
verdict = "no"
doc_id = ""
page = ""
page_label = ""
note = "an effect"

[[values]]
doc_id = "fanuc"
page = 9
value = "0 to 120 %"
quote = "Load meter 0 to 120 %"
suggested_role = "spec"
role = "example"
note = ""
"""

PROV: dict[str, Any] = {"source": "llm_judgement", "ref": "x", "confidence": 0.5}
SETTINGS = UnderstandSettings()


def _ev(doc: str, page: int, quote: str = "q", **more: Any) -> Evidence:
    return Evidence(doc_id=doc, page=page, quote=quote, **more)


def _channel(name: str, role: str, **more: Any) -> SensorChannel:
    return SensorChannel(name=name, description="d", role=role, dtype="float", unit="%", sampling="unknown",
                         **{**PROV, **more})


def _artifacts() -> tuple[MachineModel, SensorContract, UnderstandReport]:
    base = {"run_id": "r1", "machine_id": "spindle"}
    second = {"role": "observation", "reason": "rises with wear", **PROV}
    contract = SensorContract(**base, usage_clock=None, channels=(
        _channel("load_meter", "operating_input", secondary_roles=(second,), range=(0.0, 120.0),
                 range_evidence=_ev("fanuc", 9, "Load meter 0 to 120 % shown", value_role="spec")),
        _channel("max_speed", "operating_input"),
        _channel("motor_temperature", "operating_input"),
        _channel("vibration", "observation"),
    ))
    model = MachineModel(**base, n_extractions=3, components=(
        Component(id="bearings", name="b", description="", category="part", parent="housing", **PROV),
        Component(id="housing", name="h", description="", category="part", **PROV),
        Component(id="spindle_motor", name="m", description="", category="motor", **PROV),
    ), fault_modes=(
        FaultMode(id="bearings_lubricant", component="bearings", name="Lubricant breakdown", mechanism="degradation",
                  effect="heat", **{**PROV, "evidence": (_ev("nsk", 53),)}),
        FaultMode(id="cage_wear", component="bearings", name="Cage material wear", mechanism="wear", effect="noise",
                  **PROV),
        FaultMode(id="motor_accuracy", component="spindle_motor", name="Loss of position accuracy",
                  mechanism="drift", effect="scrap", **PROV),
        FaultMode(id="seal_leak", component="bearings", name="Seal leak", mechanism="leak", effect="dirt", **PROV),
    ))
    report = UnderstandReport(**base, context=ContextSelection(mode="whole", chunk_ids=("c1",), estimated_tokens=10,
                                                                context_sha256="0" * 64),
                              n_samples=3, prompts=("p v1 m",), items=(
        ItemCheck(kind="channel", id="load_meter", agreement=1.0, evidence=("near", "verbatim"), source="manual",
                  confidence=1.0),
        ItemCheck(kind="channel", id="vibration", agreement=1.0, evidence=(), source="llm_judgement", confidence=0.0),
    ))
    return model, contract, report


def _score(tmp_path: Path) -> Any:
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN)
    return score(golden, *_artifacts(), SETTINGS)


def test_channels_match_by_name_or_alias_and_unreviewed_extras_count_against_precision(tmp_path: Path) -> None:
    channels = _score(tmp_path).channels
    assert channels.matched == ("load_meter -> spindle_load", "motor_temperature -> motor_temperature")
    assert channels.wrong == ("max_speed -> max_speed (not_channel)",)
    assert channels.unmatched == ("vibration",)
    assert channels.missed == ("operating_hours",)
    assert (channels.precision, channels.recall) == (0.5, round(2 / 3, 6))
    assert channels.role_errors == ("motor_temperature: operating_input, golden observation",)
    assert (channels.role_accuracy, channels.secondary_found) == (0.5, 1.0)


def test_components_score_parents_through_the_golden_ids(tmp_path: Path) -> None:
    components = _score(tmp_path).components
    assert (components.precision, components.recall) == (round(2 / 3, 6), 1.0)
    assert components.wrong == ("housing -> housing (not_a_component)",)
    assert components.parent_errors == ("bearings: parent housing, golden spindle_motor",)


def test_fault_modes_match_by_cited_page_or_words_and_effects_are_listed(tmp_path: Path) -> None:
    modes = _score(tmp_path).fault_modes
    assert modes.matched == ("bearings_lubricant -> bearings: grease", "cage_wear -> bearings: cage material")
    assert modes.wrong == ("motor_accuracy -> spindle_motor: position accuracy (no)",)
    assert modes.unmatched == ("seal_leak",)
    assert (modes.precision, modes.recall, modes.missed) == (0.5, 1.0, ())


def test_a_golden_example_stored_as_a_range_is_reported(tmp_path: Path) -> None:
    values = _score(tmp_path).values
    assert values.found == ("0 to 120 %: example -> spec",) and values.examples_stored == ("0 to 120 % (p9)",)


def test_grounding_counts_items_by_source_and_best_evidence(tmp_path: Path) -> None:
    grounding = _score(tmp_path).grounding
    assert grounding.by_source == {"channel": {"llm_judgement": 1, "manual": 1}}
    assert grounding.by_evidence == {"channel": {"none": 1, "verbatim": 1}}


def test_component_name_words_do_not_make_a_fault_mode_match(tmp_path: Path) -> None:
    """Seen on eversys: 'grinder_blocking' matched 'ceramic_grinders: grinder burrs' on the word 'grinder' alone."""
    model, contract, report = _artifacts()
    blocking = FaultMode(id="bearings_blocking", component="bearings", name="Bearing blocked", mechanism="blockage",
                         effect="stops", **PROV)
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN.replace('what_degrades = "cage material"', 'what_degrades = "bearing cage"'))
    model = model.model_copy(update={"fault_modes": (blocking,)})
    modes = score(golden, model, contract, report, SETTINGS).fault_modes
    assert modes.matched == () and modes.unmatched == ("bearings_blocking",)


def test_a_golden_row_is_credited_once_and_duplicates_are_listed(tmp_path: Path) -> None:
    model, contract, report = _artifacts()
    twin = Component(id="bearing_set", name="b", description="", category="part", parent="spindle_motor", **PROV)
    model = model.model_copy(update={"components": (model.components[0], twin, *model.components[1:])})
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN.replace('id = "bearings"', 'id = "bearings"\naliases = ["bearing_set"]'))
    components = score(golden, model, contract, report, SETTINGS).components
    assert components.matched == ("bearing_set -> bearings", "spindle_motor -> spindle_motor")
    assert components.duplicates == ("bearings -> bearings",)
    assert components.precision == 0.5  # 2 credited of 4 output components


def test_every_disagreement_is_listed_for_adjudication(tmp_path: Path) -> None:
    result = _score(tmp_path)
    lines = {(d.kind, d.item, d.run, d.golden) for d in result.disagreements}
    assert ("channel", "max_speed", "a channel (operating_input)", "not_channel") in lines
    assert ("channel", "vibration", "a channel (observation)", "no row") in lines
    assert ("channel", "operating_hours", "not extracted", "channel (usage_clock)") in lines
    assert ("channel", "motor_temperature", "role operating_input", "role observation") in lines
    assert ("component", "bearings", "parent housing", "parent spindle_motor") in lines
    assert ("fault_mode", "motor_accuracy", "a fault mode", "no (spindle_motor: position accuracy)") in lines
    assert ("value", "0 to 120 %", "stored as a range (spec)", "example") in lines
    assert all(d.ref for d in result.disagreements if d.run != "not extracted")


def test_an_unanswered_row_is_not_scored(tmp_path: Path) -> None:
    """A candidate row waiting for a verdict ('') counts like 'unsure': neither credit nor a wrong channel."""
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN.replace('name = "max_speed"\nverdict = "not_channel"', 'name = "max_speed"\nverdict = ""'))
    channels = score(golden, *_artifacts(), SETTINGS).channels
    assert channels.wrong == () and "max_speed" not in channels.unmatched


def test_a_golden_page_range_is_cited_by_any_page_in_it(tmp_path: Path) -> None:
    """Natalie moved CNC bearing fault modes to the NSK failure section, p60-61 (2026-11-07)."""
    from kaful.phases.p2_understand.tools.golden_score import pages_of

    assert (pages_of(""), pages_of(53), pages_of("53"), pages_of("60-61")) == ([], [53], [53], [60, 61])
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN.replace('page = "53"\npage_label = "52"', 'page = "52-54"\npage_label = "51-53"'))
    modes = score(golden, *_artifacts(), SETTINGS).fault_modes
    assert "bearings_lubricant -> bearings: grease" in modes.matched  # cites p53, inside 52-54


def test_a_fault_mode_row_matches_its_aliases_on_any_component(tmp_path: Path) -> None:
    """Natalie, 2026-11-08: eversys coffee_outlet_clogging (component coffee_outlet) is the coffee residue row
    (component brewing_chamber); the words and pages alone cannot tell."""
    model, contract, report = _artifacts()
    golden = tmp_path / "spindle.toml"
    golden.write_text(GOLDEN.replace('what_degrades = "cage material"', 'what_degrades = "cage material"\n'
                                     'aliases = ["seal_leak"]'))
    modes = score(golden, model, contract, report, SETTINGS).fault_modes
    assert "cage_wear -> bearings: cage material" in modes.matched
    assert modes.duplicates == ("seal_leak -> bearings: cage material",) and modes.unmatched == ()
