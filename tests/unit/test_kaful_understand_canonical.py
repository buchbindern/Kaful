"""Near-duplicate component names across samples become one component (approved 2026-11-07),
and names that are different parts never do."""

from typing import Any

from kaful.phases.p2_understand.b_merge_samples import merge_components
from kaful.phases.p2_understand.helpers.canonical_names import canonical_ids
from kaful.prompts.understand.components.models import ComponentExtraction, ExtractedComponent


def _c(id: str, quote: str, page: int = 37, parent: str | None = None) -> ExtractedComponent:
    return ExtractedComponent(id=id, name=id.replace("_", " "), description="d", category="part", parent=parent,
                              evidence=[{"doc_id": "manual", "page": page, "quote": quote}])  # type: ignore[list-item]


def _reply(*components: ExtractedComponent, flows: list[dict[str, Any]] | None = None) -> ComponentExtraction:
    return ComponentExtraction(components=list(components), boundaries=[], flow_paths=flows or [])


def test_names_that_differ_only_in_form_or_paraphrase_become_one() -> None:
    """eversys, Haiku: brew_chamber, brewing_chamber and brewing_unit were three components for one part."""
    quote = "The brewing unit compacts the coffee in the chamber."
    samples = [[_c("brew_chamber", quote)], [_c("brewing_chamber", "The brewing chamber holds the coffee.", 38)],
               [_c("brewing_chamber", "The brewing chamber holds the coffee.", 38), _c("bean_hoppers", "Hoppers.")],
               [_c("brewing_unit", quote)]]
    ids, merges = canonical_ids(samples)
    assert {ids["brew_chamber"], ids["brewing_unit"], ids["brewing_chamber"]} == {"brewing_chamber"}
    assert merges == ("brewing_chamber <- brew_chamber (same name form), brewing_unit (same quote, manual p37)",)


def test_two_parts_listed_in_one_sample_never_merge() -> None:
    quote = "The left grinder and the right grinder grind the beans."
    samples = [[_c("left_grinder", quote), _c("right_grinder", quote)], [_c("left_grinder", quote)]]
    ids, merges = canonical_ids(samples)
    assert (ids["left_grinder"], ids["right_grinder"], merges) == ("left_grinder", "right_grinder", ())


def test_a_quote_that_names_both_parts_does_not_merge_them() -> None:
    """Coffee vs steam boiler: one sentence names both, and no sample lists both, but they are different parts."""
    quote = "The coffee boiler and the steam boiler are heated separately."
    ids, merges = canonical_ids([[_c("coffee_boiler", quote)], [_c("steam_boiler", quote)]])
    assert (ids["coffee_boiler"], ids["steam_boiler"], merges) == ("coffee_boiler", "steam_boiler", ())


def test_merged_components_carry_their_aliases_and_renamed_references() -> None:
    quote = "The brewing unit compacts the coffee."
    samples = [_reply(_c("coffee_boiler", "Coffee boiler."), _c("brewing_chamber", "The brewing chamber.", 38,
                                                                   parent="coffee_boiler")),
               _reply(_c("coffee_boiler", "Coffee boiler."), _c("brewing_unit", quote, parent="coffee_boiler"),
                      flows=[{"from_node": "brewing_unit", "to_node": "coffee_boiler", "medium": "fluid",
                              "quantity": None, "evidence": [{"doc_id": "manual", "page": 37, "quote": quote}]}]),
               _reply(_c("brewing_chamber", quote))]
    merged = merge_components(samples)
    by_id = {m.item.id: m for m in merged.components}
    assert sorted(by_id) == ["brewing_chamber", "coffee_boiler"] and by_id["brewing_chamber"].agreement == 1.0
    assert merged.aliases == {"brewing_chamber": ("brewing_unit",)}
    assert [f.item.from_node for f in merged.flow_paths] == ["brewing_chamber"]
    assert merged.merges == ("brewing_chamber <- brewing_unit (same quote, manual p37)",)


def test_a_generic_head_word_does_not_make_two_parts_different() -> None:
    """eversys whole manual (2026-11-07): grinder / grinder_module and milk_module / milk_system were kept apart."""
    samples = [[_c("grinder_module", "Grinder unit not connected.", 103), _c("milk_module", "Milk unit.", 102)],
               [_c("grinder", "Up to 4 x Grinders", 27), _c("milk_system", "For machines with a milk system.", 33)],
               [_c("grinder", "Ceramic burrs.", 32)]]
    ids, merges = canonical_ids(samples)
    assert (ids["grinder_module"], ids["milk_module"], ids["milk_system"]) == ("grinder", "milk_system", "milk_system")
    assert merges == ("grinder <- grinder_module (same name form)", "milk_system <- milk_module (same name form)")


def test_a_generic_head_word_never_joins_parts_one_sample_lists_together() -> None:
    samples = [[_c("milk_module", "Milk unit.", 102), _c("milk_system", "The milk system.", 33)], [_c("milk", "Milk.")]]
    ids, _ = canonical_ids(samples)
    assert ids["milk_module"] != ids["milk_system"]


def test_a_caption_naming_a_part_inside_a_longer_name_is_one_part() -> None:
    """eversys p26: 'Bean hopper with lid (Position 2)' names one part, not two."""
    caption = "Bean hopper with lid (Position 2) (standard)"
    samples = [[_c("bean_hopper", caption, 26)], [_c("bean_hopper", "Bean hoppers.", 136)],
               [_c("bean_hopper_with_lid_position_2", caption, 26)]]
    ids, merges = canonical_ids(samples)
    assert ids["bean_hopper_with_lid_position_2"] == "bean_hopper"
    assert merges == ("bean_hopper <- bean_hopper_with_lid_position_2 (same quote, manual p26)",)


def _fm(component: str, mechanism: str, quote: str, page: int = 14) -> Any:
    from kaful.prompts.understand.fault_modes.models import ExtractedFaultMode

    return ExtractedFaultMode(component=component, id=f"{component}_{mechanism.replace(' ', '_')}",
                              name=f"{component} {mechanism}", mechanism=mechanism, effect="e", driven_by=[],
                              observable_by=[], evidence=[{"doc_id": "manual", "page": page, "quote": quote}])  # type: ignore[list-item]


def test_fault_modes_named_differently_for_one_quote_merge_and_a_table_of_mechanisms_does_not() -> None:
    """eversys 2026-11-08: coffee_outlet clogging / residue buildup, both from the p14 sentence, were two modes.
    One NSK table that names flaking and scoring is two mechanisms, whichever samples give them."""
    from kaful.phases.p2_understand.b_merge_samples import merge_fault_modes
    from kaful.prompts.understand.fault_modes.models import FaultModeExtraction

    p14 = "residue may accumulate in the machine, clog the outlets or get into drinks."
    table = "Flaking: rolling fatigue. Scoring: poor lubrication."
    samples = [FaultModeExtraction(carriers=[], fault_modes=[_fm("coffee_outlet", "clogging", p14), _fm("bearing", "flaking", table, 61)]),
               FaultModeExtraction(carriers=[], fault_modes=[_fm("coffee_outlet", "residue buildup", p14),
                                                _fm("bearing", "scoring", table, 61)]),
               FaultModeExtraction(carriers=[], fault_modes=[_fm("coffee_outlet", "clogging", p14)])]
    merged, merges = merge_fault_modes(samples)
    assert [(m.item.component, m.item.mechanism, round(m.agreement, 2)) for m in merged] == [
        ("bearing", "flaking", 0.33), ("bearing", "scoring", 0.33), ("coffee_outlet", "clogging", 1.0)]
    assert merges == ("coffee_outlet clogging <- coffee_outlet residue buildup (same quote, manual p14)",)


def _named(id: str, name: str, quote: str, page: int = 7) -> ExtractedComponent:
    return ExtractedComponent(id=id, name=name, description="d", category="controller", parent=None,
                              evidence=[{"doc_id": "dpc", "page": page, "quote": quote}])  # type: ignore[list-item]


def test_an_abbreviation_joins_the_full_name_it_is_printed_with() -> None:
    """M7c (CDV run 20261109T143015Z-596dd5): the samples named one controller dpc, digital_process_controller_dpc
    and digital_process_controller, and the DTC likewise; each sample listed one of them, and the run kept them all."""
    together = "The Digital Process Controller (DPC) together with the Digital Temperature Controller (DTC)"
    samples = [[_named("digital_process_controller_dpc", "Digital Process Controller (DPC)", together),
                _named("digital_temperature_controller_dtc", "Digital Temperature Controller (DTC)",
                       "which is a highly accurate microprocessor-based controller")],
               [_named("dpc", "Digital Process Controller (DPC)", "The Digital Process Controller (DPC) has been "
                       "designed for high accuracy control"),
                _named("dtc", "Digital Temperature Controller (DTC)", together)],
               [_named("digital_process_controller", "Digital Process Controller (DPC)", together),
                _named("digital_temperature_controller", "Digital Temperature Controller (DTC)", together)]]
    ids, merges = canonical_ids(samples)
    assert {ids[i] for i in ("dpc", "digital_process_controller", "digital_process_controller_dpc")} == {
        "digital_process_controller"}
    assert {ids[i] for i in ("dtc", "digital_temperature_controller", "digital_temperature_controller_dtc")} == {
        "digital_temperature_controller"}
    assert any("abbreviation" in m for m in merges)


def test_initials_alone_do_not_join() -> None:
    """Initials that happen to match, with no text printing the full name followed by them, stay apart."""
    ids, merges = canonical_ids([[_named("ac", "AC", "AC input 200 V.")],
                                 [_named("air_compressor", "air compressor", "The air compressor runs.")]])
    assert (ids["ac"], ids["air_compressor"], merges) == ("ac", "air_compressor", ())


def test_a_generic_word_matches_only_a_single_built_component() -> None:
    from kaful.phases.p2_understand.helpers.canonical_names import generic_match

    assert generic_match("filters", ["inline_gas_filter", "mass_flow_controller"]) == "inline_gas_filter"
    assert generic_match("filters", ["inline_gas_filter", "oil_filter"]) is None  # two candidates: no join
    assert generic_match("oil_filter", ["inline_gas_filter"]) is None  # not one generic word
    assert generic_match("pump", ["vacuum_pump_system"]) == "vacuum_pump_system"  # heads dropped first
