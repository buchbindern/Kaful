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
