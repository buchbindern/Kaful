"""Value roles (kaful.value_roles): code only downgrades the LLM's proposal to `example`, and
only spec, default, setting_range (and declared variants) may set a prior."""

import pytest

from kaful.artifacts.base import Evidence
from kaful.artifacts.document_index import PageLayer, PageSignals, PageText
from kaful.config import UnderstandSettings
from kaful.value_roles import may_set_prior, settle_role

SETTINGS = UnderstandSettings()
SIGNALS = PageSignals(chars=100, image_frac=0.0, drawings=0, curves=0, lines=5, short_line_frac=0.0,
                      numeric_line_frac=0.0, garble_ratio=0.0, ruled_tables=0)
SCREEN = ("Figure (screenshot): Products menu\nScreenshot of the Products and Keyboard screen.\n"
          "Text in figure: Relax Time; 2,0; [s]; Pre-Infusion; 0,8; [s]")
SCANNED_TABLE = "Table (image): Rated values\n| Rated flow | 6 m3/h |"


def _page(raw: str, layer: str = "") -> PageText:
    layers = (PageLayer(kind="llm", text=layer, producer="ingest.extract_page v3"),) if layer else ()
    return PageText(doc_id="manual", page=69, label="67", text=raw, page_class="figure", signals=SIGNALS, layers=layers)


def _ev(quote: str, verification: str) -> Evidence:
    return Evidence(doc_id="manual", page=69, quote=quote, verification=verification)  # type: ignore[arg-type]


def test_a_value_only_in_a_screenshot_is_an_example() -> None:
    page = _page("6.2. Use of the software menu", SCREEN)
    assert settle_role("spec", _ev("Relax Time; 2,0; [s]", "in_layer"), page, SETTINGS) == (
        "example", "found only in a screenshot or photo")


def test_a_value_only_in_a_scanned_table_keeps_its_role() -> None:
    """Scanned pages have all their text in the LLM layer: that alone does not make a value an example."""
    page = _page("", SCANNED_TABLE)
    assert settle_role("spec", _ev("Rated flow 6 m3/h", "in_layer"), page, SETTINGS) == ("spec", None)


def test_a_value_after_an_example_marker_is_an_example() -> None:
    page = _page("Example: at a flow of 6 m3/h the filter dp rises by 0.05 bar per 100 h.\nRated flow 6 m3/h.")
    assert settle_role("spec", _ev("the filter dp rises by 0.05 bar per 100 h", "verbatim"), page, SETTINGS) == (
        "example", "quoted from a worked example")
    assert settle_role("spec", _ev("Rated flow 6 m3/h", "verbatim"), _page("Rated flow 6 m3/h."), SETTINGS) == (
        "spec", None)


def test_code_never_upgrades_an_example() -> None:
    page = _page("Rated flow 6 m3/h.")
    assert settle_role("example", _ev("Rated flow 6 m3/h", "verbatim"), page, SETTINGS) == ("example", None)


@pytest.mark.parametrize(("role", "declared", "allowed"), [
    ("spec", False, True), ("default", False, True), ("setting_range", False, True),
    ("variant", False, False), ("variant", True, True), ("example", True, False), (None, True, False)])
def test_which_roles_may_set_a_prior(role: str | None, declared: bool, allowed: bool) -> None:
    assert may_set_prior(role, declared) is allowed  # type: ignore[arg-type]
