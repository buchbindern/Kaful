"""Printed units parse with pint; only exponent and product spelling is rewritten."""

import pytest

from kaful.units import UnitError, dimensionality, parse_unit, pint_spelling, same_dimension, to_seconds


@pytest.mark.parametrize(("printed", "spelled", "dims"), [
    ("m3/h", "m**3/h", "[length] ** 3 / [time]"),
    ("min-1", "min**-1", "1 / [time]"),
    ("N-m", "N*m", "[mass] * [length] ** 2 / [time] ** 2"),
    ("kgf/cm2", "kgf/cm**2", "[mass] / [length] / [time] ** 2"),
    ("degC", "degC", "[temperature]"),
    ("mm/s", "mm/s", "[length] / [time]"),
    ("tick", "tick", "dimensionless"),
    ("pulse", "pulse", "dimensionless"),  # Fanuc p200 (label 176), p128 (label 104): encoder pulses
    ("Pulse", "Pulse", "dimensionless"),
    ("bar(g)", "barg", "[mass] / [length] / [time] ** 2"),
    ("psia", "psia", "[mass] / [length] / [time] ** 2"),
    ("1", "1", "dimensionless"),
])
def test_printed_units_parse(printed: str, spelled: str, dims: str) -> None:
    assert pint_spelling(printed) == spelled
    assert dimensionality(printed) == dims


@pytest.mark.parametrize("printed", ["", "  ", "1638.4 min-1/V", "furlongz"])
def test_unparseable_units_raise(printed: str) -> None:
    with pytest.raises(UnitError):
        parse_unit(printed)


def test_dimensions_and_periods() -> None:
    assert same_dimension("bar", "psig") and not same_dimension("bar", "m3/h")
    assert to_seconds(10, "seconds") == 10.0 and to_seconds(2, "min") == 120.0
    with pytest.raises(UnitError, match="not a time unit"):
        to_seconds(1, "bar")


@pytest.mark.parametrize("printed", ["bar", "m3/h", "degC", "A", "mm/s", "N-m", "%", "Hz", "rpm", "bar(g)"])
def test_spelling_is_idempotent(printed: str) -> None:
    once = pint_spelling(printed)
    assert pint_spelling(once) == once and parse_unit(once) == parse_unit(printed)


@pytest.mark.parametrize(("printed", "same_as"), [
    ("ºC", "degC"),  # the ordinal sign (CDV run 20261109T143015Z-596dd5, m220 p90 'Spike Temp 2')
    ("˚C", "degC"),  # ring above: pint read it as coulomb
    ("° C", "degC"),  # pint read it as coulomb x degree
    ("deg C", "degC"),
    ("℃", "degC"),  # pint crashed on it (AssertionError)
    ("℉", "degF"),
    ("°F", "degF"),
    ("Torr", "torr"),
    ("sccm", "cm**3/min"),  # standard flows: a volume per time, standard conditions stay in the printed string
    ("slm", "l/min"),
    ("slpm", "l/min"),
])
def test_look_alike_and_standard_spellings(printed: str, same_as: str) -> None:
    """M7c (2026-11-09): look-alike degree signs, Torr and standard flows parse as what they mean."""
    assert parse_unit(printed) == parse_unit(same_as)


def test_standard_flows_compare_with_volume_flows() -> None:
    assert same_dimension("sccm", "l/min") and same_dimension("slm", "m3/h")


@pytest.mark.parametrize("printed", ["℃/s", "kº", "deg"])
def test_parser_crashes_are_unit_errors(printed: str) -> None:
    """Whatever pint raises, a bad unit is a UnitError (the gate), never a crash."""
    try:
        parse_unit(printed)
    except UnitError:
        pass
