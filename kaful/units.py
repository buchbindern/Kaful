"""Unit parsing with pint. Units are kept as printed in artifacts and parsed here.

Only the spelling of exponents and products is rewritten before parsing, never the
unit's name or a value's sign (the legacy pipeline ran units through name
normalization, which corrupted them):
- a digit glued to a unit symbol is a power: `m3/h` -> `m**3/h`, `mm2` -> `mm**2`;
- a hyphen before a digit is a negative power: `min-1` -> `min**-1`;
- a hyphen between two symbols is a product: `N-m` -> `N*m`.
Counters printed as units in equipment manuals (`tick`, `cycle`, `count`, `pulse`) are
defined as dimensionless, capitalized as well (Fanuc prints `Pulse`). Gauge and absolute pressures (`barg`, `bar(g)`, `psia`, ...) are
pressures; which one was printed stays in the printed string. Pint reads `bar(g)` as
bar x gram, so the `(g)`/`(a)` suffix is rewritten first.

Look-alike signs are respelled (M7c, 2026-11-09): the ordinal sign `º` and the ring above `˚` are
the degree sign; `° C` and `deg C` are `degC` (pint read them as coulomb x degree); `℃` and `℉` are
`degC` and `degF` (pint crashed on them). `Torr` is torr. Standard flows (`sccm`, `slm`, `slpm`) are a
volume per time; the standard conditions stay in the printed string, as for gauge pressures (pint's
own `slm` has the dimensions of power, so it never compared with l/min).

Known ambiguity, not resolved here: `Nm3` (normal cubic metre) parses as a
"number_meter" cubed. Callers must not trust a parse alone: check the dimension.
"""

import re
from functools import lru_cache

import pint

COUNTER_UNITS = ("tick", "cycle", "count", "stroke", "pulse")
PRESSURE_REFERENCES = {"barg": "bar", "bara": "bar", "psig": "psi", "psia": "psi"}
GAUGE_SUFFIX_RE = re.compile(r"\b(bar|psi)\s*\(([ga])\)")
POWER_RE = re.compile(r"(?<=[A-Za-zµμ])([23])(?![0-9.])")
NEG_POWER_RE = re.compile(r"(?<=[A-Za-zµμ])-(?=\d)")
PRODUCT_RE = re.compile(r"(?<=[A-Za-zµμ])-(?=[A-Za-zµμ])")
DEGREE_LOOKALIKES = str.maketrans({"º": "°", "˚": "°"})
DEGREE_SYMBOLS = {"℃": "degC", "℉": "degF"}
SPACED_DEGREE_RE = re.compile(r"(?:°|\bdeg)\s+([CFK])\b")
STANDARD_FLOWS = {"sccm": "cm**3/min", "slm": "l/min", "slpm": "l/min"}
STANDARD_FLOW_RE = re.compile(r"\b(" + "|".join(sorted(STANDARD_FLOWS)) + r")\b", re.IGNORECASE)
MILLITORR_RE = re.compile(r"\bmtorr\b", re.IGNORECASE)  # a general vacuum unit, printed in any case (mTORR)


class UnitError(ValueError):
    """A printed unit that pint cannot parse."""


@lru_cache(maxsize=1)
def registry() -> "pint.UnitRegistry[float]":
    reg: pint.UnitRegistry[float] = pint.UnitRegistry()
    for name in COUNTER_UNITS:
        reg.define(f"{name} = [] = {name}s = {name.capitalize()} = {name.capitalize()}s")
    for name, base in sorted(PRESSURE_REFERENCES.items()):
        reg.define(f"{name} = {base}")
    reg.define("@alias torr = Torr")
    return reg


def pint_spelling(printed: str) -> str:
    """The printed unit with exponents and products spelled the way pint reads them."""
    text = printed.strip().translate(DEGREE_LOOKALIKES)
    for sign, name in sorted(DEGREE_SYMBOLS.items()):
        text = text.replace(sign, name)
    text = SPACED_DEGREE_RE.sub(r"deg\1", text)
    text = STANDARD_FLOW_RE.sub(lambda m: f"({STANDARD_FLOWS[m.group(1).lower()]})", text)
    text = MILLITORR_RE.sub("millitorr", text)
    text = GAUGE_SUFFIX_RE.sub(r"\1\2", text)
    text = NEG_POWER_RE.sub("**-", text)
    text = POWER_RE.sub(r"**\1", text)
    return PRODUCT_RE.sub("*", text)


def parse_unit(printed: str) -> pint.Unit:
    """Parse a printed unit; '1' is dimensionless. Raises UnitError."""
    if not printed.strip():
        raise UnitError("empty unit")
    try:
        unit = registry().parse_units(pint_spelling(printed))
    except (pint.errors.PintError, TypeError, ValueError, AttributeError, AssertionError) as exc:
        raise UnitError(f"cannot parse unit {printed!r}: {exc}") from exc
    if not isinstance(unit, pint.Unit):
        raise UnitError(f"{printed!r} is not a unit")
    return unit


def dimensionality(printed: str) -> str:
    """A stable text form of the unit's dimensions, e.g. '[length] ** 3 / [time]'."""
    return str(parse_unit(printed).dimensionality)


def same_dimension(a: str, b: str) -> bool:
    return parse_unit(a).dimensionality == parse_unit(b).dimensionality


def to_seconds(value: float, printed: str) -> float:
    """`value` in `printed` time units, in seconds."""
    quantity = registry().Quantity(value, parse_unit(printed))
    try:
        return float(quantity.to("second").magnitude)
    except pint.errors.DimensionalityError as exc:
        raise UnitError(f"{printed!r} is not a time unit") from exc
