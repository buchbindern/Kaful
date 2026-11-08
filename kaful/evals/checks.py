"""Deterministic checks and golden comparators. Every score is in [0, 1].

The judge is used only by `quote_in_source`, and only for quotes code cannot place
in the source (a possible paraphrase).
"""

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from kaful.evals.models import CheckOutcome, CheckSpec, GoldenSpec, QuoteClass
from kaful.evidence import LINE_BREAK_HYPHEN
from kaful.units import UnitError, parse_unit

# Judge: (claim, source) -> verdict ("supported" | "partial" | "unsupported").
JudgeFn = Callable[[str, str], str]
MAX_DETAILS = 20


FILTER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\[(?:(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\]]*))?\]$")


def resolve(data: Any, path: str) -> list[Any]:
    """Values at `path`: 'a.b' -> [a.b]; 'items[].name' -> every item's name;
    'items[kind=x].name' -> the name of every item whose `kind` is 'x'."""
    values = [data]
    for part in path.split("."):
        match = FILTER_RE.match(part)
        name = match.group("name") if match else part
        found: list[Any] = []
        for value in values:
            item = value.get(name) if isinstance(value, dict) else None
            if match is None:
                found.append(item)
                continue
            key = match.group("key")
            found.extend(i for i in (item or []) if key is None or (isinstance(i, dict) and str(i.get(key)) == match.group("value")))
        values = found
    return values


def _ws(text: str) -> str:
    return " ".join(text.split())


def _loose(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text.casefold()).split())


def classify_quote(quote: str, source: str) -> QuoteClass:
    """verbatim: in the source up to whitespace; near: up to case and punctuation. A word hyphenated at a
    line break counts joined, as in the pipeline's rule-4 checker (kaful.evidence, matcher-v3)."""
    source = LINE_BREAK_HYPHEN.sub("", source)
    if _ws(quote) and _ws(quote) in _ws(source):
        return "verbatim"
    if _loose(quote) and _loose(quote) in _loose(source):
        return "near"
    return "unsupported"


def run_check(spec: CheckSpec, output: Any, inputs: Mapping[str, str],
              judge: JudgeFn | None) -> tuple[CheckOutcome, dict[str, float]]:
    """Run one check on a validated output (as JSON data). Returns outcome and metric scores."""
    values = resolve(output, spec.field)
    if spec.kind == "quote_in_source":
        return _quotes(spec, values, inputs[spec.source or ""], judge)
    if spec.kind == "in_range":
        bad = [v for v in values if not _within(v, spec.min, spec.max)]
        score = 1.0 - len(bad) / len(values) if values else 0.0
        return _outcome(spec.name, not bad and bool(values), [f"out of range: {v!r}" for v in bad]), {spec.name: score}
    if spec.kind == "numbers_in_source":
        return _numbers(spec, values, inputs[spec.source or ""])
    if spec.kind == "listed_in_source":
        listed = set(re.findall(r"[a-z][a-z0-9_]*", inputs[spec.source or ""]))
        unknown = [v for v in values if v not in listed]
        score = 1.0 - len(unknown) / len(values) if values else 1.0
        return _outcome(spec.name, not unknown, [f"not listed: {v!r}" for v in unknown]), {spec.name: score}
    if spec.kind == "count_between":
        count = len(values[0] or []) if values else 0
        ok = _within(count, spec.min, spec.max)
        return _outcome(spec.name, ok, [] if ok else [f"count {count}"]), {spec.name: float(ok)}
    return _units(spec, values)


def _units(spec: CheckSpec, values: list[Any]) -> tuple[CheckOutcome, dict[str, float]]:
    """Share of the printed units (nulls skipped) that pint parses."""
    units = [v for v in values if v is not None]
    if not units:
        return CheckOutcome(check=spec.name, status="skip", details=("no units in the output",)), {}
    bad = []
    for unit in units:
        try:
            parse_unit(str(unit))
        except UnitError as exc:
            bad.append(str(exc))
    return _outcome(spec.name, not bad, bad), {spec.name: 1.0 - len(bad) / len(units)}


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(spec: CheckSpec, values: list[Any], source: str) -> tuple[CheckOutcome, dict[str, float]]:
    """Share of the numbers in the output that also appear in the source (as printed).

    No numbers in the output gives no metric: there is nothing to check."""
    available = set(NUMBER_RE.findall(source))
    found = [n for v in values for n in NUMBER_RE.findall(str(v or ""))]
    if not found:
        return CheckOutcome(check=spec.name, status="skip", details=("no numbers in the output",)), {}
    missing = sorted({n for n in found if n not in available})
    score = sum(1 for n in found if n in available) / len(found)
    return _outcome(spec.name, not missing, [f"not in source: {n}" for n in missing]), {spec.name: score}


def _within(value: Any, low: float | None, high: float | None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def _outcome(name: str, ok: bool, details: list[str]) -> CheckOutcome:
    return CheckOutcome(check=name, status="pass" if ok else "fail", details=tuple(details[:MAX_DETAILS]))


def _quotes(spec: CheckSpec, quotes: list[Any], source: str,
            judge: JudgeFn | None) -> tuple[CheckOutcome, dict[str, float]]:
    verbatim = supported = 0
    details = []
    for quote in quotes:
        cls = classify_quote(str(quote or ""), source)
        if cls == "verbatim":
            verbatim += 1
            supported += 1
            continue
        if cls == "near":
            supported += 1
            details.append(f"near: {quote!r}")
            continue
        verdict = judge(str(quote), source) if (spec.judge and judge is not None and quote) else "unchecked"
        supported += verdict == "supported"
        details.append(f"not in source (judge: {verdict}): {quote!r}")
    n = len(quotes)
    scores = {f"{spec.name}.verbatim": verbatim / n if n else 0.0,
              f"{spec.name}.supported": supported / n if n else 0.0}
    return _outcome(spec.name, n > 0 and supported == n, details), scores


# ── Golden ────────────────────────────────────────────────────────────────────

def _norm(value: Any) -> str:
    return _ws(str(value)).casefold()


def _same(got: Any, expected: Any, rel_tol: float) -> bool:
    numbers = (int, float)
    if isinstance(expected, numbers) and not isinstance(expected, bool):
        if not isinstance(got, numbers) or isinstance(got, bool):
            return False
        return abs(got - expected) <= rel_tol * abs(expected) if rel_tol else got == expected
    if expected is None:
        return got is None
    return got is not None and _norm(got) == _norm(expected)


def _flatten(value: Any) -> str:
    """Text of a value for contains_all: list items and object fields joined with ' | '."""
    if isinstance(value, dict):
        return " | ".join(_flatten(v) for v in value.values() if v is not None)
    if isinstance(value, list):
        return " | ".join(_flatten(v) for v in value)
    return "" if value is None else str(value)


def score_golden(spec: GoldenSpec, output: Any) -> float:
    """Score one golden expectation against a validated output."""
    values = resolve(output, spec.field)
    if spec.compare == "set_f1":
        got, want = {_norm(v) for v in values}, {_norm(v) for v in spec.expected}
        hits = len(got & want)
        if not hits:
            return 0.0
        precision, recall = hits / len(got), hits / len(want)
        return 2 * precision * recall / (precision + recall)
    if spec.compare == "not_in":  # every value found is outside `expected`; absence passes
        banned = {_norm(v) for v in spec.expected}
        return sum(_norm(v) not in banned for v in values) / len(values) if values else 1.0
    if spec.compare == "within":
        allowed = {_norm(v) for v in spec.expected}
        return sum(_norm(v) in allowed for v in values) / len(values) if values else 1.0
    if spec.compare == "excludes_all":
        text = f" {_loose(chr(10).join(_flatten(v) for v in values))} "
        forbidden: list[str] = spec.expected
        return sum(1 for w in forbidden if f" {_loose(w)} " not in text) / len(forbidden) if forbidden else 1.0
    if spec.compare == "contains_all":
        text = f" {_loose(chr(10).join(_flatten(v) for v in values))} "
        wanted: list[str] = spec.expected
        return sum(1 for w in wanted if f" {_loose(w)} " in text) / len(wanted) if wanted else 1.0
    if spec.compare == "keyed":
        rows = values[0] if len(values) == 1 and isinstance(values[0], list) else values
        items = {_norm(item.get(spec.key)): item.get(spec.value) for item in rows if isinstance(item, dict)}
        expected: dict[str, Any] = spec.expected
        matched = sum(_norm(k) in items and _same(items[_norm(k)], v, spec.rel_tol) for k, v in expected.items())
        return matched / len(expected)
    value = values[0] if values else None
    if spec.compare == "violation":
        return float(value is not None and _norm(value) == _norm(spec.expected))
    if spec.compare == "exact":
        return float(json.dumps(value, sort_keys=True) == json.dumps(spec.expected, sort_keys=True))
    return float(_same(value, spec.expected, spec.rel_tol))
