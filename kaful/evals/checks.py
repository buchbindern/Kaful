"""Deterministic checks and golden comparators. Every score is in [0, 1].

The judge is used only by `quote_in_source`, and only for quotes code cannot place
in the source (a possible paraphrase).
"""

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from kaful.evals.models import CheckOutcome, CheckSpec, GoldenSpec, QuoteClass

# Judge: (claim, source) -> verdict ("supported" | "partial" | "unsupported").
JudgeFn = Callable[[str, str], str]
MAX_DETAILS = 20


def resolve(data: Any, path: str) -> list[Any]:
    """Values at `path`: 'a.b' -> [a.b]; 'items[].name' -> every item's name."""
    values = [data]
    for part in path.split("."):
        spread = part.endswith("[]")
        name = part[:-2] if spread else part
        found: list[Any] = []
        for value in values:
            item = value.get(name) if isinstance(value, dict) else None
            if spread:
                found.extend(item or [])
            else:
                found.append(item)
        values = found
    return values


def _ws(text: str) -> str:
    return " ".join(text.split())


def _loose(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9.]+", " ", text.casefold()).split())


def classify_quote(quote: str, source: str) -> QuoteClass:
    """verbatim: in the source up to whitespace; near: up to case and punctuation."""
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
    if spec.kind == "count_between":
        count = len(values[0] or []) if values else 0
        ok = _within(count, spec.min, spec.max)
        return _outcome(spec.name, ok, [] if ok else [f"count {count}"]), {spec.name: float(ok)}
    # unit_parses needs pint, which M7 adds; the check is declared now and skipped.
    return CheckOutcome(check=spec.name, status="skip", details=("unit parsing is implemented in M7",)), {}


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
