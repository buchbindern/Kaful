"""Self-consistency: merge N sampled extractions into one list with agreement.

Items are matched across samples by a key (a component id, a channel name, a flow
path's ends and medium, a fault mode's component and mechanism). For each key:
- agreement = share of the N samples that produced it (kept, never used to discard);
- each field takes its most common value among those samples, ties going to the
  earliest sample (deterministic, independent of hash order);
- evidence quotes are the union over samples, deduplicated and sorted.
"""

import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kaful.evidence import tokens

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Merged(Generic[T]):
    key: tuple[str, ...]
    item: T
    agreement: float
    quotes: tuple[tuple[str, int, str], ...]  # (doc_id, page, quote), deduplicated and sorted


def consensus(values: Sequence[Any]) -> Any:
    """The most common value; ties go to the value seen first."""
    counts: dict[str, int] = {}
    first: dict[str, Any] = {}
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, value)
    winner = max(counts, key=lambda k: (counts[k], -list(first).index(k)))
    return first[winner]


def _quotes(dumps: Sequence[dict[str, Any]]) -> tuple[tuple[str, int, str], ...]:
    seen: dict[tuple[str, int, str], tuple[str, int, str]] = {}
    for data in dumps:
        for quote in data.get("evidence") or []:
            norm = (quote["doc_id"], quote["page"], " ".join(tokens(quote["quote"])))
            seen.setdefault(norm, (quote["doc_id"], quote["page"], quote["quote"]))
    return tuple(sorted(seen.values()))


def merge(samples: Sequence[Sequence[T]], key: Callable[[T], tuple[str, ...]]) -> list[Merged[T]]:
    """Merge per-sample item lists; result sorted by key. The merged item keeps the first
    sample's evidence list (to stay valid); `quotes` holds the union over samples."""
    n = len(samples)
    groups: dict[tuple[str, ...], list[T]] = {}
    for sample in samples:
        seen_here: set[tuple[str, ...]] = set()
        for item in sample:
            k = key(item)
            if k not in seen_here:  # one vote per sample
                seen_here.add(k)
                groups.setdefault(k, []).append(item)
    merged = []
    for k in sorted(groups):
        items = groups[k]
        dumps = [i.model_dump() for i in items]
        fields = {name: consensus([d[name] for d in dumps]) for name in dumps[0] if name != "evidence"}
        if "evidence" in dumps[0]:
            fields["evidence"] = dumps[0]["evidence"]
        merged.append(Merged(key=k, item=type(items[0]).model_validate(fields), agreement=len(items) / n,
                             quotes=_quotes(dumps)))
    return merged


def normalized(text: str) -> str:
    return "_".join(tokens(text)) or text


def keyed(*parts: Callable[[Any], Hashable]) -> Callable[[Any], tuple[str, ...]]:
    return lambda item: tuple(str(p(item)) for p in parts)
