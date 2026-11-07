"""Step b: sample each understand prompt N times and merge the samples (self-consistency).

`sample` makes the N calls (sample 0 writes the prompt cache, the rest read it); the
merge_* functions merge each prompt's outputs into one list per item type.

Items are matched across samples by a key (a component id, a channel name, a flow
path's ends and medium, a fault mode's component and mechanism). For each key:
- agreement = share of the N samples that produced it (kept, never used to discard);
- each field takes its most common value among those samples, ties going to the
  earliest sample (deterministic, independent of hash order);
- evidence quotes are the union over samples, deduplicated and sorted.
"""

import json
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kaful.evidence import tokens
from kaful.llm.gateway import Call, CallScope, Gateway
from kaful.phases.p2_understand.helpers.canonical_names import canonical_ids
from kaful.prompts.registry import Prompt
from kaful.prompts.understand.components.models import (
    ComponentExtraction,
    ExtractedComponent,
    ExtractedFlowPath,
)
from kaful.prompts.understand.fault_modes.models import ExtractedFaultMode, FaultModeExtraction
from kaful.prompts.understand.sensor_channels.models import ChannelExtraction, ExtractedChannel

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


Reused = list[tuple[str, int, int]]  # (prompt id, older version, version) per sample served by an older response


def sample(gateway: Gateway, prompt: Prompt, inputs: dict[str, str], n: int, step: str,
           artifact: str, reused: Reused) -> list[BaseModel]:
    """N sampled outputs, in sample order (sample 0 writes the prompt cache, the rest read it).
    Samples served by an older version's response (a ReuseRule) are added to `reused`."""
    scope = CallScope(phase="understand", step=step, artifact=artifact)
    outputs = []
    for s in range(n):
        result = gateway.call(Call(prompt=prompt, inputs=inputs, scope=scope, sample=s))
        if result.reused_from is not None:
            reused.append((prompt.id, result.reused_from[0], prompt.version))
        outputs.append(result.output)
    return outputs


def reuse_notes(reused: Reused) -> tuple[str, ...]:
    """One note per prompt and older version whose responses were reused, with the sample count."""
    return tuple(f"{reused.count(key)} sample(s): {key[0]} v{key[1]} responses reused under v{key[2]} "
                 f"(the request differed only in a lower max_tokens; the response ended normally)"
                 for key in sorted(set(reused)))


@dataclass(frozen=True)
class MergedComponents:
    components: list[Merged[ExtractedComponent]]
    boundaries: list[str]  # every boundary any sample named, sorted
    flow_paths: list[Merged[ExtractedFlowPath]]
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)  # kept id -> the ids merged into it
    merges: tuple[str, ...] = ()  # one line per merge, with its reasons (helpers/canonical_names)


def _renamed(o: ComponentExtraction, ids: dict[str, str]) -> ComponentExtraction:
    """The sample with every component id (and parent and flow-path end) replaced by its kept id."""
    def node(n: str) -> str:
        return ids.get(n, n)
    components = [c.model_copy(update={"id": node(c.id), "parent": None if c.parent is None else node(c.parent)})
                  for c in o.components]
    flows = [f.model_copy(update={"from_node": node(f.from_node), "to_node": node(f.to_node)}) for f in o.flow_paths]
    return o.model_copy(update={"components": components, "flow_paths": flows})


def merge_components(parts: Sequence[BaseModel]) -> MergedComponents:
    """Near-duplicate ids across samples are first made one (canonical_ids), then samples merge by id."""
    raw = [p for p in parts if isinstance(p, ComponentExtraction)]
    ids, merges = canonical_ids([o.components for o in raw])
    outs = [_renamed(o, ids) for o in raw]
    aliases: dict[str, list[str]] = {}
    for old, kept in sorted(ids.items()):
        if old != kept:
            aliases.setdefault(kept, []).append(old)
    return MergedComponents(
        components=merge([o.components for o in outs], keyed(lambda c: c.id)),
        boundaries=sorted({b for o in outs for b in o.boundaries}),
        flow_paths=merge([o.flow_paths for o in outs], keyed(lambda f: f.from_node, lambda f: f.to_node,
                                                             lambda f: f.medium)),
        aliases={k: tuple(v) for k, v in sorted(aliases.items())}, merges=merges)


def merge_channels(parts: Sequence[BaseModel]) -> list[Merged[ExtractedChannel]]:
    return merge([p.channels for p in parts if isinstance(p, ChannelExtraction)], keyed(lambda c: c.name))


def merge_fault_modes(parts: Sequence[BaseModel]) -> list[Merged[ExtractedFaultMode]]:
    return merge([p.fault_modes for p in parts if isinstance(p, FaultModeExtraction)],
                 keyed(lambda f: f.component, lambda f: normalized(f.mechanism)))
