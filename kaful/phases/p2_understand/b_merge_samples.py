"""Step b: sample each understand prompt N times and merge the samples (self-consistency).

`sample` makes the N calls (sample 0 writes the prompt cache, the rest read it); `sample_parts` does it
for every context part (M7c); the merge_* functions merge each prompt's outputs into one list per item type.

Items are matched across samples by a key (a component id, a channel name, a flow
path's ends and medium, a fault mode's component and mechanism). For each key:
- agreement = share of the N samples that produced it (kept, never used to discard); when the context
  came in parts (M7c), the samples are grouped by part and agreement is the best part's share, with the
  number of parts that produced it kept as `parts_found` (approved 2026-11-09);
- each field takes its most common value among those samples, ties going to the
  earliest sample (deterministic, independent of hash order);
- evidence quotes are the union over samples, deduplicated and sorted.
"""

import json
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from kaful.artifacts.machine_model import Component
from kaful.evidence import tokens
from kaful.llm.gateway import Call, CallScope, Gateway
from kaful.phases.p2_understand.helpers.canonical_names import (
    DEFAULT_HEADS,
    DEFAULT_VARIANT_WORDS,
    Pages,
    canonical_ids,
    channel_kind,
    component_kind,
    fault_mode_id,
    fault_mode_kind,
    generic_match,
    page_occurrences,
    variant_base,
    variant_bases,
)
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
    parts_found: int = 1


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


def merge(samples: Sequence[Sequence[T]], key: Callable[[T], tuple[str, ...]],
          part_of: Sequence[int] | None = None) -> list[Merged[T]]:
    """Merge per-sample item lists; result sorted by key. The merged item keeps the first
    sample's evidence list (to stay valid); `quotes` holds the union over samples. `part_of` gives each
    sample's context part (default: one part)."""
    part = list(part_of) if part_of is not None else [0] * len(samples)
    size = Counter(part)
    groups: dict[tuple[str, ...], list[tuple[T, int]]] = {}
    for sample, where in zip(samples, part, strict=True):
        seen_here: set[tuple[str, ...]] = set()
        for item in sample:
            k = key(item)
            if k not in seen_here:  # one vote per sample
                seen_here.add(k)
                groups.setdefault(k, []).append((item, where))
    merged = []
    for k in sorted(groups):
        items = [item for item, _ in groups[k]]
        found = Counter(where for _, where in groups[k])
        dumps = [i.model_dump() for i in items]
        fields = {name: consensus([d[name] for d in dumps]) for name in dumps[0] if name != "evidence"}
        if "evidence" in dumps[0]:
            fields["evidence"] = dumps[0]["evidence"]
        merged.append(Merged(key=k, item=type(items[0]).model_validate(fields),
                             agreement=max(found[w] / size[w] for w in found), quotes=_quotes(dumps),
                             parts_found=len(found)))
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


def sample_parts(gateway: Gateway, prompt: Prompt, inputs: Sequence[dict[str, str]], n: int, step: str,
                 artifact: str, reused: Reused, batch: bool) -> tuple[list[BaseModel], list[int]]:
    """N sampled outputs per context part (`inputs`, one per part), with each output's part. One part runs
    as `sample` does; several go through the Batch API when `batch` (M7c), in part then sample order."""
    if len(inputs) == 1 or not batch:
        outs = [(out, part) for part, given in enumerate(inputs)
                for out in sample(gateway, prompt, given, n, step, artifact, reused)]
        return [o for o, _ in outs], [w for _, w in outs]
    scope = CallScope(phase="understand", step=step, artifact=artifact)
    calls = [Call(prompt=prompt, inputs=given, scope=scope, sample=s) for given in inputs for s in range(n)]
    results = gateway.run_many(calls, mode="batch")
    for result in results:
        if result.reused_from is not None:
            reused.append((prompt.id, result.reused_from[0], prompt.version))
    return [r.output for r in results], [i // n for i in range(len(calls))]


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
    components = [c.model_copy(update={"id": node(c.id), "parent": None if c.parent is None
                                       or node(c.parent) == node(c.id) else node(c.parent)})
                  for c in o.components]
    flows = [f.model_copy(update={"from_node": node(f.from_node), "to_node": node(f.to_node)}) for f in o.flow_paths]
    return o.model_copy(update={"components": components, "flow_paths": flows})


def _typed(parts: Sequence[BaseModel], kind: type[T], part_of: Sequence[int] | None) -> tuple[list[T], list[int]]:
    """The samples of one output type, with each one's context part (default: one part)."""
    where = list(part_of) if part_of is not None else [0] * len(parts)
    kept = [(p, w) for p, w in zip(parts, where, strict=True) if isinstance(p, kind)]
    return [p for p, _ in kept], [w for _, w in kept]


def _base_first(o: ComponentExtraction, folds: dict[str, str]) -> ComponentExtraction:
    """The sample with its variants last, so a sample that lists a base and its variant votes with the base."""
    return o.model_copy(update={"components": sorted(o.components, key=lambda c: c.id in folds)})


def merge_components(parts: Sequence[BaseModel], heads: Sequence[str] = DEFAULT_HEADS,
                     part_of: Sequence[int] | None = None,
                     variants: Sequence[str] = DEFAULT_VARIANT_WORDS) -> MergedComponents:
    """Near-duplicate ids across samples are first made one (canonical_ids), a series or type of another
    component folds into it (variant_bases, M7c), then samples merge by id."""
    raw, where = _typed(parts, ComponentExtraction, part_of)
    ids, merges = canonical_ids([o.components for o in raw], component_kind(heads))
    outs = [_renamed(o, ids) for o in raw]
    folds, folded = variant_bases([o.components for o in outs], heads, variants)
    outs = [_renamed(_base_first(o, folds), folds) for o in outs]
    ids = {old: folds.get(kept, kept) for old, kept in ids.items()}
    merges = (*merges, *folded)
    aliases: dict[str, list[str]] = {}
    for old, kept in sorted(ids.items()):
        if old != kept:
            aliases.setdefault(kept, []).append(old)
    return MergedComponents(
        components=merge([o.components for o in outs], keyed(lambda c: c.id), where),
        boundaries=sorted({b for o in outs for b in o.boundaries}),
        flow_paths=merge([o.flow_paths for o in outs], keyed(lambda f: f.from_node, lambda f: f.to_node,
                                                             lambda f: f.medium), where),
        aliases={k: tuple(v) for k, v in sorted(aliases.items())}, merges=merges)


def merge_channels(parts: Sequence[BaseModel], pages: Pages, part_of: Sequence[int] | None = None,
                   ) -> tuple[list[Merged[ExtractedChannel]], tuple[str, ...]]:
    """Channels named differently for one quote or one repeated reading are first made one
    (canonical_ids), then samples merge by name; returns the merged channels and the merges."""
    typed, where = _typed(parts, ChannelExtraction, part_of)
    raw = [list(p.channels) for p in typed]
    names, merges = canonical_ids(raw, channel_kind(page_occurrences(pages)))
    renamed = [[c.model_copy(update={"name": names[c.name]}) for c in sample] for sample in raw]
    return merge(renamed, keyed(lambda c: c.name), where), merges


@dataclass(frozen=True)
class MergedCarriers:
    carriers: list[Merged[ExtractedComponent]]  # new parts only, merged across samples
    on: dict[str, str]  # each proposed carrier id -> the id its fault modes sit on (a built component or a carrier)
    merges: tuple[str, ...]


def merge_carriers(parts: Sequence[BaseModel], built: Sequence[Component],
                   heads: Sequence[str] = DEFAULT_HEADS, part_of: Sequence[int] | None = None,
                   variants: Sequence[str] = DEFAULT_VARIANT_WORDS) -> MergedCarriers:
    """Carriers the fault-mode samples propose (v4, M7b): near-duplicates are made one (canonical_ids, with the
    built components as one more sample), and a carrier that is a built component under another name becomes
    that component, whatever the sample counts. A carrier named by one generic word joins the only built
    component whose name ends in it (M7c: CDV filters -> inline_gas_filter), and one that is a series or type
    of a built component folds into it (variant_base, M7c); the rest merge by id."""
    typed, where = _typed(parts, FaultModeExtraction, part_of)
    raw = [list(p.carriers) for p in typed]
    ids, merges = canonical_ids([list(built), *raw], component_kind(heads))
    known = {c.id for c in built}
    groups: dict[str, set[str]] = {}
    for i, kept in sorted(ids.items()):
        groups.setdefault(kept, set()).add(i)
    on = {i: min(members & known, default=kept) for kept, members in groups.items() for i in members}
    texts: dict[str, set[str]] = {}
    for c in (c for sample in raw for c in sample):
        texts.setdefault(ids[c.id], set()).update({c.name, *(q.quote for q in c.evidence)})
    generic = []
    for kept, members in sorted(groups.items()):
        if members & known:
            continue
        target = generic_match(kept, sorted(known), heads)
        why = "one generic word; the only built component named so"
        if target is None:
            target = variant_base(kept, sorted(texts.get(kept, ())), sorted(known), heads, variants)
            why = "a series or type of it"
        if target is not None:
            on.update({i: target for i in members})
            generic.append(f"{target} <- {kept} ({why})")
    merges = tuple(sorted((*merges, *generic)))
    renamed = [[c.model_copy(update={"id": on[c.id]}) for c in sample if on[c.id] not in known] for sample in raw]
    return MergedCarriers(merge(renamed, keyed(lambda c: c.id), where),
                          {i: t for i, t in sorted(on.items()) if i not in known}, merges)


def merge_fault_modes(parts: Sequence[BaseModel], on: dict[str, str] | None = None,
                      part_of: Sequence[int] | None = None) -> tuple[list[Merged[ExtractedFaultMode]], tuple[str, ...]]:
    """Modes of one component named differently for one quote are first made one (canonical_ids), then
    samples merge by component and mechanism; returns the merged modes and the merges. `on` renames the
    carriers modes sit on (merge_carriers)."""
    typed, where = _typed(parts, FaultModeExtraction, part_of)
    raw = [[f.model_copy(update={"component": (on or {}).get(f.component, f.component)}) for f in p.fault_modes]
           for p in typed]
    ids, merges = canonical_ids(raw, fault_mode_kind())
    first: dict[str, ExtractedFaultMode] = {}
    for f in (f for sample in raw for f in sample):
        first.setdefault(fault_mode_id(f), f)
    renamed = [[f if ids[fault_mode_id(f)] == fault_mode_id(f) else
                f.model_copy(update={k: getattr(first[ids[fault_mode_id(f)]], k) for k in ("id", "mechanism")})
                for f in sample] for sample in raw]
    return merge(renamed, keyed(lambda f: f.component, lambda f: normalized(f.mechanism)), where), merges
