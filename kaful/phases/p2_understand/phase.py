"""The understand phase: DocumentIndex -> MachineModel + SensorContract (+ UnderstandReport).

Steps, in the order `_build` runs them:
  a_select_context   each prompt's contexts (prompt_contexts): the machine documents read in full,
                     whole if they fit whole_max_tokens, else in parts of at most part_max_tokens
                     (whole TOC sections; M7c) for every prompt; for fault modes also the reference
                     documents, retrieved with its queries (kaful.retrieval for search); always in
                     document order
  b_merge_samples    sample `understand.components` and `understand.sensor_channels`
                     N times on each part (several parts through the Batch API) and merge them
                     (agreement = the best part's share of its samples; parts_found kept);
                     helpers/canonical_names makes near-duplicate component ids, channel names and
                     fault-mode mechanisms one
  c_verify_quotes    the checker: every evidence quote against the raw page text,
                     provenance and confidence set in code
  d_build_artifacts  components (with the ones machine.toml declares), boundaries and flow
                     paths; channels (with the declared ones; declared items have source
                     `deployment`; a channel for an instance beyond machine.toml [instances] is
                     dropped; a bool or category observation becomes an event); items quoted only from
                     reference documents are dropped
then b and d again for `understand.fault_modes`, which is given the built component ids
and channel names and may propose carriers (v4, M7b: a part that wears that the components
step missed; merged across samples, joined to a built component of another name, kept only
with a quote from a machine document, else listed for the onboarding review), and the
UnderstandReport. Units pint cannot parse fail the phase.

Every prompt input is built from sorted data, so prompts (and their cache keys) do not
depend on hash order.
"""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.phases import PhaseName
from kaful.artifacts.document_index import DocumentIndex
from kaful.config import IngestSettings, UnderstandSettings
from kaful.llm.embeddings import Embedder, Vectors
from kaful.llm.gateway import Gateway
from kaful.llm.transport import TransportError
from kaful.phases.p2_understand.a_select_context import Context, estimated_tokens, select_context, select_parts
from kaful.phases.p2_understand.b_merge_samples import (
    Reused,
    merge_channels,
    merge_components,
    merge_carriers,
    merge_fault_modes,
    reuse_notes,
    sample_parts,
)
from kaful.phases.p2_understand.c_verify_quotes import checker
from kaful.phases.p2_understand.d_build_artifacts import Log, build_components, carriers, channels, fault_modes
from kaful.prompts.registry import Prompt, PromptNotLocked, PromptRegistry, read_lock, require_locked
from kaful.retrieval.index import INDEX_KIND, INGEST_PHASE, load_index, load_vectors
from kaful.runs import RunContext, now_utc

log = logging.getLogger("kaful.understand")

PROMPT_IDS = ("understand.components", "understand.fault_modes", "understand.sensor_channels")


def channel_list(channels: Sequence[SensorChannel]) -> str:
    return "\n".join(f"- {c.name}: {c.description}" for c in sorted(channels, key=lambda c: c.name)) or "(none)"



def prompt_contexts(index: DocumentIndex, vectors: Vectors | None, embedder: Embedder | None,
                    settings: UnderstandSettings, ingest: IngestSettings,
                    count: Callable[[str], int] | None) -> dict[str, list[Context]]:
    """Each prompt's contexts: components and channels see the machine documents in full (whole or parts;
    decided 2026-11-08, parts M7c 2026-11-09); fault modes see them too, then the reference documents,
    retrieved for mechanisms only."""
    machine = select_parts(index, settings, count, settings.machine_kinds)
    reference = sorted({d.kind for d in index.documents} - set(settings.machine_kinds))
    extra = [select_context(index, vectors, embedder, settings, ingest, count=count, kinds=reference,
                            topics=("fault_modes",))] if reference else []
    return {"understand.components": machine, "understand.fault_modes": [*machine, *extra],
            "understand.sensor_channels": machine}

class MissingService(RuntimeError):
    """The run has no LLM gateway, and understand needs it."""


class UnderstandPhase:
    name: PhaseName = "understand"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            return self._build(ctx, started)
        except (PromptNotLocked, FileNotFoundError, MissingService) as exc:
            log.error("understand cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))

    @staticmethod
    def _prompts() -> dict[str, Prompt]:
        registry, lock = PromptRegistry(), read_lock()
        prompts = {pid: registry.load(pid) for pid in PROMPT_IDS}
        for prompt in prompts.values():
            require_locked(prompt, lock)
        return prompts

    def _build(self, ctx: RunContext, started: datetime) -> PhaseRecord:
        prompts, settings, n = self._prompts(), ctx.settings.understand, ctx.settings.understand.n_samples
        if ctx.gateway is None:
            raise MissingService("understand needs the LLM gateway")
        gateway, index = ctx.gateway, load_index(ctx)
        vectors = load_vectors(ctx.machine_dir, index.vectors) if index.vectors and ctx.embedder else None
        counted: list[str] = []
        count = self._counter(gateway, ctx.settings.llm.choice("understand.components").model, settings, counted)
        contexts = prompt_contexts(index, vectors, ctx.embedder, settings, ctx.settings.ingest, count)
        reused, check = Reused(), checker(index, settings)

        def sampled(prompt: str, inputs: dict[str, str], step: str, artifact: str) -> tuple[list[Any], list[int]]:
            given = [{**inputs, "context": c.text} for c in contexts[prompt]]
            return sample_parts(gateway, prompts[prompt], given, n, step, artifact, reused, settings.batch_parts)
        component_samples, component_parts = sampled("understand.components", {}, "extract_components", "machine_model")
        merged = merge_components(component_samples, settings.merge_generic_heads, component_parts,
                                  settings.merge_variant_words)
        channel_samples, channel_parts = sampled("understand.sensor_channels", {}, "extract_sensor_channels",
                                                 "sensor_contract")
        merged_channels, channel_merges = merge_channels(channel_samples, check.pages, channel_parts)
        record = Log(notes=counted)
        components, boundaries, flows = build_components(merged, ctx.machine.components, check, record)
        built_channels = channels(merged_channels, ctx.machine.instrumentation, check, record, ctx.machine.instances)
        inputs = {"channels": channel_list(built_channels),
                  "components": "\n".join(f"- {c.id}: {c.name}" for c in components)}
        fault_samples, fault_parts = sampled("understand.fault_modes", inputs, "extract_fault_modes", "machine_model")
        merged_carriers = merge_carriers(fault_samples, components, settings.merge_generic_heads, fault_parts,
                                         settings.merge_variant_words)
        components = sorted([*components, *carriers(merged_carriers, components, check, record)], key=lambda c: c.id)
        merged_faults, fault_merges = merge_fault_modes(fault_samples, merged_carriers.on, fault_parts)
        fault_merges = merged_carriers.merges + fault_merges
        faults = fault_modes(merged_faults,
                             {c.id for c in components}, {c.name for c in built_channels}, check, record)
        base: dict[str, Any] = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id,
                                "index_hash": index.index_hash, "inputs": (self._index_ref(ctx),)}
        machine = MachineModel(**base, components=tuple(components), boundaries=tuple(boundaries),
                               flow_paths=tuple(sorted(flows, key=lambda f: f.id)), fault_modes=tuple(faults),
                               n_extractions=n)
        clock = next((c.name for c in built_channels if c.role == "usage_clock"), None)
        contract = SensorContract(**base, usage_clock=clock, channels=tuple(built_channels))
        report = self._report(ctx, base, prompts, contexts, record, merged.merges + channel_merges + fault_merges)
        refs = tuple(ctx.save_artifact(self.name, a) for a in (machine, contract, report))
        failures = tuple(f"unit: {p}" for p in report.unit_problems)
        log.info("machine understood", extra={"fields": {
            "components": len(components), "channels": len(built_channels), "fault_modes": len(faults),
            "context": contexts["understand.components"][0].selection.mode,
            "parts": len(contexts["understand.components"]), "downgraded": len(report.downgraded()), "dropped": len(report.dropped)}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=refs, failures=failures, notes=reuse_notes(reused))

    @staticmethod
    def _counter(gateway: Gateway, model: str, settings: UnderstandSettings,
                 counted: list[str]) -> Callable[[str], int] | None:
        """The API token count of a text (free), noted for the report; None when the transport cannot count."""
        transport = gateway.transport
        if transport is None or not transport.can_count_tokens:
            return None

        seen: dict[str, int] = {}

        def count(text: str) -> int:  # once per distinct text: the prompts often share one context
            if text not in seen:
                seen[text] = measure(text)
            return seen[text]

        def measure(text: str) -> int:
            try:
                n = transport.count_tokens({"model": model, "messages": [{"role": "user", "content": text}]})
            except TransportError as exc:
                log.warning("count_tokens failed; using the estimate", extra={"fields": {"error": str(exc)}})
                n = estimated_tokens(text, settings)
                counted.append(f"context: count_tokens failed; {n} tokens by the estimate")
                return n
            counted.append(f"context: {n} tokens by count_tokens")
            return n
        return count

    @staticmethod
    def _report(ctx: RunContext, base: dict[str, Any], prompts: dict[str, Prompt], contexts: dict[str, list[Context]],
                record: Log, merges: tuple[str, ...]) -> UnderstandReport:
        llm = ctx.settings.llm
        return UnderstandReport(
            **base, context=contexts["understand.components"][0].selection,
            contexts={k: tuple(c.selection for c in v) for k, v in sorted(contexts.items())}, n_samples=ctx.settings.understand.n_samples,
            prompts=tuple(sorted(f"{p.id} v{p.version} {llm.choice(p.id).model}" for p in prompts.values())),
            items=tuple(sorted(record.items, key=lambda i: (i.kind, i.id))), merges=tuple(sorted(merges)),
            dropped=tuple(sorted(record.dropped)),
            unit_problems=tuple(sorted(record.unit_problems)), notes=tuple(sorted(record.notes)))

    @staticmethod
    def _index_ref(ctx: RunContext) -> ArtifactRef:
        refs = [r for r in ctx.manifest.phase(INGEST_PHASE).artifacts if r.kind == INDEX_KIND]
        if len(refs) != 1:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {INDEX_KIND} from {INGEST_PHASE}")
        return refs[0]
