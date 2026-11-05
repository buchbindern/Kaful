"""The understand phase: DocumentIndex -> MachineModel + SensorContract (+ UnderstandReport).

1. select the context (whole index if it fits, else retrieval), in document order;
2. sample `understand.components` and `understand.sensor_channels` N times each and
   merge them (agreement = share of samples);
3. sample `understand.fault_modes` N times, given the merged component ids and channel
   names, and merge;
4. verify every evidence quote against the raw page text, set provenance and
   confidence in code, add the channels machine.toml declares (source `deployment`);
5. save the three artifacts. Units pint cannot parse fail the phase.

Every prompt input is built from sorted data, so prompts (and their cache keys) do not
depend on hash order.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import Component, FaultMode, FlowPath, MachineModel
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.llm.gateway import Call, CallScope, Gateway
from kaful.phases import PhaseName
from kaful.phases.ingest.index import INDEX_KIND, INGEST_PHASE, load_index, load_vectors
from kaful.phases.understand import build
from kaful.phases.understand.context import Context, select_context
from kaful.phases.understand.merge import keyed, merge, normalized
from kaful.phases.understand.verify import PageLookup
from kaful.prompts.registry import Prompt, PromptNotLocked, PromptRegistry, read_lock, require_locked
from kaful.prompts.understand.components.models import ComponentExtraction
from kaful.prompts.understand.fault_modes.models import FaultModeExtraction
from kaful.prompts.understand.sensor_channels.models import ChannelExtraction
from kaful.runs import RunContext, now_utc

log = logging.getLogger("kaful.understand")

PROMPT_IDS = ("understand.components", "understand.fault_modes", "understand.sensor_channels")


def sample(gateway: Gateway, prompt: Prompt, inputs: dict[str, str], n: int, step: str,
           artifact: str) -> list[BaseModel]:
    """N sampled outputs, in sample order (sample 0 writes the prompt cache, the rest read it)."""
    scope = CallScope(phase="understand", step=step, artifact=artifact)
    return [gateway.call(Call(prompt=prompt, inputs=inputs, scope=scope, sample=s)).output for s in range(n)]


def channel_list(channels: Sequence[SensorChannel]) -> str:
    return "\n".join(f"- {c.name}: {c.description}" for c in sorted(channels, key=lambda c: c.name)) or "(none)"


class UnderstandPhase:
    name: PhaseName = "understand"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            return self._build(ctx, started)
        except (PromptNotLocked, FileNotFoundError) as exc:
            log.error("understand cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))

    def _prompts(self) -> dict[str, Prompt]:
        registry, lock = PromptRegistry(), read_lock()
        prompts = {pid: registry.load(pid) for pid in PROMPT_IDS}
        for prompt in prompts.values():
            require_locked(prompt, lock)
        return prompts

    def _context(self, ctx: RunContext, index: DocumentIndex) -> Context:
        settings = ctx.settings.understand
        vectors = None
        if index.vectors is not None and ctx.embedder is not None:
            vectors = load_vectors(ctx.machine_dir, index.vectors)
        return select_context(index, vectors, ctx.embedder, settings, ctx.settings.ingest)

    def _build(self, ctx: RunContext, started: datetime) -> PhaseRecord:
        prompts = self._prompts()
        if ctx.gateway is None:
            raise RuntimeError("understand needs the LLM gateway")
        index = load_index(ctx)
        run = _Run(ctx, ctx.gateway, prompts, self._context(ctx, index),
                   build.Checker(PageLookup(index), ctx.settings.understand), build.Log())
        components, boundaries, flows = run.components()
        channels = run.channels()
        faults = run.fault_modes(components, channels)
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": index.index_hash,
                "inputs": (self._index_ref(ctx),)}
        machine = MachineModel(**base, components=tuple(components), boundaries=tuple(boundaries),
                               flow_paths=tuple(sorted(flows, key=lambda f: f.id)), fault_modes=tuple(faults),
                               n_extractions=run.n)
        clock = next((c.name for c in channels if c.role == "usage_clock"), None)
        contract = SensorContract(**base, usage_clock=clock, channels=tuple(channels))
        understand = run.report(base)
        refs = tuple(ctx.save_artifact(self.name, a) for a in (machine, contract, understand))
        failures = tuple(f"unit: {p}" for p in understand.unit_problems)
        log.info("machine understood", extra={"fields": {
            "components": len(components), "channels": len(channels), "fault_modes": len(faults),
            "context": run.context.selection.mode, "downgraded": len(understand.downgraded()),
            "dropped": len(understand.dropped)}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=refs, failures=failures)

    @staticmethod
    def _index_ref(ctx: RunContext) -> ArtifactRef:
        refs = [r for r in ctx.manifest.phase(INGEST_PHASE).artifacts if r.kind == INDEX_KIND]
        if len(refs) != 1:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {INDEX_KIND} from {INGEST_PHASE}")
        return refs[0]


@dataclass
class _Run:
    """One understand run's steps, sharing the context, the checker and the report log."""

    ctx: RunContext
    gateway: Gateway
    prompts: dict[str, Prompt]
    context: Context
    check: build.Checker
    log: build.Log

    @property
    def n(self) -> int:
        return self.ctx.settings.understand.n_samples

    def _sample(self, prompt_id: str, inputs: dict[str, str], step: str, artifact: str) -> list[BaseModel]:
        return sample(self.gateway, self.prompts[prompt_id], inputs, self.n, step, artifact)

    def components(self) -> tuple[list[Component], list[str], list[FlowPath]]:
        parts = self._sample("understand.components", {"context": self.context.text}, "extract_components",
                             "machine_model")
        outs = [p for p in parts if isinstance(p, ComponentExtraction)]
        components = build.components(merge([o.components for o in outs], keyed(lambda c: c.id)), self.check, self.log)
        boundaries = sorted({b for o in outs for b in o.boundaries} - {c.id for c in components})
        flows = build.flow_paths(merge([o.flow_paths for o in outs],
                                       keyed(lambda f: f.from_node, lambda f: f.to_node, lambda f: f.medium)),
                                 {c.id for c in components} | set(boundaries), self.check, self.log)
        used = {f.from_component for f in flows} | {f.to_component for f in flows}
        self.log.notes += [f"boundary {b}: no flow path uses it; not kept" for b in boundaries if b not in used]
        return components, [b for b in boundaries if b in used], flows

    def channels(self) -> list[SensorChannel]:
        parts = self._sample("understand.sensor_channels", {"context": self.context.text}, "extract_sensor_channels",
                             "sensor_contract")
        outs = [p for p in parts if isinstance(p, ChannelExtraction)]
        return build.channels(merge([o.channels for o in outs], keyed(lambda c: c.name)),
                              self.ctx.machine.instrumentation, self.check, self.log)

    def fault_modes(self, components: Sequence[Component], channels: Sequence[SensorChannel]) -> list[FaultMode]:
        inputs = {"context": self.context.text, "channels": channel_list(channels),
                  "components": "\n".join(f"- {c.id}: {c.name}" for c in components)}
        parts = self._sample("understand.fault_modes", inputs, "extract_fault_modes", "machine_model")
        outs = [p for p in parts if isinstance(p, FaultModeExtraction)]
        return build.fault_modes(merge([o.fault_modes for o in outs],
                                       keyed(lambda f: f.component, lambda f: normalized(f.mechanism))),
                                 {c.id for c in components}, {c.name for c in channels}, self.check, self.log)

    def report(self, base: dict[str, Any]) -> UnderstandReport:
        llm = self.ctx.settings.llm
        return UnderstandReport(
            **base, context=self.context.selection, n_samples=self.n,
            prompts=tuple(sorted(f"{p.id} v{p.version} {llm.choice(p.id).model}" for p in self.prompts.values())),
            items=tuple(sorted(self.log.items, key=lambda i: (i.kind, i.id))), dropped=tuple(sorted(self.log.dropped)),
            unit_problems=tuple(sorted(self.log.unit_problems)), notes=tuple(sorted(self.log.notes)))
