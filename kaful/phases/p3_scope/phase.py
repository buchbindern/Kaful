"""The scope phase: MachineModel + SensorContract (+ DocumentIndex) -> Scope (M8). No LLM call.

Steps, in the order `build_scope` runs them:
  a_collect_signals        G (counted fault modes), O (observing channels, with documented limits; and, kept
                           apart, channels whose name or description names the component), D (driving
                           channels) and declared, from what understand produced
  b_find_life_statements   L: stated lives and intervals in machine documents (sentences and schedule-table
                           rows; kaful.doctext), life ratings and life models on reference-document pages the
                           component's fault modes cite, and machine.toml [[life]]; plus machine-wide statements
  c_decide                 full_model = G and (O or L), O fault-mode linked or named; connection_only for ancestors and flow-path neighbours
                           of full models and for G alone (flagged); excluded otherwise; then [[scope]] overrides
  d_check                  the gates; a failure fails the phase
Everything is sorted by id, so the Scope does not depend on hash order (rule 14).
"""

import logging
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.config import KafulSettings, MachineConfig
from kaful.phases import PhaseName
from kaful.phases.p3_scope.a_collect_signals import collect
from kaful.phases.p3_scope.b_find_life_statements import find_life
from kaful.phases.p3_scope.c_decide import decide
from kaful.phases.p3_scope.d_check import check
from kaful.retrieval.index import load_index
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.scope")

UPSTREAM: PhaseName = "understand"


def build_scope(base: dict[str, Any], index: DocumentIndex, model: MachineModel, contract: SensorContract,
                machine: MachineConfig, settings: KafulSettings) -> tuple[Scope, tuple[str, ...]]:
    """The Scope and the gate failures; `base` holds the ArtifactBase fields (run id, machine, inputs)."""
    collected = collect(model, contract, machine, settings.scope, settings.understand.merge_generic_heads)
    life = find_life(index, model, machine, settings.scope, settings.understand)
    decisions = decide(model, collected, life.by_component, machine.scope, settings.scope)
    scope = Scope(**base, decisions=tuple(decisions), machine_life=tuple(life.machine_wide))
    return scope, check(model, decisions, machine)


class ScopePhase:
    name: PhaseName = "scope"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            refs = {r.kind: r for r in ctx.manifest.phase(UPSTREAM).artifacts}
            model = self._load(ctx, refs, "machine_model", MachineModel)
            contract = self._load(ctx, refs, "sensor_contract", SensorContract)
            index = load_index(ctx)
        except FileNotFoundError as exc:
            log.error("scope cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": index.index_hash,
                "inputs": tuple(sorted((refs["machine_model"], refs["sensor_contract"]),
                                       key=lambda r: (r.kind, r.path)))}
        scope, failures = build_scope(base, index, model, contract, ctx.machine, ctx.settings)
        ref = ctx.save_artifact(self.name, scope)
        counts = {k: sum(d.decision == k for d in scope.decisions) for k in ("full_model", "connection_only", "excluded")}
        log.info("scope decided", extra={"fields": {**counts, "overrides": sum(d.decided_by == "override"
                                                                              for d in scope.decisions),
                                                    "machine_life": len(scope.machine_life)}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=failures)

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind} from {UPSTREAM}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
