"""The build phase: ParamSet + Scope + SensorContract -> TwinDefinition (M10). No LLM call.

Steps, in the order `build_twin` runs them:
  a_bind_templates   one binding per full-model component: its family (M9), slot -> parameter, D in D_limit's
                     unit, health derived, one event per binding
  b_wire             what feeds each template condition: a driving channel, a nominal parameter, or nothing
                     (undocumented: f = 1, flagged); exact matching, sorted; no component-to-component quantity
                     yet (M13)
  c_maintenance      maintenance rules from documented and declared actions (kaful.lifetime) and the stored
                     profiles: nominal and documented_maintenance
  d_check            the gates; a failure fails the phase
Everything is sorted by id, so the TwinDefinition does not depend on hash order (rule 14).
"""

import logging
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import EventDef, ExternalInput, TwinDefinition
from kaful.config import KafulSettings, MachineConfig
from kaful.lifetime import documented_actions
from kaful.phases import PhaseName
from kaful.phases.p5_build.a_bind_templates import bind, missing
from kaful.phases.p5_build.b_wire import wire
from kaful.phases.p5_build.c_maintenance import horizon, profiles, rules
from kaful.phases.p5_build.d_check import check
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.build")


def build_twin(base: dict[str, Any], ps: ParamSet, scope: Scope, contract: SensorContract, machine: MachineConfig,
               settings: KafulSettings) -> tuple[TwinDefinition, tuple[str, ...]]:
    templates = [t for t in ps.templates if not missing(t, ps)]
    lacking = [f"component {t.component}: no {', '.join(missing(t, ps))} in the ParamSet"
               for t in ps.templates if missing(t, ps)]
    bindings, inputs, flags = [], [], {}
    for t in templates:
        binding = bind(t, ps)
        wired, conditions, wire_flags = wire(binding, ps, scope, contract)
        bindings.append(binding.model_copy(update={"conditions": conditions}))
        inputs += wired
        flags[t.component] = wire_flags
    bound = {b.component for b in bindings}
    actions = documented_actions(ps.life_statements, machine, settings.parameterize, settings.build)
    maintenance, maintenance_flags = rules(actions, bound, settings.build)
    for component, more in maintenance_flags.items():
        flags.setdefault(component, []).extend(more)
    twin = TwinDefinition(
        **base, time_unit=ps.time_unit, components=tuple(sorted(bound)), bindings=tuple(bindings),
        external_inputs=tuple(sorted(inputs, key=lambda i: i.name)),
        events=tuple(EventDef(name=f"{b.component}_failure", binding=b.id) for b in bindings),
        maintenance=tuple(maintenance), profiles=profiles(maintenance, horizon(ps, bound, settings.build)),
        flags={c: tuple(sorted(f)) for c, f in sorted(flags.items()) if f})
    return twin, check(twin, ps, lacking)


class BuildPhase:
    name: PhaseName = "build"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            understand = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
            scoped = {r.kind: r for r in ctx.manifest.phase("scope").artifacts}
            params = {r.kind: r for r in ctx.manifest.phase("parameterize").artifacts}
            contract = self._load(ctx, understand, "sensor_contract", SensorContract)
            scope = self._load(ctx, scoped, "scope", Scope)
            ps = self._load(ctx, params, "param_set", ParamSet)
        except FileNotFoundError as exc:
            log.error("build cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": ps.index_hash,
                "inputs": tuple(sorted((params["param_set"], scoped["scope"], understand["sensor_contract"]),
                                       key=lambda r: (r.kind, r.path)))}
        twin, failures = build_twin(base, ps, scope, contract, ctx.machine, ctx.settings)
        ref = ctx.save_artifact(self.name, twin)
        log.info("built", extra={"fields": {"bindings": len(twin.bindings), "inputs": len(twin.external_inputs),
                                            "maintenance": len(twin.maintenance)}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=failures)

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
