"""The parameterize phase: Scope + MachineModel (+ DocumentIndex) -> ParamSet (M9). No LLM call yet.

Steps, in the order `build_param_set` runs them:
  a_choose_templates       one template family per full-model component (templates/families.py), from its
                           fault modes' mechanism words; the deciding mode, the others not separately modelled
  b_read_life_statements   the life statements scope found, read back from their pages (value, unit, kind,
                           consumable), plus machine.toml [[life]]
  c_life_priors            per component, a lognormal prior on nominal life T from the binding statement, and
                           k = D_limit / T; flags for no documented life (life_judgement) and missing ratings
  d_check                  the gates; a failure fails the phase
Checkpoint 3 adds value roles (thresholds, capacities, nominal inputs) and life_judgement.
Everything is sorted by id, so the ParamSet does not depend on hash order (rule 14).
"""

import logging
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ComponentTemplate, ParamSet
from kaful.artifacts.scope import Scope
from kaful.config import KafulSettings, MachineConfig
from kaful.phases import PhaseName
from kaful.phases.p4_parameterize.a_choose_templates import Choice, choose
from kaful.phases.p4_parameterize.b_read_life_statements import read
from kaful.phases.p4_parameterize.c_life_priors import FLAG_RATINGS, LifePrior, life_prior, parameters
from kaful.phases.p4_parameterize.d_check import check
from kaful.retrieval.index import load_index
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.parameterize")

RATINGS_MARK = "load ratings needed"  # scope's flag for a life model without ratings


def choose_families(model: MachineModel, scope: Scope, settings: KafulSettings) -> list[Choice]:
    return choose(model, scope, settings.parameterize)


def _template(c: Choice, lp: LifePrior, scope: Scope) -> ComponentTemplate:
    flags = set(c.flags) | set(lp.flags)
    decision = next(d for d in scope.decisions if d.component == c.component)
    if any(RATINGS_MARK in f for f in decision.flags):
        flags.add(FLAG_RATINGS)
    return ComponentTemplate(component=c.component, family=c.family, decided_by=c.decided_by,
                             other_modes=c.other_modes, flags=tuple(sorted(flags)))


def build_param_set(base: dict[str, Any], index: DocumentIndex, model: MachineModel, scope: Scope,
                    machine: MachineConfig, settings: KafulSettings) -> tuple[ParamSet, list[LifePrior], tuple[str, ...]]:
    """The ParamSet, the life priors (for reports) and the gate failures."""
    time_unit = machine.confirmed_time_unit()
    choices = choose(model, scope, settings.parameterize)
    found = read(index, model, scope, machine, settings)
    priors = [life_prior(c.component, [s for s in found.statements if s.component == c.component],
                         settings.parameterize, time_unit) for c in choices]
    params = [x for lp in priors for x in parameters(lp, time_unit)]
    templates = [_template(c, lp, scope) for c, lp in zip(choices, priors, strict=True)]
    param_set = ParamSet(**base, time_unit=time_unit, parameters=tuple(sorted(params, key=lambda x: x.param_id)),
                         life_statements=tuple(sorted(found.statements, key=lambda s: s.id)),
                         templates=tuple(templates))
    return param_set, priors, check(found.statements, priors, params, settings.parameterize, time_unit)


class ParameterizePhase:
    name: PhaseName = "parameterize"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            understand = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
            scoped = {r.kind: r for r in ctx.manifest.phase("scope").artifacts}
            model = self._load(ctx, understand, "machine_model", MachineModel)
            scope = self._load(ctx, scoped, "scope", Scope)
            index = load_index(ctx)
        except FileNotFoundError as exc:
            log.error("parameterize cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": index.index_hash,
                "inputs": tuple(sorted((understand["machine_model"], scoped["scope"]), key=lambda r: (r.kind, r.path)))}
        param_set, _, failures = build_param_set(base, index, model, scope, ctx.machine, ctx.settings)
        ref = ctx.save_artifact(self.name, param_set)
        log.info("parameterized", extra={"fields": {"components": len(param_set.templates),
                                                    "parameters": len(param_set.parameters),
                                                    "by_source": param_set.counts_by_source()}})
        return PhaseRecord(phase=self.name, status="failed" if failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=failures)

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
