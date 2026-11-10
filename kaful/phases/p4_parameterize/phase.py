"""The parameterize phase: Scope + MachineModel + SensorContract (+ DocumentIndex) -> ParamSet (M9).

Steps, in the order `build_param_set` runs them:
  a_choose_templates       one template family per full-model component (templates/families.py), from its
                           fault modes' mechanism words; the deciding mode, the others not separately modelled
  b_read_life_statements   the life statements scope found, read back from their pages (value, unit, kind,
                           consumable; verbatim, or a scanned page's transcription), plus machine.toml [[life]]
  c_life_priors            per component, a lognormal prior on nominal life T from the binding statement, and
                           k = D_limit / T; flags for no documented life and missing ratings
  d_judge_lives            parameterize.life_judgement for components with no usable documented life (the only
                           LLM step; wide, flagged); an amount-processed life converted through [[spec]] or a
                           flagged default amount per time unit
  e_value_roles            observation limits from documented channel ranges (spec, setting_range; never an
                           example), template f parameters (standards, literature, [[spec]])
  f_check                  the gates; a failure fails the phase
Everything is sorted by id, so the ParamSet does not depend on hash order (rule 14).
"""

import logging
from dataclasses import dataclass
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import PhaseRecord
from kaful.artifacts.param_set import ComponentTemplate, ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.llm.gateway import Call, CallScope, Gateway
from kaful.prompts.parameterize.life_judgement.models import LifeJudgement
from kaful.prompts.registry import Prompt, PromptNotLocked, PromptRegistry, read_lock, require_locked
from kaful.config import KafulSettings, MachineConfig
from kaful.phases import PhaseName
from kaful.phases.p4_parameterize.a_choose_templates import Choice, choose
from kaful.phases.p4_parameterize.b_read_life_statements import read
from kaful.phases.p4_parameterize.c_life_priors import FLAG_NO_LIFE, FLAG_RATINGS, LifePrior, life_prior, parameters
from kaful.phases.p4_parameterize.d_judge_lives import PROMPT_ID, Judge, Judged, inputs_for, to_prior
from kaful.phases.p4_parameterize.d_judge_lives import parameters as judged_parameters
from kaful.phases.p4_parameterize.e_value_roles import value_roles
from kaful.phases.p4_parameterize.f_check import check
from kaful.retrieval.index import load_index
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.parameterize")

RATINGS_MARK = "load ratings needed"  # scope's flag for a life model without ratings


class MissingService(RuntimeError):
    """A service the phase needs (the LLM gateway) is not configured."""


def choose_families(model: MachineModel, scope: Scope, settings: KafulSettings) -> list[Choice]:
    return choose(model, scope, settings.parameterize)


def _template(c: Choice, flags: set[str], scope: Scope) -> ComponentTemplate:
    decision = next(d for d in scope.decisions if d.component == c.component)
    if any(RATINGS_MARK in f for f in decision.flags):
        flags = flags | {FLAG_RATINGS}
    return ComponentTemplate(component=c.component, family=c.family, decided_by=c.decided_by,
                             other_modes=c.other_modes, flags=tuple(sorted(flags | set(c.flags))))


@dataclass
class Built:
    param_set: ParamSet
    priors: list[LifePrior]
    judged: list[Judged]
    failures: tuple[str, ...]


def build_param_set(base: dict[str, Any], index: DocumentIndex, model: MachineModel, scope: Scope,
                    contract: SensorContract, machine: MachineConfig, settings: KafulSettings,
                    judge: Judge | None, prompt_ref: str = PROMPT_ID) -> Built:
    """The ParamSet with its life priors, judgements (for reports) and gate failures. `judge` runs
    life_judgement on a list of prompt inputs (None: no judgement; those components stay without a life)."""
    p, time_unit = settings.parameterize, machine.confirmed_time_unit()
    choices = choose(model, scope, p)
    found = read(index, model, scope, machine, settings)
    priors = [life_prior(c.component, [s for s in found.statements if s.component == c.component], p, time_unit)
              for c in choices]
    params = [x for lp in priors for x in parameters(lp, time_unit)]
    unjudged = [c for c, lp in zip(choices, priors, strict=True) if lp.prior is None]
    judged: list[Judged] = []
    if unjudged and judge is not None:
        outputs = judge(inputs_for(unjudged, model, machine))
        judged = [to_prior(c.component, out, machine, p) for c, out in zip(unjudged, outputs, strict=True)]
        params += [x for j in judged for x in judged_parameters(j, time_unit, p, prompt_ref)]
    names = {c.id: c.name for c in model.components}
    roles, role_flags = value_roles(choices, names, scope, contract, machine, p)
    params += roles
    by_judged = {j.component: j for j in judged}
    templates = []
    for c, lp in zip(choices, priors, strict=True):
        j = by_judged.get(c.component)
        flags = (set(lp.flags) - ({FLAG_NO_LIFE} if j else set())) | set(j.flags if j else ()) \
            | set(role_flags.get(c.component, ()))
        templates.append(_template(c, flags, scope))
    param_set = ParamSet(**base, time_unit=time_unit, parameters=tuple(sorted(params, key=lambda x: x.param_id)),
                         life_statements=tuple(sorted(found.statements, key=lambda s: s.id)),
                         templates=tuple(templates))
    return Built(param_set, priors, judged, check(found.statements, priors, params, p, time_unit))


class ParameterizePhase:
    name: PhaseName = "parameterize"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            understand = {r.kind: r for r in ctx.manifest.phase("understand").artifacts}
            scoped = {r.kind: r for r in ctx.manifest.phase("scope").artifacts}
            model = self._load(ctx, understand, "machine_model", MachineModel)
            contract = self._load(ctx, understand, "sensor_contract", SensorContract)
            scope = self._load(ctx, scoped, "scope", Scope)
            index = load_index(ctx)
            prompt = PromptRegistry().load(PROMPT_ID)
            require_locked(prompt, read_lock())
            if ctx.gateway is None:
                raise MissingService("parameterize needs the LLM gateway for life_judgement")
        except (FileNotFoundError, PromptNotLocked, MissingService) as exc:
            log.error("parameterize cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": index.index_hash,
                "inputs": tuple(sorted((understand["machine_model"], scoped["scope"]), key=lambda r: (r.kind, r.path)))}
        built = build_param_set(base, index, model, scope, contract, ctx.machine, ctx.settings,
                                self._judge(ctx.gateway, prompt, ctx.settings.parameterize.judgement_samples),
                                f"{PROMPT_ID} v{prompt.version}")
        ref = ctx.save_artifact(self.name, built.param_set)
        log.info("parameterized", extra={"fields": {"components": len(built.param_set.templates),
                                                    "parameters": len(built.param_set.parameters),
                                                    "by_source": built.param_set.counts_by_source()}})
        return PhaseRecord(phase=self.name, status="failed" if built.failures else "passed", started_at=started,
                           finished_at=now_utc(), artifacts=(ref,), failures=built.failures)

    @staticmethod
    def _judge(gateway: Gateway, prompt: Prompt, n: int) -> Judge:
        def run(inputs: list[dict[str, str]]) -> list[list[LifeJudgement]]:
            scope = CallScope(phase="parameterize", step="judge_lives", artifact="param_set")
            calls = [Call(prompt=prompt, inputs=given, scope=scope, sample=s) for given in inputs for s in range(n)]
            results = gateway.run_many(calls, mode="batch") if calls else []
            outs = [r.output for r in results]
            return [[o for o in outs[i * n:(i + 1) * n] if isinstance(o, LifeJudgement)] for i in range(len(inputs))]
        return run

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
