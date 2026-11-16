"""The verify phase: TwinDefinition + ParamSet + SensorContract -> VerificationReport and the verified
TwinDefinition (M11). No LLM call.

Steps, in the order `verify` runs them:
  a_recheck_static   the build's static gates (kaful.twin_checks) and provenance, re-run on the loaded artifacts
  b_simulate         build the twin through its backend; nominal and documented-maintenance runs at prior medians,
                     and documented-maintenance runs at each corner quantile
  c_check_dynamics   per component and check over every run: starts healthy, finite and bounded, monotone between
                     resets, event_state agrees with threshold_met, resets behave, life within the band
                     (kaful.dynamics: traces, trace_checks)
  d_quarantine       each component failing a dynamic check (and anything it feeds) leaves the twin; a, b and c
                     re-run on the reduced twin
Outcomes (hard rule 2): every check passes -> `passed`, and the verified twin is the built one. Only components
fail -> `quarantined`: the run goes on with the reduced twin, and the manifest lists each quarantined component and
why. A static or composite failure, a failure on the reduced twin, or nothing left -> `failed`, and the run stops.
`--allow-failures` keeps every component (nothing is quarantined) and the report records that failures were
allowed. Everything is sorted by target and check, so the report does not depend on hash order (rule 14).
"""

import logging
from dataclasses import dataclass
from typing import Any

from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.manifest import PhaseRecord, PhaseStatus
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult, Quarantine, SimulationRun, VerificationReport
from kaful.config import KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p6_verify.a_recheck_static import static_results
from kaful.phases.p6_verify.b_simulate import simulate
from kaful.phases.p6_verify.c_check_dynamics import dynamic_results
from kaful.phases.p6_verify.d_quarantine import quarantine, reduce
from kaful.runs import RunContext, load_artifact, now_utc

log = logging.getLogger("kaful.verify")


@dataclass(frozen=True)
class Verified:
    report: VerificationReport
    twin: TwinDefinition | None  # the twin later phases use; None when the run stops
    status: PhaseStatus
    failures: tuple[str, ...]
    notes: tuple[str, ...]


def check_twin(twin: TwinDefinition, ps: ParamSet, contract: SensorContract, settings: KafulSettings,
               quarantined: frozenset[str] = frozenset()) -> tuple[tuple[CheckResult, ...], tuple[SimulationRun, ...]]:
    """Steps a, b and c on one twin (`quarantined`: the components step d left out of it)."""
    static = static_results(twin, ps, contract, quarantined)
    sims, composite = simulate(twin, ps, contract, settings.build.backend, settings.verify)
    dynamic = dynamic_results(twin, ps, sims, settings.verify)
    results = tuple(sorted((*static, *composite, *dynamic), key=lambda r: (r.target, r.check)))
    return results, tuple(sorted((s.run for s in sims), key=lambda r: r.name))


def _line(r: CheckResult, prefix: str = "") -> str:
    return f"{prefix}{r.target} {r.check}: {r.message}"


def verify(report_base: dict[str, Any], twin_base: dict[str, Any], twin: TwinDefinition, ps: ParamSet,
           contract: SensorContract, settings: KafulSettings, allow_failures: bool) -> Verified:
    """`report_base` and `twin_base`: the artifact fields (run, machine, index hash, inputs) of each output."""
    results, runs = check_twin(twin, ps, contract, settings)
    quarantined = quarantine(twin, results)
    as_twin = twin.model_copy(update=twin_base)
    failing = [r for r in results if r.status == "fail"]
    if allow_failures or not failing:
        report = VerificationReport(**report_base, results=results, runs=runs, allowed_failures=bool(failing))
        return Verified(report, as_twin, "failed" if failing else "passed", tuple(_line(r) for r in failing),
                        (f"verified twin: all {len(twin.components)} components",))
    held = {q.component for q in quarantined}
    hard = [r for r in failing if r.target not in held]
    lines = tuple([_line(r) for r in hard] + [f"quarantined {q.component}: {q.reason}" for q in quarantined])
    if hard or held >= set(twin.components):
        why = () if hard else ("quarantine leaves no component",)
        report = VerificationReport(**report_base, results=results, runs=runs, quarantined=quarantined)
        return Verified(report, None, "failed", lines + why, ())
    reduced = reduce(twin, quarantined, twin_base)
    recheck, _ = check_twin(reduced, ps, contract, settings, frozenset(held))
    report = VerificationReport(**report_base, results=results, runs=runs, quarantined=quarantined, recheck=recheck)
    bad = tuple(_line(r, "recheck ") for r in recheck if r.status == "fail")
    note = f"verified twin: {len(reduced.components)} of {len(twin.components)} components ({len(held)} quarantined)"
    return Verified(report, None if bad else reduced, "failed" if bad else "quarantined", lines + bad, (note,))


class VerifyPhase:
    name: PhaseName = "verify"

    def run(self, ctx: RunContext) -> PhaseRecord:
        started = now_utc()
        try:
            refs = {r.kind: r for p in ("understand", "parameterize", "build") for r in ctx.manifest.phase(p).artifacts}
            twin = self._load(ctx, refs, "twin_definition", TwinDefinition)
            ps = self._load(ctx, refs, "param_set", ParamSet)
            contract = self._load(ctx, refs, "sensor_contract", SensorContract)
        except FileNotFoundError as exc:
            log.error("verify cannot start", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=self.name, status="failed", started_at=started, finished_at=now_utc(),
                               failures=(str(exc),))
        inputs = tuple(sorted((refs["twin_definition"], refs["param_set"], refs["sensor_contract"]),
                              key=lambda r: (r.kind, r.path)))
        base = {"run_id": ctx.manifest.run_id, "machine_id": ctx.machine.machine_id, "index_hash": twin.index_hash}
        done = verify({**base, "inputs": inputs}, {**base, "inputs": (refs["twin_definition"],)}, twin, ps, contract,
                      ctx.settings, ctx.manifest.allow_failures)
        saved = [ctx.save_artifact(self.name, done.report)]
        if done.twin is not None:
            saved.append(ctx.save_artifact(self.name, done.twin))
        log.info("verified", extra={"fields": {"status": done.status, "quarantined": list(
            done.report.quarantined_components), "checks": len(done.report.results)}})
        return PhaseRecord(phase=self.name, status=done.status, started_at=started, finished_at=now_utc(),
                           artifacts=tuple(sorted(saved, key=lambda r: (r.kind, r.path))), failures=done.failures,
                           notes=done.notes)

    @staticmethod
    def _load(ctx: RunContext, refs: dict[str, ArtifactRef], kind: str, model: type[ArtifactBase]) -> Any:
        if kind not in refs:
            raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {kind}")
        return load_artifact(ctx.machine_dir, refs[kind], model)
