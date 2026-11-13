"""The verify phase: TwinDefinition + ParamSet + SensorContract -> VerificationReport (M11). No LLM call.

Steps, in the order `verify_twin` runs them:
  a_recheck_static   the build's static gates (kaful.twin_checks) and provenance, re-run on the loaded artifacts
  b_simulate         build the twin through its backend; nominal and documented-maintenance runs at prior medians,
                     and documented-maintenance runs at each corner quantile
  c_check_dynamics   per component and check over every run: starts healthy, finite and bounded, monotone between
                     resets, event_state agrees with threshold_met, resets behave, life within the band
                     (helpers: traces, trace_checks)
Everything is sorted by target and check, so the report does not depend on hash order (rule 14).
"""

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult, SimulationRun
from kaful.config import KafulSettings
from kaful.phases.p6_verify.a_recheck_static import static_results
from kaful.phases.p6_verify.b_simulate import simulate
from kaful.phases.p6_verify.c_check_dynamics import dynamic_results


def verify_twin(twin: TwinDefinition, ps: ParamSet, contract: SensorContract,
                settings: KafulSettings) -> tuple[tuple[CheckResult, ...], tuple[SimulationRun, ...]]:
    static = static_results(twin, ps)
    sims, composite = simulate(twin, ps, contract, settings.build.backend, settings.verify)
    dynamic = dynamic_results(twin, ps, sims, settings.verify)
    results = tuple(sorted((*static, *composite, *dynamic), key=lambda r: (r.target, r.check)))
    return results, tuple(sorted((s.run for s in sims), key=lambda r: r.name))
