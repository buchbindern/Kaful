"""Step b: build the twin through its backend and simulate it, with no noise and fixed parameter values.

Runs, in this order:
- `nominal@q0.5`: no maintenance, every parameter at its prior median (the life check reads this run);
- `documented_maintenance@q0.5`: the stored maintenance schedule, at medians (the reset checks read this run);
- `documented_maintenance@q<q>` for each of `verify.corner_quantiles`: every parameter at that prior quantile;
- `nominal_uncoupled@q0.5`, only for a twin with connections (M13b): `nominal@q0.5` with every coupling parameter at
  its form's neutral value (m = 1), so the life check reads the bindings' calibration exactly as before; the
  coupled crossing check reads `nominal@q0.5`.

The step is the shortest median life among the bindings over `verify.steps_per_shortest_life`; each run goes to
its profile's end (build: `profile_horizon_lives` times the longest median life). A twin the backend cannot
build, a run that raises, or one longer than `verify.max_steps` is a `composite` failure: it stops the run.
"""

import math
from dataclasses import dataclass

from kaful.artifacts.distributions import Delta
from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult, SimulationRun
from kaful.backends import load_backend
from kaful.backends.base import BuildError, SimulationResult
from kaful.config import VerifySettings
from kaful.phases.p6_verify.a_recheck_static import COMPOSITE
from kaful.templates.couplings import COUPLINGS

MEDIAN = 0.5
NOMINAL, MAINTAINED = "nominal", "documented_maintenance"
UNCOUPLED = "nominal_uncoupled@q0.5"


@dataclass(frozen=True)
class Simulated:
    run: SimulationRun
    profile: OperatingProfile
    values: dict[str, float]  # param_id -> the value this run used
    result: SimulationResult


def param_values(ps: ParamSet, q: float) -> dict[str, float]:
    return {x.param_id: x.prior.value if isinstance(x.prior, Delta) else x.prior.quantile(q)
            for x in sorted(ps.parameters, key=lambda x: x.param_id)}


def step(twin: TwinDefinition, ps: ParamSet, v: VerifySettings) -> float:
    lives = [ps.get(f"{b.component}.life_nominal").prior.quantile(MEDIAN) for b in twin.bindings]
    return min(lives) / v.steps_per_shortest_life


def planned(v: VerifySettings) -> tuple[tuple[str, float], ...]:
    return ((NOMINAL, MEDIAN), (MAINTAINED, MEDIAN), *((MAINTAINED, q) for q in v.corner_quantiles))


def neutral(twin: TwinDefinition, values: dict[str, float]) -> dict[str, float]:
    """`values` with every connection parameter at its coupling form's neutral value (no coupling)."""
    off = {pid: COUPLINGS[c.form].neutral[slot] for c in twin.connections for slot, pid in c.params.items()}
    return {pid: off.get(pid, v) for pid, v in values.items()}


def _runs(twin: TwinDefinition, ps: ParamSet, v: VerifySettings) -> list[tuple[str, str, float, dict[str, float]]]:
    """(run name, profile, quantile, parameter values) in the order they run."""
    out = [(f"{name}@q{q:g}", name, q, param_values(ps, q)) for name, q in planned(v)]
    if twin.connections:
        out.append((UNCOUPLED, NOMINAL, MEDIAN, neutral(twin, param_values(ps, MEDIAN))))
    return out


def _fail(check: str, message: str) -> CheckResult:
    return CheckResult(check=check, target=COMPOSITE, status="fail", message=message)


def simulate(twin: TwinDefinition, ps: ParamSet, contract: SensorContract, backend_name: str,
             v: VerifySettings) -> tuple[tuple[Simulated, ...], tuple[CheckResult, ...]]:
    backend = load_backend(backend_name)
    try:
        built = backend.build(twin, ps, contract)
    except BuildError as exc:
        why = "; ".join(f"{target}: {problem}" for target, problem in sorted(exc.problems.items()))
        return (), (_fail("build", why),)
    profiles = {p.name: p for p in twin.profiles}
    lacking = sorted({NOMINAL, MAINTAINED} - set(profiles))
    if lacking:
        return (), (CheckResult(check="build", target=COMPOSITE, status="pass"),
                    _fail("simulate", f"no stored profile {', '.join(lacking)}"))
    dt, out, problems = step(twin, ps, v), [], []
    for run_name, name, q, values in _runs(twin, ps, v):
        profile = profiles[name]
        horizon = profile.segments[-1].end
        run = SimulationRun(name=run_name, profile=name, quantile=q, dt=dt, horizon=horizon,
                            steps=math.ceil(horizon / dt))
        if run.steps > v.max_steps:
            problems.append(f"{run.name}: {run.steps:,} steps is more than verify.max_steps ({v.max_steps:,})")
            continue
        try:
            result = backend.simulate(built, profile, values, dt=dt, horizon=horizon, seed=0)
        except Exception as exc:  # noqa: BLE001 - any backend error is a recorded composite failure
            problems.append(f"{run.name}: {type(exc).__name__}: {exc}")
            continue
        out.append(Simulated(run=run, profile=profile, values=values, result=result))
    results = [CheckResult(check="build", target=COMPOSITE, status="pass"),
               _fail("simulate", "; ".join(problems)) if problems
               else CheckResult(check="simulate", target=COMPOSITE, status="pass")]
    return tuple(out), tuple(results)
