"""Step a: one unit's true parameters, drawn from the ParamSet priors, and its observation window.

Only the parameters the twin reads are drawn (binding slots, D_limit, maintenance recovery, nominal inputs,
observation parameters), in sorted param_id order; a delta prior gives its value. A parameter that restates another
is derived, not drawn: `life_nominal` = D_limit / k (the k and life_nominal priors are independent, PLAN M12), so the
unit's truth stays consistent with what the twin runs.

Unit i draws from SeedSequence(seed, spawn_key=(i,)), split into three streams (parameters, measurement noise,
window), so a unit's draws never depend on how many units there are or on what another stream used.
"""

from dataclasses import dataclass

import numpy as np

from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import SynthesizeSettings

PARAMS, NOISE, WINDOW = 0, 1, 2


@dataclass(frozen=True)
class Draw:
    unit_id: str
    spawn_key: int
    true_params: dict[str, float]  # param_id -> value, sorted
    derived_params: dict[str, float]
    window_fraction: float
    noise: np.random.SeedSequence


def unit_id(i: int, n: int) -> str:
    return f"unit_{i:0{max(2, len(str(n - 1)))}d}"


def read_params(twin: TwinDefinition) -> tuple[str, ...]:
    """Every param_id the twin reads, sorted."""
    ids = {b.threshold_param for b in twin.bindings} | {p for b in twin.bindings for p in b.params.values()}
    ids |= {r.recovery_param for r in twin.maintenance if r.recovery_param is not None}
    ids |= {i.nominal_param for i in twin.external_inputs if i.nominal_param is not None}
    ids |= {p for o in twin.observations for p in o.params.values()}
    return tuple(sorted(ids))


def _derived(twin: TwinDefinition, ps: ParamSet, true: dict[str, float]) -> dict[str, float]:
    known = {x.param_id for x in ps.parameters}
    out = {}
    for b in twin.bindings:
        life, k = f"{b.component}.life_nominal", b.params.get("k")
        if life in known and k in true and life not in true and true[k] > 0:
            out[life] = true[b.threshold_param] / true[k]
    return dict(sorted(out.items()))


def draw_unit(ps: ParamSet, twin: TwinDefinition, s: SynthesizeSettings, i: int) -> Draw:
    streams = np.random.SeedSequence(s.seed, spawn_key=(i,)).spawn(3)
    rng = np.random.default_rng(streams[PARAMS])
    true = {}
    for pid in read_params(twin):
        prior = ps.get(pid).prior
        true[pid] = prior.value if isinstance(prior, Delta) else float(prior.sample(rng, 1)[0])
    low, high = s.window_fraction
    fraction = float(np.random.default_rng(streams[WINDOW]).uniform(low, high))
    return Draw(unit_id=unit_id(i, s.n_units), spawn_key=i, true_params=true,
                derived_params=_derived(twin, ps, true), window_fraction=fraction, noise=streams[NOISE])
