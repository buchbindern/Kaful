"""Step b: simulate one unit's truth through the twin's backend, with its true parameters and no noise.

The step is the shortest median life among the bindings over verify.steps_per_shortest_life, as verify runs, so a
unit is simulated on the grid verify checked; true failure times are resolved to one step. The run goes to the end of
the stored profile (build: profile_horizon_lives x the longest median life); a failure after it is None. Truth is
deterministic given the true parameters: no process noise (M13's estimators add their own).
"""

from collections.abc import Mapping

from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import BuiltTwin, SimulationResult, TwinBackend

MEDIAN = 0.5


def shortest_median_life(twin: TwinDefinition, ps: ParamSet) -> float:
    return min(ps.get(f"{b.component}.life_nominal").prior.quantile(MEDIAN) for b in twin.bindings)


def simulate_unit(backend: TwinBackend, built: BuiltTwin, profile: OperatingProfile, values: Mapping[str, float],
                  dt: float, seed: int) -> SimulationResult:
    return backend.simulate(built, profile, values, dt=dt, horizon=profile.segments[-1].end, seed=seed)
