"""The simulation step every phase that steps a twin uses (synthesize, estimate): the shortest median life among
the twin's bindings over verify.steps_per_shortest_life, so data is generated and estimated on the grid verify
checked."""

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition

MEDIAN = 0.5


def shortest_median_life(twin: TwinDefinition, ps: ParamSet) -> float:
    return min(ps.get(f"{b.component}.life_nominal").prior.quantile(MEDIAN) for b in twin.bindings)


def simulation_step(twin: TwinDefinition, ps: ParamSet, steps_per_shortest_life: int) -> float:
    return shortest_median_life(twin, ps) / steps_per_shortest_life
