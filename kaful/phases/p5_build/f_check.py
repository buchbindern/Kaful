"""Step f: the build gates (hard rule 2). Each failure is a line in the phase record, and the run stops.

The checks live in kaful.twin_checks (verify re-runs them):
- every full-model component of the ParamSet has exactly one binding, with k and D_limit;
- every param_id a binding, input or rule names exists in the ParamSet;
- units: D_limit's unit is D's; k's is "<D unit> / <time unit>"; a channel input has its condition's
  dimension (checked when wired); every connection's ends are pint-compatible;
- a clean or descale rule has its recovery parameter;
- observation models (M12): channel in the contract, form suits the channel, slot parameters exist in the channel's
  unit, one channel's linear observations share y_healthy and noise_sd.
"""

from collections.abc import Sequence

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.twin_checks import observation_problems, static_problems


def check(twin: TwinDefinition, ps: ParamSet, contract: SensorContract, missing: Sequence[str]) -> tuple[str, ...]:
    return (*missing, *(p.message for p in static_problems(twin, ps)),
            *(p.message for p in observation_problems(twin, ps, contract)))
