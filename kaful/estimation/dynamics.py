"""What a backend gives the estimators: a vectorized step over a group of bindings under one operating profile.

The backend applies each binding's template rate and the profile's scheduled maintenance (known inputs). It does
not renew a binding at D_limit and adds no noise: the estimators add process noise, detect crossings and renew
(`cloud.advance`), so filtering and forecasting treat crossings the same way.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from kaful.arrays import FloatArray


@dataclass(frozen=True)
class BindingSpec:
    """One binding as the estimators see it."""

    id: str
    event: str  # the failure event's name
    limit_param: str  # param_id of D_limit
    renews_at_limit: bool  # a threshold maintenance rule replaces the part when D reaches D_limit


class Dynamics(Protocol):
    """A vectorized model step over some bindings; parameters are fixed values or one value per particle."""

    @property
    def bindings(self) -> tuple[BindingSpec, ...]:
        """The bindings stepped, sorted by id."""
        ...

    @property
    def fixed(self) -> Mapping[str, float]:
        """param_id -> value for every parameter the bindings read that is not carried per particle."""
        ...

    def step(self, damage: Mapping[str, FloatArray], params: Mapping[str, FloatArray], t: float,
             dt: float) -> tuple[dict[str, FloatArray], frozenset[str]]:
        """D after [t, t + dt) (drift and scheduled maintenance), and the bindings a maintenance reset this step."""
        ...

    def reset_times(self, binding: str) -> tuple[float, ...]:
        """When the profile's scheduled maintenance acts on the binding, sorted (it fires in the step containing
        each time)."""
        ...

    def rates(self, params: Mapping[str, FloatArray], t: float) -> dict[str, FloatArray]:
        """Each binding's dD/dt at time t (templates are state-free: the rate does not depend on D)."""
        ...
