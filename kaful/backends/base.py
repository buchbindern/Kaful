"""The TwinBackend interface and the library-neutral types it exchanges.

An adapter (ProgPy first, in kaful/backends/progpy/) turns a TwinDefinition into a
runnable twin. No signature here mentions a library type; backend objects stay
behind the opaque BuiltTwin handle. Swapping libraries means writing one adapter.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable

from kaful.arrays import FloatArray
from kaful.artifacts.forecast import EventForecast, OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.data.source import Observation


class BuildError(RuntimeError):
    """A TwinDefinition could not be built; `problems` maps component id to reason."""

    def __init__(self, problems: Mapping[str, str]) -> None:
        self.problems = dict(sorted(problems.items()))
        super().__init__("; ".join(f"{k}: {v}" for k, v in self.problems.items()))


class BuiltTwin(Protocol):
    """Opaque runnable twin. Backend-specific objects stay behind this handle."""

    @property
    def definition(self) -> TwinDefinition:
        """The definition this twin was built from."""
        ...

    @property
    def state_names(self) -> tuple[str, ...]:
        """Damage states (and any auxiliary states), sorted."""
        ...

    @property
    def event_names(self) -> tuple[str, ...]:
        """Failure events, sorted."""
        ...


@dataclass(frozen=True)
class StateBelief:
    """Weighted particles over damage states (and augmented parameters) at time t."""

    t: float
    particles: Mapping[str, FloatArray]  # state name or param_id -> (n,) array
    weights: FloatArray  # (n,), sums to 1


@dataclass(frozen=True)
class SimulationResult:
    """A forward simulation: states and outputs over time, and when events fired."""

    times: FloatArray
    states: Mapping[str, FloatArray]
    outputs: Mapping[str, FloatArray]
    event_times: Mapping[str, float | None]  # None = no event before the horizon


class Estimator(Protocol):
    """A stateful filter fed one observation at a time, so batch and streaming match."""

    def update(self, obs: Observation) -> StateBelief:
        """Assimilate one observation and return the posterior belief."""
        ...

    @property
    def belief(self) -> StateBelief:
        """The current belief, without assimilating anything."""
        ...


@runtime_checkable
class TwinBackend(Protocol):
    """Adapter from a TwinDefinition to a simulation library."""

    name: ClassVar[str]

    def build(self, twin: TwinDefinition, params: ParamSet, contract: SensorContract) -> BuiltTwin:
        """Construct a runnable twin. Deterministic and LLM-free.

        Raises BuildError with per-component detail. Escape-hatch code is executed
        only through the sandbox.
        """
        ...

    def simulate(
        self,
        twin: BuiltTwin,
        profile: OperatingProfile,
        param_values: Mapping[str, float],
        *,
        dt: float,
        horizon: float,
        seed: int,
    ) -> SimulationResult:
        """Forward-simulate with fixed parameter values (param_id -> value) under a profile,
        applying its scheduled maintenance events."""
        ...

    def estimate(
        self,
        twin: BuiltTwin,
        params: ParamSet,
        *,
        method: Literal["particle_filter", "ukf"],
        augment: Sequence[str],
        n_particles: int,
        seed: int,
    ) -> Estimator:
        """Start an estimator from the priors in `params`.

        `augment` lists param_ids estimated jointly with the state, so a machine's own
        data can update their posteriors.
        """
        ...

    def predict(
        self,
        twin: BuiltTwin,
        belief: StateBelief,
        profile: OperatingProfile,
        *,
        horizon: float,
        n_samples: int,
        seed: int,
    ) -> tuple[EventForecast, ...]:
        """Sample times of event from `belief` under `profile`, keeping every sample and
        the censoring counts. From the prior belief this is the open-loop baseline."""
        ...
