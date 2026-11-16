"""The observation-model interface (M12, approved 2026-11-14): how a binding's damage shows on a channel.

A form maps the damage fraction D / D_limit of one binding to what one SensorContract channel reports:

- `expected(view, params)`: the noise-free reading;
- `sample(view, params, rng)`: a reading with measurement noise (synthesize);
- `log_likelihood(y, view, params)`: log p(y | damage), what M13's estimators weigh particles by.

Forms are pure numpy and broadcast over samples or particles, like the degradation templates. Their parameters
are ParamSet entries (role `observation` for values, `noise` for measurement noise), in the channel's unit.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from kaful.arrays import BoolArray, FloatArray

SlotRole = Literal["observation", "noise"]
ChannelKind = Literal["numeric", "event"]


@dataclass(frozen=True)
class ObservationSlot:
    """A parameter a form reads: its ParamSet role; its unit is always the channel's."""

    name: str
    role: SlotRole
    description: str


@dataclass(frozen=True)
class DamageView:
    """What a form sees of one binding at each sample time.

    `fraction`: D / D_limit (raw, >= 0; a renewed binding is back near 0). `crossed`: D reached D_limit in the
    interval that ends at this sample (a first failure or a threshold renewal).
    """

    fraction: FloatArray
    crossed: BoolArray

    def __post_init__(self) -> None:
        if np.shape(self.fraction) != np.shape(self.crossed):
            raise ValueError("fraction and crossed must have the same shape")


@runtime_checkable
class ObservationForm(Protocol):
    """One parametric observation model."""

    name: ClassVar[str]
    version: ClassVar[str]
    channel_kind: ClassVar[ChannelKind]  # numeric: float or int channels; event: bool or category channels
    params: ClassVar[tuple[ObservationSlot, ...]]

    def expected(self, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        """The noise-free reading (an event form: 1.0 where the event fires, else 0.0)."""
        ...

    def sample(self, view: DamageView, params: Mapping[str, float], rng: np.random.Generator) -> FloatArray:
        """A noisy reading (an event form: 1.0 where the channel reports the event)."""
        ...

    def log_likelihood(self, y: FloatArray, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        """log p(y | damage), elementwise."""
        ...
