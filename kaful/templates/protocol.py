"""The degradation-template interface.

A template is a pure-numpy parametric law for a physical damage variable D, e.g.
mm of wear, crack length or scale thickness:

    dD/dt = k * f(conditions, D) >= 0,   D(0) = 0

Physics or standards give f; k is calibrated so that nominal conditions take D
from 0 to D_limit (a ParamSet threshold) in the documented life. Health is derived,
never stored: health = 1 - D / D_limit, which lies in [0, 1] and starts at 1.0
(hard rule 6). Rates are per the machine's time_unit, in D's unit.

Templates import nothing beyond numpy. The library itself arrives in M10.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from kaful.arrays import FloatArray


@dataclass(frozen=True)
class TemplateSlot:
    """A named input of a template: a parameter it binds or a condition it reads."""

    name: str
    dimension: str  # pint dimensionality, e.g. "[length] / [time]" or "[temperature]"
    description: str


@runtime_checkable
class DegradationTemplate(Protocol):
    """Parametric damage law dD/dt = k * f(conditions, D), in plain numpy.

    Every array argument broadcasts, so one call evaluates all particles or samples.
    """

    name: ClassVar[str]
    version: ClassVar[str]
    damage_dimension: ClassVar[str]  # pint dimensionality of D, e.g. "[length]"
    params: ClassVar[tuple[TemplateSlot, ...]]  # bound to ParamSet entries (k, exponents…)
    conditions: ClassVar[tuple[TemplateSlot, ...]]  # operating inputs f reads (load, speed…)

    def rate(
        self,
        damage: FloatArray,
        conditions: Mapping[str, FloatArray],
        params: Mapping[str, FloatArray],
    ) -> FloatArray:
        """dD/dt in D's unit per time_unit; never negative."""
        ...

    def calibrate_k(
        self,
        life: float,
        nominal: Mapping[str, float],
        params: Mapping[str, float],
        damage_limit: float,
    ) -> float:
        """The k for which D grows from 0 to `damage_limit` in exactly `life` time
        units under constant `nominal` conditions."""
        ...

    def life_at(
        self,
        k: float,
        conditions: Mapping[str, float],
        params: Mapping[str, float],
        damage_limit: float,
    ) -> float:
        """Time for D to grow from 0 to `damage_limit` under constant conditions.

        The inverse of `calibrate_k`; verify uses it to check degradation speed
        against the calibrated life.
        """
        ...

    def apply_maintenance(
        self, damage: FloatArray, action: str, params: Mapping[str, FloatArray]
    ) -> FloatArray:
        """Damage immediately after a maintenance `action` (an event-channel category,
        e.g. "descale" or "replace"). Unknown actions leave D unchanged.
        Implemented per template in M10."""
        ...


def health_from_damage(damage: FloatArray, damage_limit: float | FloatArray) -> FloatArray:
    """health = 1 - D / D_limit, clipped to [0, 1]. D_limit must be positive."""
    limit = np.asarray(damage_limit, dtype=np.float64)
    if np.any(limit <= 0.0):
        raise ValueError("damage_limit must be > 0")
    return np.clip(1.0 - np.asarray(damage, dtype=np.float64) / limit, 0.0, 1.0)
