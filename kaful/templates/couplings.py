"""Coupling forms (M13b): how one binding's damage scales another's rate. Pure numpy; no other import.

A connection from a source binding to a target binding multiplies the target's template rate by m(x), where
x = clip(D_source / D_limit_source, 0, 1) is how far the source is through its life:

    dD_target/dt = k_target * f(conditions) * m(x)

- hardness_passthrough: a water filter (the source) passes a growing share of the inlet hardness as it exhausts,
  and the scaling rate of what it feeds (a boiler) grows with that hardness:

      m(x) = (r + (1 - r) x) / ((1 + r) / 2),   r in [0, 1]

  r is the hardness fraction a fresh filter passes (bypass and leakage); an exhausted filter passes all of it.
  The divisor makes m average exactly 1 over a source cycle renewed at its limit (x running 0 -> 1 at a constant
  rate), so the target's k = D_limit / life keeps the meaning "life in normal service, the source renewed at its
  limit" (decided 2026-11-16). r = 1 is no coupling (m = 1). Past its limit the source passes everything: x is
  clipped at 1, m(1) = 2 / (1 + r). The linear breakthrough is an assumption: an ion-exchange filter's output
  hardness stays low for most of its capacity and rises near the end; no document gives the curve.
"""

import math
from collections.abc import Mapping
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

from kaful.arrays import FloatArray
from kaful.templates.protocol import TemplateSlot


@runtime_checkable
class CouplingForm(Protocol):
    name: ClassVar[str]
    version: ClassVar[str]
    params: ClassVar[tuple[TemplateSlot, ...]]
    bounds: ClassVar[dict[str, tuple[float, float]]]  # slot -> closed interval its value must lie in
    neutral: ClassVar[dict[str, float]]  # slot values that make m = 1 (no coupling)

    def multiplier(self, fraction: FloatArray | float, params: Mapping[str, FloatArray | float]) -> FloatArray:
        """m(x) for the source's damage fraction x (clipped to [0, 1]); never negative."""
        ...

    def cycle_integral(self, phi: float, params: Mapping[str, float]) -> float:
        """The integral of m(x) dx from 0 to phi, phi in [0, 1]; 1 at phi = 1 (m averages 1 over a cycle)."""
        ...

    def delayed_mean(self, params: Mapping[str, float], factor: float) -> float:
        """The long-run mean of m when the source is renewed at `factor` x its limit instead of at it (factor >= 1),
        the source wearing at a constant rate: (1 + (factor - 1) m(1)) / factor. 1 at factor 1."""
        ...

    def crossing_time(self, target_life: float, source_life: float, params: Mapping[str, float],
                      renews: bool) -> float:
        """When the target reaches its limit from new at nominal, the source new at t = 0 with nominal life
        `source_life`: renewed at its limit (`renews`), or left at x = 1 after it. Uncoupled, it is target_life."""
        ...


def _clip_fraction(x: FloatArray | float) -> FloatArray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


class HardnessPassthrough:
    name: ClassVar[str] = "hardness_passthrough"
    version: ClassVar[str] = "1"
    params: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("residual_fraction", "[]", "hardness fraction a fresh source passes (1: no coupling)"),)
    bounds: ClassVar[dict[str, tuple[float, float]]] = {"residual_fraction": (0.0, 1.0)}
    neutral: ClassVar[dict[str, float]] = {"residual_fraction": 1.0}

    @staticmethod
    def _r(params: Mapping[str, FloatArray | float]) -> FloatArray:
        return np.clip(np.asarray(params["residual_fraction"], dtype=np.float64), 0.0, 1.0)

    def multiplier(self, fraction: FloatArray | float, params: Mapping[str, FloatArray | float]) -> FloatArray:
        r = self._r(params)
        return np.asarray((r + (1.0 - r) * _clip_fraction(fraction)) / ((1.0 + r) / 2.0), dtype=np.float64)

    def cycle_integral(self, phi: float, params: Mapping[str, float]) -> float:
        r, p = float(self._r(params)), min(max(phi, 0.0), 1.0)
        return (r * p + (1.0 - r) * p * p / 2.0) / ((1.0 + r) / 2.0)

    def delayed_mean(self, params: Mapping[str, float], factor: float) -> float:
        if factor < 1.0:
            raise ValueError("factor must be >= 1")
        return (self.cycle_integral(1.0, params) + (factor - 1.0) * float(self.multiplier(1.0, params))) / factor

    def _phase(self, share: float, r: float) -> float:
        """phi in [0, 1] with cycle_integral(phi) = share (share in [0, 1])."""
        if r >= 1.0:
            return share
        a, c = (1.0 - r) / 2.0, share * (1.0 + r) / 2.0  # a phi^2 + r phi - c = 0
        return (-r + math.sqrt(r * r + 4.0 * a * c)) / (2.0 * a)

    def crossing_time(self, target_life: float, source_life: float, params: Mapping[str, float],
                      renews: bool) -> float:
        if target_life <= 0.0 or source_life <= 0.0:
            raise ValueError("lives must be > 0")
        r = float(self._r(params))
        cycles = target_life / source_life
        if not renews and cycles > 1.0:  # past one source life: x stays at 1
            return source_life + (target_life - source_life) / float(self.multiplier(1.0, params))
        whole = math.floor(cycles) if renews else 0
        return (whole + self._phase(cycles - whole, r)) * source_life


COUPLINGS: dict[str, type[HardnessPassthrough]] = {c.name: c for c in (HardnessPassthrough,)}
