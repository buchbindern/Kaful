"""The first template families (M9, approved 2026-11-09): accumulation, arrhenius, bearing_fatigue.

Each is a state-independent damage law in the normalised form of the protocol,

    dD/dt = k * f(conditions; params),   f(nominal conditions) = 1,

so k = D_limit / (life * f(nominal)) and life = D_limit / (k * f(conditions)) in closed form, for any
D unit. A condition a caller does not pass is taken at its nominal value (f's factor for it is 1).
Non-linear behaviour of what a sensor sees (e.g. a filter's pressure drop growing with clogging) belongs
to the observation model, not to D. Pure numpy; no other import.

- accumulation: f = (load / load_nominal) ** load_exponent. Clogging, fouling, scaling, residue,
  exhaustion, wear: damage grows with a throughput or load (water volume, air flow, products). D's unit
  is the component's D_limit unit (ANY_DIMENSION): mm of scale, g of deposit, L treated, a fraction.
- arrhenius: f = exp(activation_energy / R * (1 / temperature_nominal - 1 / temperature)). Thermal
  ageing of oil, grease and insulation; overheating.
- bearing_fatigue: f = (load / load_nominal) ** life_exponent * (speed / speed_nominal). Rolling-contact
  fatigue (ISO 281: L10 is proportional to (C / P) ** p revolutions); D is Miner's damage fraction.
"""

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from kaful.arrays import FloatArray
from kaful.templates.protocol import ANY_DIMENSION, TemplateSlot

GAS_CONSTANT = 8.314462618  # J / (mol K), CODATA 2018
RESET_ACTIONS = ("replace",)  # maintenance actions that renew the part: D back to 0


def _ratio(conditions: Mapping[str, FloatArray] | Mapping[str, float], name: str,
           nominal: FloatArray | float) -> FloatArray:
    if name not in conditions:
        return np.ones_like(np.asarray(nominal, dtype=np.float64))
    return np.clip(np.asarray(conditions[name], dtype=np.float64), 0.0, None) / np.asarray(nominal, dtype=np.float64)


class _StateFree:
    """Closed-form calibration and life for dD/dt = k * f, f independent of D."""

    name: ClassVar[str]
    version: ClassVar[str]
    params: ClassVar[tuple[TemplateSlot, ...]]
    conditions: ClassVar[tuple[TemplateSlot, ...]]

    def multiplier(self, conditions: Mapping[str, FloatArray] | Mapping[str, float],
                   params: Mapping[str, FloatArray] | Mapping[str, float]) -> FloatArray:
        raise NotImplementedError

    def rate(self, damage: FloatArray, conditions: Mapping[str, FloatArray],
             params: Mapping[str, FloatArray]) -> FloatArray:
        k = np.asarray(params["k"], dtype=np.float64)
        out = k * self.multiplier(conditions, params)
        return np.broadcast_to(out, np.broadcast_shapes(np.shape(damage), np.shape(out))).astype(np.float64)

    def calibrate_k(self, life: float, nominal: Mapping[str, float], params: Mapping[str, float],
                    damage_limit: float) -> float:
        if life <= 0 or damage_limit <= 0:
            raise ValueError("life and damage_limit must be > 0")
        return float(damage_limit / (life * float(self.multiplier(nominal, params))))

    def life_at(self, k: float, conditions: Mapping[str, float], params: Mapping[str, float],
                damage_limit: float) -> float:
        speed = k * float(self.multiplier(conditions, params))
        return float("inf") if speed <= 0 else float(damage_limit / speed)

    def apply_maintenance(self, damage: FloatArray, action: str,
                          params: Mapping[str, FloatArray]) -> FloatArray:
        return np.zeros_like(damage) if action in RESET_ACTIONS else damage


class Accumulation(_StateFree):
    name: ClassVar[str] = "accumulation"
    version: ClassVar[str] = "1"
    damage_dimension: ClassVar[str] = ANY_DIMENSION
    params: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("k", f"{ANY_DIMENSION} / [time]", "damage per time unit at nominal load"),
        TemplateSlot("load_exponent", "[]", "how damage rate scales with load (1: in proportion)"),
        TemplateSlot("load_nominal", ANY_DIMENSION, "nominal driving load (throughput, flow, products)"))
    conditions: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("load", ANY_DIMENSION, "driving load, in load_nominal's unit"),)

    def multiplier(self, conditions: Mapping[str, FloatArray] | Mapping[str, float],
                   params: Mapping[str, FloatArray] | Mapping[str, float]) -> FloatArray:
        ratio = _ratio(conditions, "load", params["load_nominal"])
        return np.power(ratio, np.asarray(params["load_exponent"], dtype=np.float64))


class Arrhenius(_StateFree):
    name: ClassVar[str] = "arrhenius"
    version: ClassVar[str] = "1"
    damage_dimension: ClassVar[str] = ANY_DIMENSION
    params: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("k", f"{ANY_DIMENSION} / [time]", "damage per time unit at the nominal temperature"),
        TemplateSlot("activation_energy", "[mass] * [length] ** 2 / [time] ** 2 / [substance]", "Ea, J/mol"),
        TemplateSlot("temperature_nominal", "[temperature]", "nominal absolute temperature, K"))
    conditions: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("temperature", "[temperature]", "absolute temperature, K"),)

    def multiplier(self, conditions: Mapping[str, FloatArray] | Mapping[str, float],
                   params: Mapping[str, FloatArray] | Mapping[str, float]) -> FloatArray:
        t_nom = np.asarray(params["temperature_nominal"], dtype=np.float64)
        if "temperature" not in conditions:
            return np.ones_like(t_nom)
        t = np.asarray(conditions["temperature"], dtype=np.float64)
        ea = np.asarray(params["activation_energy"], dtype=np.float64)
        return np.exp(ea / GAS_CONSTANT * (1.0 / t_nom - 1.0 / t))


class BearingFatigue(_StateFree):
    name: ClassVar[str] = "bearing_fatigue"
    version: ClassVar[str] = "1"
    damage_dimension: ClassVar[str] = "[]"
    params: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("k", "[] / [time]", "Miner damage per time unit at nominal load and speed"),
        TemplateSlot("life_exponent", "[]", "ISO 281 p: 3 for ball, 10/3 for roller bearings"),
        TemplateSlot("load_nominal", "[mass] * [length] / [time] ** 2", "nominal equivalent load P"),
        TemplateSlot("speed_nominal", "1 / [time]", "nominal speed"))
    conditions: ClassVar[tuple[TemplateSlot, ...]] = (
        TemplateSlot("load", "[mass] * [length] / [time] ** 2", "equivalent dynamic load P"),
        TemplateSlot("speed", "1 / [time]", "shaft speed"))

    def multiplier(self, conditions: Mapping[str, FloatArray] | Mapping[str, float],
                   params: Mapping[str, FloatArray] | Mapping[str, float]) -> FloatArray:
        load = _ratio(conditions, "load", params["load_nominal"])
        speed = _ratio(conditions, "speed", params["speed_nominal"])
        return np.power(load, np.asarray(params["life_exponent"], dtype=np.float64)) * speed


FAMILIES: dict[str, type[_StateFree]] = {t.name: t for t in (Accumulation, Arrhenius, BearingFatigue)}
