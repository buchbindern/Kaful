"""Interfaces: every member is documented, toy implementations conform, helpers work."""

import inspect
from collections.abc import Iterator, Mapping
from typing import ClassVar

import numpy as np
import pytest

from kaful.arrays import FloatArray
from kaful.artifacts.sensor_contract import SensorContract
from kaful.backends.base import BuildError, BuiltTwin, Estimator, TwinBackend
from kaful.data.source import DataSource, Observation, check_observation
from kaful.phases import Phase
from kaful.templates.protocol import DegradationTemplate, TemplateSlot, health_from_damage
from tests.harness import kaful_factories as f

PROTOCOLS = [DataSource, DegradationTemplate, TwinBackend, Estimator, BuiltTwin, Phase]


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=lambda p: p.__name__)
def test_every_interface_member_has_a_docstring(protocol: type) -> None:
    assert inspect.getdoc(protocol)
    members = {name: value for name, value in vars(protocol).items()
               if not name.startswith("_") or name == "__iter__"}
    callables = {n: v for n, v in members.items() if callable(v) or isinstance(v, property)}
    assert callables, f"{protocol.__name__} declares no methods"
    undocumented = [n for n, v in callables.items() if not inspect.getdoc(v)]
    assert undocumented == []


def test_template_reserves_apply_maintenance() -> None:
    assert "apply_maintenance" in vars(DegradationTemplate)


class ToySource:
    """Two observations from a fixed list."""

    def __init__(self, contract: SensorContract) -> None:
        self._contract = contract

    @property
    def contract(self) -> SensorContract:
        return self._contract

    def __iter__(self) -> Iterator[Observation]:
        yield Observation(t=0.0, values={"flow_rate": 20.0, "run_hours": 0.0})
        yield Observation(t=1.0, values={"flow_rate": 19.5, "run_hours": 1.0})

    def close(self) -> None:
        pass


class LinearWear:
    """dD/dt = k: the simplest template, for conformance only."""

    name: ClassVar[str] = "linear"
    version: ClassVar[str] = "0"
    damage_dimension: ClassVar[str] = "[length]"
    params: ClassVar[tuple[TemplateSlot, ...]] = (TemplateSlot("k", "[length]/[time]", "wear rate"),)
    conditions: ClassVar[tuple[TemplateSlot, ...]] = ()

    def rate(self, damage: FloatArray, conditions: Mapping[str, FloatArray],
             params: Mapping[str, FloatArray]) -> FloatArray:
        return np.broadcast_to(params["k"], np.shape(damage)).astype(np.float64)

    def calibrate_k(self, life: float, nominal: Mapping[str, float], params: Mapping[str, float],
                    damage_limit: float) -> float:
        return damage_limit / life

    def life_at(self, k: float, conditions: Mapping[str, float], params: Mapping[str, float],
                damage_limit: float) -> float:
        return damage_limit / k

    def apply_maintenance(self, damage: FloatArray, action: str,
                          params: Mapping[str, FloatArray]) -> FloatArray:
        return np.zeros_like(damage) if action == "replace_seal" else damage


def test_toy_implementations_satisfy_the_protocols() -> None:
    source = ToySource(f.sensor_contract())
    assert isinstance(source, DataSource)
    assert [o.t for o in source] == [0.0, 1.0]
    template = LinearWear()
    assert isinstance(template, DegradationTemplate)
    k = template.calibrate_k(2000.0, {}, {}, damage_limit=1.5)
    assert template.life_at(k, {}, {}, damage_limit=1.5) == pytest.approx(2000.0)
    damage = np.array([0.3, 1.0])
    assert np.all(template.rate(damage, {}, {"k": np.array(k)}) >= 0)
    np.testing.assert_array_equal(template.apply_maintenance(damage, "replace_seal", {}), [0.0, 0.0])
    assert not isinstance(object(), TwinBackend)


def test_health_is_derived_from_damage() -> None:
    health = health_from_damage(np.array([0.0, 0.75, 1.5, 2.0]), 1.5)
    np.testing.assert_allclose(health, [1.0, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError):
        health_from_damage(np.array([0.0]), 0.0)


def test_check_observation_reports_sorted_violations() -> None:
    contract = f.sensor_contract()
    obs = Observation(t=1.0, values={
        "zeta": 1.0, "flow_rate": 99.0, "maintenance": "repaint", "run_hours": True, "suction_temp": None})
    found = [(v.channel, v.problem) for v in check_observation(contract, obs)]
    assert found == [("flow_rate", "out_of_range"), ("maintenance", "unknown_category"),
                     ("run_hours", "wrong_dtype"), ("zeta", "unknown_channel")]
    clean = Observation(t=1.0, values={"flow_rate": 20, "maintenance": "service"})
    assert check_observation(contract, clean) == ()


def test_build_error_sorts_problems() -> None:
    error = BuildError({"seal": "no template", "impeller": "bad unit"})
    assert list(error.problems) == ["impeller", "seal"]
    assert str(error) == "impeller: bad unit; seal: no template"
