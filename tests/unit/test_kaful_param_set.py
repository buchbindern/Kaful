"""ParamSet rules: point values (delta), calibration references, source counts."""

from typing import Any

import pytest
from pydantic import ValidationError

from kaful.artifacts.base import SOURCE_KINDS
from kaful.artifacts.distributions import Delta, Lognormal
from kaful.artifacts.param_set import DELTA_ROLES, DELTA_SOURCES, LifeStatement, Parameter, ParamSet
from tests.harness import kaful_factories as f

ROLES = ("rate", "threshold", "capacity", "nominal_input", "initial_state", "exponent", "noise")
EVIDENCE = {"evidence": (f.EVIDENCE,)}


def _param(role: str, source: str, prior: Any) -> Parameter:
    extra = EVIDENCE if source == "manual" else {}
    return Parameter(name="p", component="seal", role=role, unit="1", prior=prior,  # type: ignore[arg-type]
                     source=source, ref="machine.toml" if source == "deployment" else "ref", confidence=0.5, **extra)  # type: ignore[arg-type]


def test_delta_rule_sets_match_the_decision() -> None:
    assert DELTA_SOURCES == {"manual", "standard", "default"}
    assert DELTA_ROLES == {"threshold", "initial_state", "capacity", "nominal_input"}


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("source", SOURCE_KINDS)
def test_delta_prior_needs_allowed_source_and_role(role: str, source: str) -> None:
    allowed = role in DELTA_ROLES and source in DELTA_SOURCES
    if allowed:
        assert isinstance(_param(role, source, Delta(value=1.0)).prior, Delta)
    else:
        with pytest.raises(ValidationError, match="delta"):
            _param(role, source, Delta(value=1.0))


@pytest.mark.parametrize("role", ["rate", "exponent", "noise"])
def test_rates_exponents_and_noise_take_real_distributions(role: str) -> None:
    assert _param(role, "standard", Lognormal(median=1.0, sigma_log=0.2)).role == role


def test_delta_posterior_needs_delta_prior() -> None:
    with pytest.raises(ValidationError, match="delta posterior"):
        Parameter(name="k", component="seal", role="rate", unit="1/h",
                  prior=Lognormal(median=1.0, sigma_log=0.2), posterior=Delta(value=1.0),
                  source="default", ref="x", confidence=0.5)


def test_calibrated_parameters_must_reference_a_life_statement() -> None:
    params = f.param_set()
    with pytest.raises(ValidationError, match="life statement"):
        ParamSet.model_validate({**params.model_dump(), "life_statements": ()})


def test_service_intervals_are_lower_bounds() -> None:
    with pytest.raises(ValidationError, match="lower bounds"):
        LifeStatement(id="s", component="seal", kind="service_interval", value=100.0, unit="h",
                      bound="point", source="default", ref="x", confidence=0.5)


def test_parameters_sorted_by_param_id() -> None:
    params = f.param_set()
    ids = [p.param_id for p in params.parameters]
    assert ids == sorted(ids)
    with pytest.raises(ValidationError, match="sorted"):
        ParamSet.model_validate({**params.model_dump(), "parameters": params.model_dump()["parameters"][::-1]})


def test_get_and_counts_by_source() -> None:
    params = f.param_set()
    assert params.get("seal.wear_rate").role == "rate"
    with pytest.raises(KeyError):
        params.get("seal.ghost")
    counts = params.counts_by_source()
    assert tuple(counts) == SOURCE_KINDS
    assert sum(counts.values()) == len(params.parameters)
    assert counts["calibrated"] == 1 and counts["handbook"] == 0
