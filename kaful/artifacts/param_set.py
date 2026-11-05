"""Phase 4 (parameterize) artifact: every number as a prior with provenance.

Nothing in the twin, simulator or forecast uses a literal constant; it reads the
ParamSet. A point value (`Delta`) is allowed only for definitional values: the
source must be manual, standard or default, and the role must be threshold,
initial_state, capacity or nominal_input. Rates, exponents and noise always get a
real distribution.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SOURCE_KINDS,
    ArtifactBase,
    Identifier,
    NonEmptyStr,
    Provenanced,
    SourceKind,
    by_attr,
    sorted_unique,
)
from kaful.artifacts.distributions import Delta, Distribution

ParamRole = Literal[
    "rate", "threshold", "capacity", "nominal_input", "initial_state", "exponent", "noise"
]

DELTA_SOURCES: frozenset[SourceKind] = frozenset({"manual", "standard", "default"})
DELTA_ROLES: frozenset[ParamRole] = frozenset(
    {"threshold", "initial_state", "capacity", "nominal_input"}
)


class Parameter(Provenanced):
    """One parameter: {name, component, unit, prior, posterior?, source, ref, confidence}."""

    name: Identifier
    component: Identifier = Field(description="Component id, or 'machine' for machine-level.")
    role: ParamRole
    unit: NonEmptyStr
    prior: Distribution
    posterior: Distribution | None = None

    @property
    def param_id(self) -> str:
        return f"{self.component}.{self.name}"

    @model_validator(mode="after")
    def _check_point_values(self) -> "Parameter":
        if isinstance(self.prior, Delta):
            if self.source not in DELTA_SOURCES:
                raise ValueError(
                    f"{self.param_id}: a delta prior needs source in {sorted(DELTA_SOURCES)}, "
                    f"got {self.source!r}"
                )
            if self.role not in DELTA_ROLES:
                raise ValueError(
                    f"{self.param_id}: role {self.role!r} needs a real distribution, not delta"
                )
        if isinstance(self.posterior, Delta) and not isinstance(self.prior, Delta):
            raise ValueError(f"{self.param_id}: a delta posterior needs a delta prior")
        return self


class LifeStatement(Provenanced):
    """A documented life or interval used to calibrate rates (PLAN M9).

    Service and replacement intervals are conservative, so they are lower bounds on
    life (`bound="lower"`), not failure times (hard rule 5).
    """

    id: Identifier
    component: Identifier
    kind: Literal["rated_life", "mtbf", "service_interval", "replacement_interval"]
    value: float = Field(gt=0.0)
    unit: NonEmptyStr
    bound: Literal["point", "lower"]

    @model_validator(mode="after")
    def _check_bound(self) -> "LifeStatement":
        if self.kind in ("service_interval", "replacement_interval") and self.bound != "lower":
            raise ValueError(f"{self.id}: service and replacement intervals are lower bounds")
        return self


class ParamSet(ArtifactBase):
    """All parameters of the machine, sorted by param_id."""

    kind: Literal["param_set"] = "param_set"
    time_unit: NonEmptyStr = Field(description="Must equal the machine's confirmed time_unit (machine.toml, M9).")
    parameters: Annotated[
        tuple[Parameter, ...], sorted_unique(lambda p: p.param_id, "param_id")
    ]
    life_statements: Annotated[tuple[LifeStatement, ...], by_attr("id")] = ()

    @model_validator(mode="after")
    def _check_calibration_refs(self) -> "ParamSet":
        life_ids = {s.id for s in self.life_statements}
        for p in self.parameters:
            if p.source == "calibrated" and p.ref not in life_ids:
                raise ValueError(f"{p.param_id}: calibrated ref {p.ref!r} is not a life statement id")
        return self

    def get(self, param_id: str) -> Parameter:
        """Return the parameter with `param_id`; KeyError if absent."""
        for p in self.parameters:
            if p.param_id == param_id:
                return p
        raise KeyError(param_id)

    def counts_by_source(self) -> dict[SourceKind, int]:
        """Parameter count per source kind, in SOURCE_KINDS order (zeros included)."""
        return {s: sum(p.source == s for p in self.parameters) for s in SOURCE_KINDS}
