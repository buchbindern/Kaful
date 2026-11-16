"""Phase 7 (synthesize) artifacts: observations plus hidden truth, as a pipeline sanity check.

Schema 2 (M12, approved 2026-11-14). The dataset artifact lists each unit's files (observations, truth, true states)
with their sha256 and the unit's observation window, and holds no true value: whatever loads it to find the
observations (a FileSource, M13's estimator) never sees the truth. Each unit's truth is its own `synthetic_truth`
artifact: the true parameters drawn from the priors, every time each binding truly reached its limit, and the profile
applied.

Scoring a twin on data generated from its own ParamSet is partly circular, so every dataset and truth file carries
the disclaimer, and reports repeat it.
"""

from datetime import datetime
from typing import Annotated, Literal, get_args

from pydantic import Field, model_validator

from kaful.artifacts.base import (
    SORTED_STRS,
    ArtifactBase,
    Identifier,
    NonEmptyStr,
    Sha256,
    StrictModel,
    by_attr,
)
from kaful.artifacts.forecast import OperatingProfile
from kaful.paths import RelPath

SyntheticDisclaimer = Literal[
    "Pipeline sanity check, not validation: data generated from the twin's own ParamSet."
]
SYNTHETIC_DISCLAIMER: SyntheticDisclaimer = get_args(SyntheticDisclaimer)[0]


class UnitFile(StrictModel):
    """A file of one unit, relative to the machine directory, with the sha256 of its bytes."""

    path: RelPath
    sha256: Sha256


class SyntheticUnit(StrictModel):
    """One simulated machine: where its observations and its truth are, and its observation window."""

    unit_id: Identifier
    spawn_key: int = Field(ge=0, description="The unit's index in the seed's SeedSequence spawn.")
    observations: UnitFile = Field(description="CSV of SensorContract channels only; what a FileSource reads.")
    truth: UnitFile = Field(description="The unit's synthetic_truth artifact (hidden from estimators).")
    truth_states: UnitFile = Field(description="CSV of the true damage states on the sampling grid (hidden).")
    t_end: float = Field(gt=0.0, description="End of the observation window, in the time unit.")
    n_rows: int = Field(ge=0)


class SyntheticDataset(ArtifactBase):
    """Generated data that follows the SensorContract, with a truth file per unit."""

    schema_version: Literal[2] = 2  # type: ignore[assignment]
    kind: Literal["synthetic_dataset"] = "synthetic_dataset"
    seed: int
    time_unit: NonEmptyStr
    profile: OperatingProfile = Field(description="The stored twin profile every unit was simulated under.")
    horizon: float = Field(gt=0.0, description="Truth is simulated to here; a failure after it is None.")
    sample_interval: float = Field(gt=0.0, description="Spacing of the sampling grid, in the time unit.")
    timestamp_origin: datetime | None = Field(default=None, description=(
        "Timestamp at t = 0 when the contract has no usage clock (t is derived from timestamps); else None."))
    channels: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        description="Channels that carry synthetic values; every other contract channel is left empty.")
    excluded_components: Annotated[tuple[Identifier, ...], SORTED_STRS] = Field(
        default=(), description="Components not in the verified twin (quarantined), so not simulated.")
    flags: Annotated[tuple[NonEmptyStr, ...], SORTED_STRS] = ()
    units: Annotated[tuple[SyntheticUnit, ...], by_attr("unit_id")]
    disclaimer: SyntheticDisclaimer = SYNTHETIC_DISCLAIMER

    @model_validator(mode="after")
    def _check_units(self) -> "SyntheticDataset":
        for unit in self.units:
            if unit.t_end > self.horizon:
                raise ValueError(f"{unit.unit_id}: t_end {unit.t_end} is past the horizon {self.horizon}")
        keys = [u.spawn_key for u in self.units]
        if len(set(keys)) != len(keys):
            raise ValueError("spawn keys must be unique")
        return self


class EventTruth(StrictModel):
    """One failure event of one unit: when it truly happened, and whether the observation window shows it."""

    event: Identifier
    first_time: float | None = Field(description="First time D reached D_limit; None: not before the horizon.")
    limit_times: tuple[float, ...] = Field(default=(), description=(
        "Every time D reached D_limit before the horizon, whatever reset it in between (a threshold renewal, or a "
        "scheduled replacement before it crossed again); the first equals first_time."))
    in_window: bool = Field(description="first_time is set and at most the unit's t_end.")

    @model_validator(mode="after")
    def _check_times(self) -> "EventTruth":
        if list(self.limit_times) != sorted(self.limit_times):
            raise ValueError(f"{self.event}: limit times must be sorted")
        if (self.limit_times[0] if self.limit_times else None) != self.first_time:
            raise ValueError(f"{self.event}: the first limit time is the first failure")
        if self.in_window and self.first_time is None:
            raise ValueError(f"{self.event}: an event in the window needs its time")
        return self


class SyntheticTruth(ArtifactBase):
    """The hidden truth of one synthetic unit."""

    kind: Literal["synthetic_truth"] = "synthetic_truth"
    unit_id: Identifier
    seed: int
    spawn_key: int = Field(ge=0)
    true_params: dict[str, float] = Field(description="param_id -> the value drawn from its prior (sorted).")
    derived_params: dict[str, float] = Field(default_factory=dict, description=(
        "param_id -> a value computed from the drawn ones, not drawn (e.g. life_nominal = D_limit / k)."))
    horizon: float = Field(gt=0.0)
    t_end: float = Field(gt=0.0)
    events: Annotated[tuple[EventTruth, ...], by_attr("event")]
    profile: OperatingProfile = Field(description="The profile applied to this unit, maintenance events included.")
    disclaimer: SyntheticDisclaimer = SYNTHETIC_DISCLAIMER

    @model_validator(mode="after")
    def _check_truth(self) -> "SyntheticTruth":
        for name, values in (("true_params", self.true_params), ("derived_params", self.derived_params)):
            if list(values) != sorted(values):
                raise ValueError(f"{name} must be sorted by param_id")
        if set(self.true_params) & set(self.derived_params):
            raise ValueError("a parameter is either drawn or derived, not both")
        if self.t_end > self.horizon:
            raise ValueError(f"t_end {self.t_end} is past the horizon {self.horizon}")
        for e in self.events:
            if e.in_window != (e.first_time is not None and e.first_time <= self.t_end):
                raise ValueError(f"{e.event}: in_window must say whether first_time is at most t_end")
        return self
