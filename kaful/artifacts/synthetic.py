"""Phase 7 (synthesize) artifact: observations plus hidden truth. First cut (M12 refines)."""

from typing import Annotated, Literal, get_args

from kaful.artifacts.base import ArtifactBase, Identifier, StrictModel, by_attr
from kaful.artifacts.forecast import OperatingProfile
from kaful.paths import RelPath

SyntheticDisclaimer = Literal[
    "Pipeline sanity check, not validation: data generated from the twin's own ParamSet."
]
SYNTHETIC_DISCLAIMER: SyntheticDisclaimer = get_args(SyntheticDisclaimer)[0]


class SyntheticUnit(StrictModel):
    """One simulated machine: its observations, true states and true failure times."""

    unit_id: Identifier
    observations: RelPath
    truth_states: RelPath
    true_failure_times: dict[Identifier, float | None]
    true_params: dict[str, float]


class SyntheticDataset(ArtifactBase):
    """Generated data that follows the SensorContract, with a truth file per unit."""

    kind: Literal["synthetic_dataset"] = "synthetic_dataset"
    seed: int
    profile: OperatingProfile
    units: Annotated[tuple[SyntheticUnit, ...], by_attr("unit_id")]
    disclaimer: SyntheticDisclaimer = SYNTHETIC_DISCLAIMER
