"""Step e: write one unit's files and describe them: observations (a FileSource reads them), true states, truth.

- `observations/<unit>.csv`: contract channels only, in kaful.data.file_source's column convention;
- `truth/<unit>.states.csv`: t, then every true state, on the sampling grid (hidden from estimators);
- `truth/<unit>.json`: the unit's synthetic_truth artifact.
Each file's sha256 goes into the dataset's SyntheticUnit, so a changed file is caught on load.
"""

import csv
import io
from collections.abc import Callable
from pathlib import Path

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticTruth, SyntheticUnit, UnitFile
from kaful.data.file_source import write_observations
from kaful.paths import to_rel
from kaful.phases.p7_synthesize.d_observe import Observed
from kaful.runs import sha256_bytes, write_text_atomic

SaveTruth = Callable[[SyntheticTruth, str], ArtifactRef]  # artifact, name under the phase dir -> its reference


def _states_csv(observed: Observed) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    names = sorted(observed.states)
    writer.writerow(["t", *names])
    for k, t in enumerate(observed.grid):
        writer.writerow([repr(float(t)), *(repr(float(observed.states[n][k])) for n in names)])
    return buf.getvalue()


def _file(machine_dir: Path, path: Path) -> UnitFile:
    return UnitFile(path=to_rel(machine_dir, path), sha256=sha256_bytes(path.read_bytes()))


def write_unit(machine_dir: Path, phase_dir: Path, contract: SensorContract, observed: Observed,
               truth: SyntheticTruth, save_truth: SaveTruth) -> SyntheticUnit:
    obs_path = phase_dir / "observations" / f"{truth.unit_id}.csv"
    n_rows = write_observations(obs_path, contract, observed.rows)
    states_path = phase_dir / "truth" / f"{truth.unit_id}.states.csv"
    write_text_atomic(states_path, _states_csv(observed))
    ref = save_truth(truth, f"truth/{truth.unit_id}.json")
    return SyntheticUnit(unit_id=truth.unit_id, spawn_key=truth.spawn_key, observations=_file(machine_dir, obs_path),
                         truth=UnitFile(path=ref.path, sha256=ref.sha256),
                         truth_states=_file(machine_dir, states_path), t_end=truth.t_end, n_rows=n_rows)
