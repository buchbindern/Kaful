"""Step a: the data sources to estimate, and their rows.

For synthetic data, each unit of the SyntheticDataset is one source: a FileSource over its observations CSV (the
dataset holds no true value, and a FileSource drops columns outside the contract, so truth never reaches an
estimator). The dataset's event channels that carry values are monitored: silence on them over the unit's rows
means no event. Rows are read in order and handed to the filter one at a time.
"""

from dataclasses import dataclass
from pathlib import Path

from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.data.file_source import FileSource
from kaful.data.source import Observation
from kaful.paths import resolve_in


@dataclass(frozen=True)
class Source:
    source_id: str
    spawn_key: int
    path: Path
    monitored: tuple[str, ...]


def monitored_events(dataset: SyntheticDataset, contract: SensorContract) -> tuple[str, ...]:
    """The dataset's channels that are event channels (bool or category)."""
    return tuple(c for c in dataset.channels if contract.channel(c).dtype in ("bool", "category"))


def sources(dataset: SyntheticDataset, contract: SensorContract, machine_dir: Path) -> tuple[Source, ...]:
    monitored = monitored_events(dataset, contract)
    return tuple(Source(source_id=u.unit_id, spawn_key=u.spawn_key, path=resolve_in(machine_dir, u.observations.path),
                        monitored=monitored) for u in dataset.units)


def read_rows(source: Source, contract: SensorContract, dataset: SyntheticDataset) -> tuple[Observation, ...]:
    reader = FileSource(source.path, contract, dataset.time_unit, dataset.timestamp_origin, source.monitored)
    try:
        return tuple(reader)
    finally:
        reader.close()
