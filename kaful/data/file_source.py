"""FileSource: a DataSource over a CSV file, and the writer that makes such files (M12).

Column convention, shared by the writer and the reader:
- the contract's `time_column` (ISO 8601 timestamps, or empty), then every contract channel, in name order;
- an empty cell is a channel that did not report (None); bools are `true` / `false`; floats use Python's shortest
  round-trip form, so writing is deterministic and reading gives back the same values.

`t` comes from the contract's usage clock when it has one. Otherwise it is derived from the timestamps: elapsed
time since `origin` (default: the first timestamp), in the machine's time unit, which must then be a time unit.
Columns that are not contract channels are never yielded; `dropped_columns` lists them. A cell that does not
parse as its channel's dtype is yielded as the raw string, so `check_observation` reports it (wrong_dtype).
Rows are read one at a time, so a file and a future stream go through the same consumer code.
"""

import csv
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.data.source import ChannelValue, Observation
from kaful.units import UnitError, to_seconds

TRUE, FALSE = "true", "false"


class FileSourceError(ValueError):
    """A file a FileSource cannot read as observations; the message names the file and row."""


def format_value(value: ChannelValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return TRUE if value else FALSE
    if isinstance(value, float):
        return repr(value)
    return str(value)


def parse_value(channel: SensorChannel, cell: str) -> ChannelValue:
    """The cell as the channel's dtype; None when empty; the raw string when it does not parse."""
    if cell == "":
        return None
    try:
        if channel.dtype == "bool":
            if cell.lower() not in (TRUE, FALSE):
                return cell
            return cell.lower() == TRUE
        if channel.dtype == "int":
            return int(cell)
        if channel.dtype == "float":
            return float(cell)
    except ValueError:
        return cell
    return cell


def columns(contract: SensorContract) -> tuple[str, ...]:
    return (contract.time_column, *sorted(c.name for c in contract.channels))


def write_observations(path: Path, contract: SensorContract, observations: Iterable[Observation]) -> int:
    """Write `observations` in the column convention; returns the row count. Unknown channels are refused."""
    known = {c.name for c in contract.channels}
    header = columns(contract)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for obs in observations:
            unknown = sorted(set(obs.values) - known)
            if unknown:
                raise ValueError(f"t={obs.t}: channels not in the contract: {unknown}")
            stamp = obs.timestamp.isoformat() if obs.timestamp is not None else ""
            writer.writerow([stamp, *(format_value(obs.values.get(name)) for name in header[1:])])
            n += 1
    return n


class FileSource:
    """A DataSource over one CSV file in the column convention."""

    def __init__(self, path: Path, contract: SensorContract, time_unit: str,
                 origin: datetime | None = None) -> None:
        self._path, self._contract, self._origin = path, contract, origin
        self._closed = False
        self.dropped_columns: tuple[str, ...] = ()
        self._unit_seconds: float | None = None
        if contract.usage_clock is None:
            try:
                self._unit_seconds = to_seconds(1.0, time_unit)
            except UnitError as exc:
                raise FileSourceError(f"{path}: no usage clock, and time unit {time_unit!r} is not a time: "
                                      f"t cannot be derived from timestamps") from exc

    @property
    def contract(self) -> SensorContract:
        return self._contract

    def close(self) -> None:
        self._closed = True

    def __iter__(self) -> Iterator[Observation]:
        if self._closed:
            raise FileSourceError(f"{self._path}: source is closed")
        by_name = {c.name: c for c in self._contract.channels}
        with self._path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if header is None or self._contract.time_column not in header:
                raise FileSourceError(f"{self._path}: no {self._contract.time_column!r} column")
            self.dropped_columns = tuple(sorted(h for h in header if h not in by_name
                                                and h != self._contract.time_column))
            last, origin = float("-inf"), self._origin
            for row_no, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise FileSourceError(f"{self._path} row {row_no}: {len(row)} cells, header has {len(header)}")
                cells = dict(zip(header, row))
                stamp = _timestamp(cells[self._contract.time_column], self._path, row_no)
                values = {name: parse_value(by_name[name], cell) for name, cell in sorted(cells.items())
                          if name in by_name}
                if origin is None and stamp is not None:
                    origin = stamp
                t = self._t(values, stamp, origin, row_no)
                if t < last:
                    raise FileSourceError(f"{self._path} row {row_no}: t={t} goes back from {last}")
                last = t
                yield Observation(t=t, timestamp=stamp, values=values)

    def _t(self, values: dict[str, ChannelValue], stamp: datetime | None, origin: datetime | None,
           row_no: int) -> float:
        clock = self._contract.usage_clock
        if clock is not None:
            value = values.get(clock)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FileSourceError(f"{self._path} row {row_no}: usage clock {clock!r} is {value!r}")
            return float(value)
        if stamp is None or origin is None:
            raise FileSourceError(f"{self._path} row {row_no}: no timestamp, and the contract has no usage clock")
        assert self._unit_seconds is not None
        return (stamp - origin).total_seconds() / self._unit_seconds


def _timestamp(cell: str, path: Path, row_no: int) -> datetime | None:
    if cell == "":
        return None
    try:
        return datetime.fromisoformat(cell)
    except ValueError as exc:
        raise FileSourceError(f"{path} row {row_no}: timestamp {cell!r} is not ISO 8601") from exc
