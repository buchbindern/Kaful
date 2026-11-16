"""FileSource and write_observations (M12): one column convention, round trips, and the errors named by row."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.data.file_source import FileSource, FileSourceError, columns, write_observations
from kaful.data.source import DataSource, Observation, check_observation

BASE = {"run_id": "r", "machine_id": "tiny_pump"}
DEFAULT = {"source": "default", "ref": "test", "confidence": 0.5}


def _channel(name: str, role: str, dtype: str, **kw: object) -> SensorChannel:
    sampling = "event_driven" if role == "event" else "unknown"
    return SensorChannel(name=name, description=name, role=role, dtype=dtype, sampling=sampling,  # type: ignore[arg-type]
                         **DEFAULT, **kw)  # type: ignore[arg-type]


def contract(clock: bool = True) -> SensorContract:
    channels = [_channel("flow", "observation", "float", unit="L/min", range=(0.0, 50.0)),
                _channel("alarm", "event", "bool"),
                _channel("mode", "event", "category", categories=("auto", "manual")),
                _channel("starts", "operating_input", "int", unit="count")]
    if clock:
        channels.append(_channel("hours", "usage_clock", "float", unit="h"))
    return SensorContract(**BASE, usage_clock="hours" if clock else None,
                          channels=tuple(sorted(channels, key=lambda c: c.name)))


def _obs(t: float, **values: object) -> Observation:
    return Observation(t=t, values={"hours": t, **values})  # type: ignore[dict-item]


def test_round_trip_gives_back_every_value(tmp_path: Path) -> None:
    rows = [_obs(0.0, flow=12.5, starts=1), _obs(0.5, flow=0.1 + 0.2, alarm=True),
            _obs(1.0, mode="manual", flow=None), _obs(1.0, alarm=False, starts=3)]
    path = tmp_path / "unit.csv"
    assert write_observations(path, contract(), rows) == 4
    source = FileSource(path, contract(), "h")
    assert isinstance(source, DataSource)
    got = list(source)
    assert [o.t for o in got] == [0.0, 0.5, 1.0, 1.0]
    for want, have in zip(rows, got):
        assert {k: v for k, v in have.values.items() if v is not None} == \
               {k: v for k, v in want.values.items() if v is not None}
        assert check_observation(contract(), have) == ()
    assert got[1].values["flow"] == 0.1 + 0.2  # shortest round-trip float form
    assert type(got[0].values["starts"]) is int and got[1].values["alarm"] is True


def test_writing_is_byte_identical(tmp_path: Path) -> None:
    rows = [_obs(float(i), flow=i / 3, alarm=i % 2 == 0) for i in range(5)]
    write_observations(tmp_path / "a.csv", contract(), rows)
    write_observations(tmp_path / "b.csv", contract(), rows)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert (tmp_path / "a.csv").read_text().splitlines()[0].split(",") == list(columns(contract()))


def test_writer_refuses_a_channel_outside_the_contract(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not in the contract"):
        write_observations(tmp_path / "x.csv", contract(), [_obs(0.0, truth_damage=0.1)])


def test_unknown_columns_are_dropped_and_listed(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("timestamp,hours,flow,secret_damage\n,0.0,1.5,0.3\n")
    source = FileSource(path, contract(), "h")
    (obs,) = list(source)
    assert "secret_damage" not in obs.values and source.dropped_columns == ("secret_damage",)


def test_a_cell_that_does_not_parse_is_reported_by_the_contract_check(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("timestamp,hours,flow,alarm\n,0.0,high,maybe\n")
    (obs,) = list(FileSource(path, contract(), "h"))
    problems = {(v.channel, v.problem) for v in check_observation(contract(), obs)}
    assert problems == {("alarm", "wrong_dtype"), ("flow", "wrong_dtype")}


def test_out_of_range_values_are_yielded_and_reported(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    write_observations(path, contract(), [_obs(0.0, flow=80.0)])
    (obs,) = list(FileSource(path, contract(), "h"))
    assert obs.values["flow"] == 80.0
    assert [v.problem for v in check_observation(contract(), obs)] == ["out_of_range"]


def test_t_going_back_is_an_error_naming_the_row(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    write_observations(path, contract(), [_obs(2.0), _obs(1.0)])
    with pytest.raises(FileSourceError, match="row 3"):
        list(FileSource(path, contract(), "h"))


def test_a_missing_usage_clock_value_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("timestamp,hours,flow\n,,1.0\n")
    with pytest.raises(FileSourceError, match="usage clock"):
        list(FileSource(path, contract(), "h"))


def test_without_a_clock_t_comes_from_timestamps_in_the_time_unit(tmp_path: Path) -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [Observation(t=t, timestamp=start + timedelta(hours=t), values={"flow": 1.0}) for t in (0.0, 1.5, 4.0)]
    path = tmp_path / "x.csv"
    write_observations(path, contract(clock=False), rows)
    assert [o.t for o in FileSource(path, contract(clock=False), "hour")] == [0.0, 1.5, 4.0]
    later = FileSource(path, contract(clock=False), "minute", origin=start - timedelta(minutes=30))
    assert [o.t for o in later] == [30.0, 120.0, 270.0]


def test_without_a_clock_the_time_unit_must_be_a_time(tmp_path: Path) -> None:
    with pytest.raises(FileSourceError, match="not a time"):
        FileSource(tmp_path / "x.csv", contract(clock=False), "count")


def test_without_a_clock_a_row_needs_a_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("timestamp,flow\n,1.0\n")
    with pytest.raises(FileSourceError, match="no timestamp"):
        list(FileSource(path, contract(clock=False), "hour"))


def test_a_closed_source_does_not_iterate(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    write_observations(path, contract(), [_obs(0.0)])
    source = FileSource(path, contract(), "h")
    source.close()
    source.close()
    with pytest.raises(FileSourceError, match="closed"):
        list(source)


def test_a_file_without_the_time_column_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("hours,flow\n0.0,1.0\n")
    with pytest.raises(FileSourceError, match="timestamp"):
        list(FileSource(path, contract(), "h"))
