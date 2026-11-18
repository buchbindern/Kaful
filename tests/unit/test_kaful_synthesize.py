"""Synthesize (M12) on the small machine of the scope-to-build tests: the shaft seal leaks onto `leak_flow` (linear),
is replaced when it reaches its limit, and each replacement raises `seal_alarm` (threshold_event)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.synthetic import SyntheticDataset, SyntheticTruth
from kaful.artifacts.sensor_contract import SensorChannel, SensorContract
from kaful.config import DeclaredMaintenance, DeclaredObservation, KafulSettings, SynthesizeSettings
from kaful.data.file_source import FileSource
from kaful.data.source import check_observation
from kaful.paths import resolve_in, to_rel
from kaful.phases.p7_synthesize import phase as synth
from kaful.phases.p7_synthesize.a_draw_units import draw_unit, read_params
from kaful.phases.p7_synthesize.phase import FLAG_NO_CLOCK, synthesize
from kaful.runs import dump_json, sha256_bytes, write_text_atomic
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_observation_models import _alarm, _leak, _run
from tests.unit.test_kaful_scope import _channel, _contract, _machine

ALARM = DeclaredObservation(component="shaft_seal", channel="seal_alarm", form="threshold_event",
                            reason="the controller alarms when the seal reaches its leak limit")
RENEW = DeclaredMaintenance(component="shaft_seal", action="replace", on_threshold=True)


def _clock() -> SensorChannel:
    return _channel("hours", "Operating hours").model_copy(update={"role": "usage_clock", "unit": "h"})


def _contract_for(clock: bool) -> Any:
    if not clock:
        return _contract(_alarm())
    plain = _contract(_alarm())
    return SensorContract(**BASE, usage_clock="hours",
                          channels=tuple(sorted((*plain.channels, _clock()), key=lambda c: c.name)))


def _twin(clock: bool = True) -> tuple[Any, Any, Any]:
    leak = _leak()
    machine = _machine(maintenance=(RENEW,), observation=(*leak["observation"], ALARM),
                       observation_channel=leak["observation_channel"])
    contract = _contract_for(clock)
    twin, ps, p_fail, b_fail = _run(machine, contract)
    assert p_fail == () and b_fail == ()
    return twin, ps, contract


def _settings(**synth_kw: Any) -> KafulSettings:
    return KafulSettings(synthesize=SynthesizeSettings(**{"n_units": 4, "window_fraction": (0.5, 1.0), **synth_kw}))


def _synthesize(tmp_path: Path, settings: KafulSettings | None = None, clock: bool = True) -> tuple[Any, Any, Any]:
    twin, ps, contract = _twin(clock)
    phase_dir = tmp_path / "outputs" / "runs" / BASE["run_id"] / "synthesize"

    def save(truth: SyntheticTruth, name: str) -> ArtifactRef:
        text = dump_json(truth)
        write_text_atomic(phase_dir / name, text)
        return ArtifactRef(kind="synthetic_truth", run_id=BASE["run_id"], path=to_rel(tmp_path, phase_dir / name),
                           sha256=sha256_bytes(text.encode()))

    done = synthesize({**BASE}, twin, ps, contract, settings or _settings(), tmp_path, phase_dir, save)
    return done, twin, contract


def _truth(tmp_path: Path, unit: Any) -> SyntheticTruth:
    return SyntheticTruth.model_validate_json(resolve_in(tmp_path, unit.truth.path).read_text())


def test_every_unit_has_matching_files_and_its_observations_load_through_file_source(tmp_path: Path) -> None:
    done, twin, contract = _synthesize(tmp_path)
    assert done.failures == () and done.dataset is not None
    dataset = SyntheticDataset.model_validate_json(dump_json(done.dataset))
    assert [u.unit_id for u in dataset.units] == ["unit_00", "unit_01", "unit_02", "unit_03"]
    assert dataset.channels == ("hours", "leak_flow", "seal_alarm")
    for unit in dataset.units:
        for f in (unit.observations, unit.truth, unit.truth_states):
            assert sha256_bytes(resolve_in(tmp_path, f.path).read_bytes()) == f.sha256
        source = FileSource(resolve_in(tmp_path, unit.observations.path), contract, "h")
        rows = list(source)
        assert len(rows) == unit.n_rows and source.dropped_columns == ()
        assert all(check_observation(contract, o) == () for o in rows)
        assert rows[-1].t <= unit.t_end + 1e-9 and rows[0].t == 0.0
        assert _truth(tmp_path, unit).disclaimer == dataset.disclaimer


def test_the_observation_files_hold_no_truth(tmp_path: Path) -> None:
    done, twin, _ = _synthesize(tmp_path)
    header = resolve_in(tmp_path, done.dataset.units[0].observations.path).read_text().splitlines()[0]
    assert header == "timestamp,hours,leak_flow,seal_alarm"
    assert not any(s in header for s in ("damage", "renewal", twin.bindings[0].damage_state))


def test_the_alarm_reports_exactly_the_renewals_inside_the_window(tmp_path: Path) -> None:
    done, _, contract = _synthesize(tmp_path, _settings(n_units=8))
    seen = 0
    for unit in done.dataset.units:
        truth = {e.event: e for e in _truth(tmp_path, unit).events}["shaft_seal_failure"]
        alarms = [o.t for o in FileSource(resolve_in(tmp_path, unit.observations.path), contract, "h")
                  if o.values.get("seal_alarm") is True]
        assert alarms == [t for t in truth.limit_times if t <= unit.t_end]
        assert truth.in_window == bool(alarms)
        seen += len(alarms)
    assert seen > 0  # some unit renews its seal inside its window


def test_a_part_replaced_on_schedule_that_crosses_again_shows_every_crossing(tmp_path: Path) -> None:
    """The filter element is replaced every 4000 h (documented). Made to wear 4x faster, it reaches its limit, is
    replaced, and reaches it again: truth keeps every crossing, and the alarm reports each one in the window."""
    twin, ps, contract = _twin()
    alarm = twin.observations[0].model_copy(update={
        "id": "filter_element_fe_on_seal_alarm", "binding": "filter_element_fe", "channel": "seal_alarm",
        "form": "threshold_event", "params": {}, "event_value": None})
    twin = twin.model_copy(update={"observations": (alarm,)})
    k = ps.get("filter_element_fe.k")
    fast = k.model_copy(update={"prior": k.prior.model_copy(update={"median": 4 * k.prior.median})})
    ps = ps.model_copy(update={"parameters": tuple(fast if x is k else x for x in ps.parameters)})
    phase_dir = tmp_path / "synthesize"

    def save(truth: SyntheticTruth, name: str) -> ArtifactRef:
        write_text_atomic(phase_dir / name, dump_json(truth))
        return ArtifactRef(kind="synthetic_truth", run_id=BASE["run_id"], path=to_rel(tmp_path, phase_dir / name),
                           sha256=sha256_bytes(dump_json(truth).encode()))

    done = synthesize({**BASE}, twin, ps, contract, _settings(n_units=6, window_fraction=(1.0, 1.0)), tmp_path,
                      phase_dir, save)
    assert done.failures == ()
    repeated = 0
    assert done.dataset is not None
    for unit in done.dataset.units:
        filt = {e.event: e for e in _truth(tmp_path, unit).events}["filter_element_fe_failure"]
        alarms = [o.t for o in FileSource(resolve_in(tmp_path, unit.observations.path), contract, "h")
                  if o.values.get("seal_alarm") is True]
        assert alarms == [t for t in filt.limit_times if t <= unit.t_end]
        repeated += len(filt.limit_times) > 1
    assert repeated > 0


def test_the_leak_reading_follows_the_true_damage_with_the_true_noise(tmp_path: Path) -> None:
    done, twin, contract = _synthesize(tmp_path, _settings(n_units=1, samples_per_shortest_life=400,
                                                           window_fraction=(1.0, 1.0)))
    unit = done.dataset.units[0]
    truth = _truth(tmp_path, unit)
    p = truth.true_params
    states = np.genfromtxt(resolve_in(tmp_path, unit.truth_states.path), delimiter=",", names=True)
    b = next(b for b in twin.bindings if b.component == "shaft_seal")
    f = np.clip(states[b.damage_state] / p[b.threshold_param], 0, 1)
    expected = p["machine.leak_flow_y_healthy"] + (p["shaft_seal.leak_flow_y_limit"] - p["machine.leak_flow_y_healthy"]) * f
    y = np.array([o.values["leak_flow"] for o in FileSource(resolve_in(tmp_path, unit.observations.path), contract, "h")
                  if o.values.get("leak_flow") is not None])
    resid, sd = y - expected, p["machine.leak_flow_noise_sd"]
    assert resid.size == states.size
    assert abs(resid.mean()) < 4 * sd / np.sqrt(resid.size) and resid.std() == pytest.approx(sd, rel=0.25)


def test_true_parameters_are_the_ones_the_twin_reads_and_life_is_derived_from_k(tmp_path: Path) -> None:
    twin, ps, _ = _twin()
    draw = draw_unit(ps, twin, _settings().synthesize, 0)
    assert tuple(draw.true_params) == read_params(twin)
    assert "shaft_seal.life_nominal" not in draw.true_params
    assert draw.derived_params["shaft_seal.life_nominal"] == pytest.approx(
        draw.true_params["shaft_seal.damage_limit"] / draw.true_params["shaft_seal.k"])
    assert draw.true_params["machine.leak_flow_noise_sd"] > 0


def test_a_unit_does_not_change_when_more_units_are_drawn() -> None:
    twin, ps, _ = _twin()
    few, more = (draw_unit(ps, twin, _settings(n_units=n).synthesize, 1) for n in (2, 50))
    assert few.true_params == more.true_params and few.window_fraction == more.window_fraction
    assert few.unit_id == "unit_01" and more.unit_id == "unit_01"


def test_truth_first_failures_are_the_simulated_event_times(tmp_path: Path) -> None:
    done, twin, _ = _synthesize(tmp_path)
    for unit in done.dataset.units:
        truth = _truth(tmp_path, unit)
        for e in truth.events:
            assert e.in_window == (e.first_time is not None and e.first_time <= unit.t_end)


def test_without_a_usage_clock_rows_carry_timestamps_from_the_origin(tmp_path: Path) -> None:
    done, _, contract = _synthesize(tmp_path, clock=False)
    assert FLAG_NO_CLOCK in done.dataset.flags and done.dataset.timestamp_origin is not None
    unit = done.dataset.units[0]
    rows = list(FileSource(resolve_in(tmp_path, unit.observations.path), contract, "h",
                           origin=done.dataset.timestamp_origin))
    assert rows[0].t == 0.0 and rows[1].t == pytest.approx(done.dataset.sample_interval)


def test_a_unit_failing_the_dynamic_checks_fails_the_phase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synth, "check_unit", lambda unit, *a, **k: [f"{unit} shaft_seal monotone: D fell"])
    done, _, _ = _synthesize(tmp_path, _settings(n_units=1))
    assert done.failures == ("unit_00 shaft_seal monotone: D fell",)


def test_a_missing_profile_fails_cleanly(tmp_path: Path) -> None:
    done, _, _ = _synthesize(tmp_path, _settings(profile="ghost"))
    assert done.dataset is None and "no stored profile 'ghost'" in done.failures[0]


SCRIPT = """
import sys, hashlib
from pathlib import Path
from tests.unit.test_kaful_synthesize import _synthesize, _settings
from kaful.runs import dump_json
root = Path(sys.argv[1])
done, _, _ = _synthesize(root, _settings(n_units=2))
print("dataset", hashlib.sha256(dump_json(done.dataset).encode()).hexdigest())
for f in sorted(p for p in root.rglob("*") if p.is_file()):
    print(f.relative_to(root), hashlib.sha256(f.read_bytes()).hexdigest())
"""


def test_the_files_are_identical_across_hash_seeds(tmp_path: Path) -> None:
    out = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run([sys.executable, "-B", "-c", SCRIPT, str(tmp_path / seed)], env=env, check=True,
                              capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
        out.append(done.stdout)
    assert out[0] == out[1] and out[0].count("\n") == 7  # the dataset, then 2 units x 3 files
