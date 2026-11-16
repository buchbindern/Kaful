"""The estimate phase (M13) on the small machine of the synthesize tests: forecasts per unit and estimator with their
sidecars, unobserved groups shared from open-loop, outputs identical for any worker count and hash seed, and truth
never read."""

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kaful.artifacts.base import ArtifactRef, ArtifactBase
from kaful.artifacts.forecast import Forecast
from kaful.config import EstimateSettings, KafulSettings, SynthesizeSettings
from kaful.paths import resolve_in, to_rel
from kaful.phases.p8_estimate.phase import estimate
from kaful.runs import dump_json, sha256_bytes, write_text_atomic
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_synthesize import _synthesize

ESTIMATE = EstimateSettings(n_particles=200, forecast_samples=50, issue_fractions=(0.5, 1.0))


def _settings(workers: int = 1) -> KafulSettings:
    return KafulSettings(synthesize=SynthesizeSettings(n_units=3, window_fraction=(0.5, 1.0)),
                         estimate=ESTIMATE.model_copy(update={"workers": workers}))


def _estimate(tmp_path: Path, workers: int = 1, damage_truth: str | None = None) -> tuple[Any, Path]:
    done, twin, contract = _synthesize(tmp_path, _settings())
    dataset = done.dataset
    for unit in dataset.units:
        for f in (unit.truth, unit.truth_states):
            target = resolve_in(tmp_path, f.path)
            if damage_truth == "delete":
                target.unlink()
            elif damage_truth == "corrupt":
                target.write_text("not the truth")
    ps = done_ps(tmp_path)
    phase_dir = tmp_path / "outputs" / "runs" / BASE["run_id"] / "estimate"

    def save(artifact: ArtifactBase, name: str) -> ArtifactRef:
        text = dump_json(artifact)
        write_text_atomic(phase_dir / name, text)
        return ArtifactRef(kind=str(getattr(artifact, "kind")), run_id=BASE["run_id"],
                           path=to_rel(tmp_path, phase_dir / name), sha256=sha256_bytes(text.encode()))

    out = estimate({**BASE}, twin, ps, contract, dataset, _settings(workers), tmp_path, phase_dir, save)
    return out, phase_dir


def done_ps(tmp_path: Path) -> Any:
    from tests.unit.test_kaful_synthesize import _twin

    return _twin()[1]


def _digest(phase_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(phase_dir.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(phase_dir)).encode() + p.read_bytes())
    return h.hexdigest()


def test_every_unit_gets_forecasts_whose_sidecars_match_and_the_unobserved_group_is_shared(tmp_path: Path) -> None:
    out, phase_dir = _estimate(tmp_path)
    fs = out.forecast_set
    assert out.failures == ()
    assert {(e.source, e.estimator) for e in fs.forecasts} == {(f"unit_{i:02d}", m) for i in range(3)
                                                               for m in ("open_loop", "particle_filter")}
    assert {(s.estimator, "renewed at its limit" in s.reason) for s in fs.skipped} == {("ukf", True)}
    assert any("estimator assumption" in f for f in fs.flags)
    for entry in fs.forecasts:
        text = resolve_in(tmp_path, entry.path).read_text()
        assert sha256_bytes(text.encode()) == entry.sha256
        f = Forecast.model_validate_json(text)
        assert sha256_bytes(resolve_in(tmp_path, f.trace).read_bytes()) == f.trace_sha256
        for issue in f.issues:
            assert sha256_bytes(resolve_in(tmp_path, issue.samples).read_bytes()) == issue.samples_sha256
            for e in issue.events:
                assert e.n_samples == 50 and e.n_hits + e.n_censored == 50
    shared = [r for e in fs.forecasts if e.estimator == "particle_filter"
              for r in Forecast.model_validate_json(resolve_in(tmp_path, e.path).read_text()).groups
              if r.shared_from_open_loop]
    assert shared and all(r.estimator == "open_loop" for r in shared)
    unit = "unit_00"
    rows = {m: list(csv.DictReader(open(phase_dir / "samples" / f"{unit}.{m}.0.csv")))
            for m in ("open_loop", "particle_filter")}
    col = "filter_element_fe_failure.toe"
    assert [r[col] for r in rows["open_loop"]] == [r[col] for r in rows["particle_filter"]]


def test_outputs_do_not_depend_on_the_number_of_workers(tmp_path: Path) -> None:
    _, one = _estimate(tmp_path / "one", workers=1)
    _, two = _estimate(tmp_path / "two", workers=2)
    assert _digest(one) == _digest(two)


def _digest_in_subprocess(tmp: Path, seed: str) -> str:
    code = ("import sys; from pathlib import Path; from tests.unit.test_kaful_estimate_phase import _estimate, _digest; "
            "print(_digest(_estimate(Path(sys.argv[1]))[1]))")
    out = subprocess.run([sys.executable, "-B", "-W", "ignore", "-c", code, str(tmp)], capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONHASHSEED": seed})
    return out.stdout.strip()


def test_outputs_do_not_depend_on_hash_order(tmp_path: Path) -> None:
    assert _digest_in_subprocess(tmp_path / "a", "0") == _digest_in_subprocess(tmp_path / "b", "1")


@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_the_estimator_never_reads_the_truth(tmp_path: Path, damage: str) -> None:
    _, clean = _estimate(tmp_path / "clean")
    _, damaged = _estimate(tmp_path / "damaged", damage_truth=damage)
    assert _digest(clean) == _digest(damaged)
