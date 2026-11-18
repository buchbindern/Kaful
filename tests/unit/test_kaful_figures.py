"""Figures (M16, checkpoint 1): drawn from a real saved lineage of the small test machine.

Every plotted number equals its source: fan percentiles are recomputed from the sample CSVs, true RULs are
calibration's own, tracking rows are the trace and observation files, coverage rows are the evaluation report's
fields. Censored intervals stay open; an already-failed issue has true RUL 0, never a truth segment; the unit is
chosen by the stated rule; the manifest names every input by relative path and sha256; outputs are byte-identical
under two hash seeds.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kaful.artifacts.evaluation import EvaluationReport
from kaful.artifacts.forecast import Forecast, ForecastSet
from kaful.config import REPO_ROOT, KafulSettings
from kaful.data.file_source import FileSource
from kaful.figures import f3_rul_fan
from kaful.figures.build import MANIFEST, FiguresManifest, build
from kaful.figures.forecasts import FAN_QUANTILES, rul_quantiles
from kaful.figures.select import choose_unit
from kaful.figures.spec import FiguresSpec, load_spec
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.f_calibration import Collected, Record, collect_forecast
from kaful.phases.p9_evaluate.helpers.parts import part_classes
from kaful.runs import read_manifest, sha256_file
from kaful.scoring.samples import read_table, toe_samples
from kaful.scoring.truth import load_units
from tests.harness.figures_lineage import EVALUATE_RUN, MACHINE, build_lineage

SPEC = f"""
[style]
single_width_in = 3.5
double_width_in = 7.0
panel_height_in = 2.0
dpi = 60
font_size = 8.0
font_family = "DejaVu Sans"
svg_hashsalt = "test"
band_alpha = 0.2
synthetic_label = "synthetic data"

[style.colors]
truth = "#000000"
particle_filter = "#0072B2"
open_loop = "#7F7F7F"
ukf = "#E69F00"
target = "#009E73"
observation = "#CC79A7"
cone = "#E6E6E6"

[labels]
shaft_seal_failure = "shaft seal"

[lineages.tiny]
machine = "{MACHINE}"
evaluate_run = "{EVALUATE_RUN}"
label = "Tiny"

[[figures]]
id = "F2"
slug = "tracking"
kind = "tracking"
lineage = "tiny"
quantity = "shaft_seal_failure"
channel = "leak_flow"
bindings = ["shaft_seal"]
estimators = ["particle_filter", "open_loop"]
caption = "tracking"

[[figures]]
id = "F3"
slug = "fan"
kind = "rul_fan"
lineage = "tiny"
quantity = "shaft_seal_failure"
estimators = ["particle_filter", "open_loop"]
caption = "fan"

[[figures]]
id = "F4"
slug = "calibration"
kind = "calibration"
lineages = ["tiny"]
estimators = ["particle_filter", "open_loop"]
caption = "calibration"
"""


def make(root: Path) -> tuple[Path, KafulSettings]:
    """The fixture lineage under root/machines and the spec under root/figures; returns (spec path, settings)."""
    settings = build_lineage(root / "machines")
    spec = root / "figures" / "figures.toml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(SPEC, encoding="utf-8")
    return spec, settings


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, KafulSettings, FiguresManifest]:
    root = tmp_path_factory.mktemp("figs")
    spec, settings = make(root)
    return root, settings, build(spec, settings, root=root)


def _csv(root: Path, name: str) -> list[dict[str, str]]:
    with (root / "figures" / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(cell: str) -> float:
    return float(cell) if cell != "" else np.inf


def _machine(root: Path) -> Path:
    return root / "machines" / MACHINE


def _forecasts(root: Path, estimator: str) -> list[Forecast]:
    md = _machine(root)
    run = read_manifest(md / "outputs" / "runs" / EVALUATE_RUN)
    ref = next(a for p in run.phases if p.phase == "estimate" for a in p.artifacts)
    fs = ForecastSet.model_validate_json(resolve_in(md, ref.path).read_text())
    return [Forecast.model_validate_json(resolve_in(md, e.path).read_text()) for e in fs.forecasts
            if e.estimator == estimator]


def test_build_writes_svg_png_csv_and_a_manifest_naming_every_input(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, manifest = built
    saved = FiguresManifest.model_validate_json((root / "figures" / MANIFEST).read_text())
    assert saved.figures == manifest.figures and [r.id for r in saved.figures] == ["F2", "F3", "F4"]
    for r in manifest.figures:
        assert len(r.outputs) == 3 and all(n.startswith(f"{r.id}_") for n in r.outputs)
        for name, sha in r.outputs.items():
            assert sha256_file(root / "figures" / name) == sha
        for i in r.inputs:
            assert not Path(i.path).is_absolute() and sha256_file(root / i.path) == i.sha256
    fan = next(r for r in manifest.figures if r.id == "F3")
    paths = [i.path for i in fan.inputs]
    assert any(p.endswith("/manifest.json") for p in paths) and any("evaluation_report" in p for p in paths)
    assert any("/truth/" in p for p in paths) and any("/samples/" in p for p in paths)
    assert fan.selection is not None and "rule median_abs_log_error" in fan.selection
    assert manifest.spec.path == "figures/figures.toml"


def test_fan_percentiles_are_the_sample_csvs_and_true_ruls_are_calibrations(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, settings, _ = built
    rows = _csv(root, "F3_fan.csv")
    unit = rows[0]["unit"]
    md = _machine(root)
    for est in ("particle_filter", "open_loop"):
        f = next(x for x in _forecasts(root, est) if x.source == unit)
        mine = [r for r in rows if r["estimator"] == est]
        assert len(mine) == len(f.issues)
        for r, issue in zip(mine, f.issues):
            rul = toe_samples(read_table(resolve_in(md, issue.samples)), "shaft_seal_failure") - issue.t_issue
            want = [float(np.quantile(rul, q, method="inverted_cdf")) for q in FAN_QUANTILES]
            got = [_f(r[f"rul_p{round(q * 100):02d}"]) for q in FAN_QUANTILES]
            assert got == want
            ev = next(e for e in issue.events if e.event == "shaft_seal_failure")
            assert float(r["p_event_before_horizon"]) == pytest.approx(ev.p_event_before_horizon)
    records = _calibration_records(root, settings)
    truth = {(r.unit, r.issue): r.truth for r in records[("particle_filter", "shaft_seal_failure")]}
    for r in rows:
        want_toe = truth[(unit, int(r["issue"]))]
        assert (_f(r["true_time_of_event"]) if r["true_time_of_event"] else None) == want_toe


def _calibration_records(root: Path, settings: KafulSettings) -> dict[tuple[str, str], list[Record]]:
    from kaful.artifacts.synthetic import SyntheticDataset
    from kaful.artifacts.twin_definition import TwinDefinition

    md = _machine(root)
    run = read_manifest(md / "outputs" / "runs" / EVALUATE_RUN)
    refs = {(p.phase, a.kind): a for p in run.phases for a in p.artifacts}
    twin = TwinDefinition.model_validate_json(resolve_in(md, refs[("verify", "twin_definition")].path).read_text())
    ds = SyntheticDataset.model_validate_json(
        resolve_in(md, refs[("synthesize", "synthetic_dataset")].path).read_text())
    units = load_units(md, ds)
    c = Collected(part=part_classes(twin))
    for est in ("particle_filter", "open_loop"):
        for f in _forecasts(root, est):
            collect_forecast(f, units[f.source], twin, md, settings.evaluate, c)
    return dict(c.records)


def test_coverage_rows_are_the_evaluation_reports_fields(built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    md = _machine(root)
    report = EvaluationReport.model_validate_json(
        (md / f"outputs/runs/{EVALUATE_RUN}/evaluate/evaluation_report.json").read_text())
    assert report.calibration is not None
    cells = {(c.quantity, c.estimator): c for c in report.calibration.cells if c.issue_fraction is None}
    rows = _csv(root, "F4_calibration.csv")
    assert rows and {(r["quantity"], r["estimator"]) for r in rows} <= set(cells)
    for r in rows:
        c = cells[(r["quantity"], r["estimator"])]
        assert c.share is not None and c.ci_units is not None
        assert (float(r["coverage"]), float(r["ci_units_low"]), float(r["ci_units_high"])) == \
            (c.share, c.ci_units.low, c.ci_units.high)
        assert (int(r["units"]), int(r["pairs"]), r["part_class"]) == (c.units, c.pairs, c.part_class)
    judged = [r for r in rows if r["verdict"]]
    assert {r["quantity"] for r in judged} == {v.quantity for v in report.calibration.verdicts}
    assert all(r["estimator"] == report.calibration.primary_estimator for r in judged)


def test_tracking_rows_are_the_trace_and_observation_files(built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    rows = _csv(root, "F2_tracking.csv")
    unit = json.loads((root / "figures" / MANIFEST).read_text())["figures"][0]["selection"].split(":")[0]
    md = _machine(root)
    f = next(x for x in _forecasts(root, "particle_filter") if x.source == unit)
    trace = read_table(resolve_in(md, f.trace))
    health = [r for r in rows if r["series"] == "health" and r["estimator"] == "particle_filter"]
    assert [(float(r["t"]), float(r["value"]), float(r["low"]), float(r["high"])) for r in health] == \
        [(float(t["t"]), float(t["shaft_seal.health"]), float(t["shaft_seal.health_p05"]),
          float(t["shaft_seal.health_p95"])) for t in trace]
    expected = [r for r in rows if r["series"] == "expected" and r["estimator"] == "particle_filter"]
    assert [float(r["value"]) for r in expected] == [float(t["leak_flow.expected"]) for t in trace]
    from kaful.artifacts.sensor_contract import SensorContract
    from kaful.artifacts.synthetic import SyntheticDataset

    run = read_manifest(md / "outputs" / "runs" / EVALUATE_RUN)
    refs = {(p.phase, a.kind): a for p in run.phases for a in p.artifacts}
    ds = SyntheticDataset.model_validate_json(
        resolve_in(md, refs[("synthesize", "synthetic_dataset")].path).read_text())
    contract = SensorContract.model_validate_json(
        resolve_in(md, refs[("understand", "sensor_contract")].path).read_text())
    u = next(x for x in ds.units if x.unit_id == unit)
    obs = [(o.t, o.values["leak_flow"]) for o in FileSource(resolve_in(md, u.observations.path), contract,
                                                             ds.time_unit, ds.timestamp_origin)
           if o.values.get("leak_flow") is not None]
    assert [(float(r["t"]), float(r["value"])) for r in rows if r["series"] == "observation"] == obs
    truth = [r for r in rows if r["series"] == "true_health"]
    assert truth and all(0.0 <= float(r["value"]) <= 1.0 for r in truth) and float(truth[0]["value"]) == 1.0


def test_a_censored_quantile_is_open_never_the_horizon() -> None:
    rul = np.concatenate([np.linspace(10.0, 50.0, 80), np.full(20, np.inf)])
    q = rul_quantiles(rul)
    assert np.isfinite(q[2]) and q[-1] == np.inf
    data = f3_rul_fan.FanData(label="m", time_unit="h", quantity="x", quantity_label="x", unit="u", selection="s",
                              alpha=0.2, truths=((0, 0.0, 100.0),),
                              points=(f3_rul_fan.FanPoint("particle_filter", 0, 0.0, q, 0.8, 100),))
    row = f3_rul_fan.rows(data)[0]
    assert row["rul_p95"] == np.inf  # written as '' by the CSV writer, never a number
    from kaful.figures.style import csv_bytes

    text = csv_bytes(f3_rul_fan.COLUMNS, [row]).decode()
    assert text.splitlines()[1].split(",")[f3_rul_fan.COLUMNS.index("rul_p95")] == ""


def test_an_issue_after_a_failure_has_true_rul_0_and_no_truth_segment() -> None:
    data = f3_rul_fan.FanData(label="m", time_unit="h", quantity="x", quantity_label="x", unit="u", selection="s",
                              alpha=0.2, truths=((0, 10.0, 60.0), (1, 70.0, 70.0), (2, 90.0, None)), points=())
    assert f3_rul_fan._truth_segments(data) == [(0.0, 60.0)]
    assert f3_rul_fan._triggered(data) == [70.0]
    renewing = data.__class__(**{**data.__dict__, "truths": ((0, 10.0, 60.0), (1, 70.0, 130.0))})
    assert f3_rul_fan._truth_segments(renewing) == [(0.0, 60.0), (60.0, 130.0)]


def _rec(unit: str, issue: int, log_ratio: float | None) -> Record:
    from kaful.phases.p9_evaluate.helpers.metrics import PointError

    err = PointError(error=0.0, relative=0.0, log_ratio=log_ratio) if log_ratio is not None else None
    return Record(unit=unit, issue=issue, t=0.0, truth=1.0, cls="covered", alpha_mass=None, alpha_ok=None,
                  error=err, agree=None, width=None)


def test_the_unit_is_the_median_ranked_one_by_the_stated_rule() -> None:
    recs = [_rec("u1", 0, 0.1), _rec("u2", 0, -0.5), _rec("u3", 0, 0.3), _rec("u3", 1, 0.7), _rec("u4", 0, None),
            _rec("u5", 0, 0.9), _rec("u5", 1, None)]
    c = choose_unit(recs, None, "particle_filter", "q")  # scores u1 .1, u3 .5, u2 .5, u5 .9 -> lower median
    assert c.unit == "u2" and "rank 2 of 4" in c.note
    every = choose_unit(recs, None, "particle_filter", "q", "median_abs_log_error_all_issues")
    assert every.unit == "u2" and "rank 2 of 3" in every.note  # u4 and u5 lack a scored error at some issue
    named = choose_unit(recs, "u4", "particle_filter", "q")
    assert named.unit == "u4" and "named in the spec" in named.note
    with pytest.raises(ValueError, match="no unit"):
        choose_unit([_rec("u4", 0, None)], None, "particle_filter", "q")


def test_the_spec_rejects_unknown_lineages_duplicate_ids_and_missing_colours(tmp_path: Path) -> None:
    good = SPEC
    p = tmp_path / "s.toml"
    p.write_text(good)
    assert isinstance(load_spec(p), FiguresSpec)
    for bad, match in ((good.replace('lineage = "tiny"', 'lineage = "nope"', 1), "unknown lineages"),
                       (good.replace('id = "F3"', 'id = "F2"'), "duplicate"),
                       (good.replace('cone = "#E6E6E6"\n', ""), "colors")):
        p.write_text(bad)
        with pytest.raises(ValueError, match=match):
            load_spec(p)


def _shas_for(root: str) -> str:
    """Build the lineage and the figures under `root` (a subprocess per hash seed); the output files' sha256."""
    spec, settings = make(Path(root))
    manifest = build(spec, settings, root=Path(root))
    return json.dumps({r.id: r.outputs for r in manifest.figures}, sort_keys=True)


def test_figures_are_byte_identical_under_two_hash_seeds(tmp_path: Path) -> None:
    out = []
    for seed in ("0", "1"):
        code = f"from tests.unit.test_kaful_figures import _shas_for; print(_shas_for({str(tmp_path / seed)!r}))"
        done = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=REPO_ROOT,
                              env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
        out.append(done.stdout.strip().splitlines()[-1])
    assert out[0] == out[1] and '"F3_fan.svg"' in out[0]


def test_the_cli_parses_figures_build() -> None:
    from kaful.cli import build_parser

    args = build_parser().parse_args(["figures", "build", "--only", "F3", "F4"])
    assert (args.command, args.figures_command, args.only, args.spec) == ("figures", "build", ["F3", "F4"], None)
