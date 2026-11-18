"""Figures (M16, checkpoint 1): drawn from a real saved lineage of the small test machine.

Every plotted number equals its source: fan percentiles are recomputed from the sample CSVs, true RULs are
calibration's own, tracking rows are the trace and observation files, coverage rows are the evaluation report's
fields. Censored intervals stay open; an already-failed issue has true RUL 0, never a truth segment; the unit is
chosen by the stated rule; the manifest names every input by relative path and sha256; outputs are byte-identical
under two hash seeds.
"""

import csv
import html
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
from kaful.figures import f3_failure_time as f3
from kaful.figures.build import MANIFEST, FiguresManifest, build
from kaful.figures.drawn import Context
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

[style.source_colors]
standard = "#56B4E9"
calibrated = "#009E73"
deployment = "#F0E442"
llm_judgement = "#E69F00"
default = "#BBBBBB"

[style.colors]
truth = "#000000"
particle_filter = "#0072B2"
open_loop = "#7F7F7F"
ukf = "#E69F00"
target = "#009E73"
observation = "#CC79A7"
cone = "#E6E6E6"
legacy = "#BBBBBB"
cost_high = "#9CC6E4"
with_maintenance = "#56B4E9"
without_maintenance = "#D55E00"

[labels]
shaft_seal_failure = "shaft seal"

[report]
file = "report.html"
title = "Tiny report"
subtitle = "test"
lineages = ["tiny"]
scope_lineage = "tiny"
selector_figure = "F3"
quote_max_chars = 80

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
kind = "failure_time"
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

[[figures]]
id = "F4t"
slug = "talk"
kind = "calibration"
lineages = ["tiny"]
estimators = ["particle_filter", "open_loop"]
observed_only = true
x_min = 0.5
caption = "talk"

[[figures]]
id = "F1"
slug = "pipeline"
kind = "pipeline"
lineage = "tiny"
caption = "pipeline"

[[figures]]
id = "F6"
slug = "provenance"
kind = "provenance"
lineages = ["tiny"]
sources = ["standard", "calibrated", "deployment", "llm_judgement", "default"]
caption = "provenance"

[[figures]]
id = "F7"
slug = "sharpness"
kind = "sharpness"
lineages = ["tiny"]
estimators = ["particle_filter", "open_loop"]
caption = "sharpness"

[[figures]]
id = "F7t"
slug = "sharpness_talk"
kind = "sharpness"
lineages = ["tiny"]
estimators = ["particle_filter", "open_loop"]
widths_panel = false
caption = "talk"

[[figures]]
id = "F8"
slug = "maintenance"
kind = "maintenance"
parts = [{{lineage = "tiny", binding = "filter_element_fe", bands = ["documented_maintenance"]}}]
draws = 3
seed = 7
steps_per_shortest = 20
lives_shown = 2.0
caption = "maintenance"

[[figures]]
id = "F11"
slug = "grid"
kind = "failure_grid"
lineage = "tiny"
quantity = "shaft_seal_failure"
estimators = ["particle_filter", "open_loop"]
columns = 2
caption = "grid"
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
    assert saved.figures == manifest.figures
    assert [r.id for r in saved.figures] == ["F2", "F3", "F4", "F4t", "F1", "F6", "F7", "F7t", "F8", "F11"]
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


def test_failure_time_percentiles_are_the_sample_csvs_and_truths_are_calibrations(
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
            want = [issue.t_issue + float(np.quantile(rul, q, method="inverted_cdf")) for q in FAN_QUANTILES]
            got = [_f(r[f"toe_p{round(q * 100):02d}"]) for q in FAN_QUANTILES]
            assert got == want and float(r["horizon"]) == f.horizon
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


def _point(toe: tuple[float, ...]) -> f3.FailurePoint:
    return f3.FailurePoint("particle_filter", 0, 0.0, 200.0, toe, 0.8, 100)


def _data(truths: tuple[tuple[int, float, float | None], ...], points: tuple[f3.FailurePoint, ...] = ()
          ) -> f3.FailureData:
    return f3.FailureData(label="m", time_unit="h", quantity="x", quantity_label="x", unit="u", selection="s",
                          truths=truths, points=points)


def test_a_censored_quantile_is_open_never_the_horizon() -> None:
    rul = np.concatenate([np.linspace(10.0, 50.0, 80), np.full(20, np.inf)])
    q = rul_quantiles(rul)
    assert np.isfinite(q[2]) and q[-1] == np.inf
    row = f3.rows(_data(((0, 0.0, 100.0),), (_point(q),)))[0]
    assert row["toe_p95"] == np.inf  # written as '' by the CSV writer, never a number
    from kaful.figures.style import csv_bytes

    text = csv_bytes(f3.COLUMNS, [row]).decode()
    assert text.splitlines()[1].split(",")[f3.COLUMNS.index("toe_p95")] == ""


def test_the_censored_bar_ends_at_the_horizon_with_an_open_arrow() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    f3._bar(ax, 50.0, np.inf, 0.0, 200.0, "#000000", 1.0)
    (line,) = ax.get_lines()
    assert np.asarray(line.get_xdata()).tolist() == [50.0, 200.0] and len(ax.texts) == 1  # the arrow: annotation
    plt.close(fig)


def test_one_true_failure_line_only_when_every_issue_shares_it() -> None:
    assert f3.shared_truth(_data(((0, 10.0, 60.0), (1, 30.0, 60.0)))) == 60.0
    # after a failure with no renewal the next event is the issue time itself (true RUL 0): one mark per row
    assert f3.shared_truth(_data(((0, 10.0, 60.0), (1, 70.0, 70.0)))) is None
    assert f3.shared_truth(_data(((0, 10.0, 60.0), (1, 70.0, 130.0)))) is None


def test_the_talk_variant_keeps_observed_rows_only(built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    full, talk = _csv(root, "F4_calibration.csv"), _csv(root, "F4t_talk.csv")
    assert talk == [r for r in full if r["part_class"] == "observed"] and len(talk) < len(full)


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
    assert manifest.report is not None
    return json.dumps({"report": manifest.report.sha256, **{r.id: r.outputs for r in manifest.figures}},
                      sort_keys=True)


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


def test_a_library_changing_matplotlib_settings_never_changes_the_figures(
        built: tuple[Path, KafulSettings, FiguresManifest], monkeypatch: pytest.MonkeyPatch) -> None:
    """ProgPy's visualize module sets global rcParams on import (line width 3, font size 14, x-large labels), and the
    backend is imported lazily inside F8's draw: settings changed before or during a draw must not change a byte."""
    import matplotlib as mpl

    from kaful.figures import build as build_module

    root, settings, manifest = built
    before = {r.id: r.outputs for r in manifest.figures}
    pollution = {"lines.linewidth": 3, "font.size": 14, "axes.labelsize": "x-large", "legend.fontsize": "large",
                 "lines.markersize": 8}
    drawers = dict(build_module.DRAWERS)

    def polluting(draw):  # type: ignore[no-untyped-def]
        def wrapped(spec, ctx):  # type: ignore[no-untyped-def]
            mpl.rcParams.update(pollution)  # as a lazy `import progpy` inside the draw does
            return draw(spec, ctx)
        return wrapped

    monkeypatch.setattr(build_module, "DRAWERS", {k: polluting(v) for k, v in drawers.items()})
    saved = dict(mpl.rcParams)
    try:
        mpl.rcParams.update(pollution)
        again = build(root / "figures" / "figures.toml", settings, root=root)
    finally:
        mpl.rcParams.update(saved)
    assert {r.id: r.outputs for r in again.figures} == before


def _report(root: Path) -> EvaluationReport:
    return EvaluationReport.model_validate_json(
        (_machine(root) / f"outputs/runs/{EVALUATE_RUN}/evaluate/evaluation_report.json").read_text())


def _refs(root: Path) -> dict[tuple[str, str], str]:
    run = read_manifest(_machine(root) / "outputs" / "runs" / EVALUATE_RUN)
    return {(p.phase, a.kind): a.path for p in run.phases for a in p.artifacts}


def test_pipeline_boxes_count_the_artifacts_and_carry_each_phases_cost(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    from kaful.artifacts.document_index import DocumentIndex
    from kaful.artifacts.machine_model import MachineModel
    from kaful.artifacts.synthetic import SyntheticDataset

    root, _, _ = built
    md, refs, report = _machine(root), _refs(root), _report(root)
    rows = _csv(root, "F1_pipeline.csv")
    lines = {r["line"] for r in rows}
    index = DocumentIndex.model_validate_json(resolve_in(md, refs[("ingest", "document_index")]).read_text())
    model = MachineModel.model_validate_json(resolve_in(md, refs[("understand", "machine_model")]).read_text())
    ds = SyntheticDataset.model_validate_json(resolve_in(md, refs[("synthesize", "synthetic_dataset")]).read_text())
    assert f"{len(index.pages)} pages, {len(index.chunks)} chunks" in lines
    assert f"{len(model.components)} components" in lines and f"{len(ds.units)} synthetic units" in lines
    cost = {p.phase: (p.cold_low_usd, p.cold_usd) for p in report.cost.by_phase}
    for r in rows:
        if r["phase"] in cost:
            assert (float(r["cost_low_usd"]), float(r["cost_high_usd"])) == cost[r["phase"]]


def test_provenance_rows_are_the_reports_grounding_and_sensitivity(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    report = _report(root)
    assert report.grounding is not None and report.sensitivity is not None
    for r in _csv(root, "F6_provenance.csv"):
        assert int(r["parameters"]) == report.grounding.params_by_source.get(r["source"], 0)
        want = report.sensitivity.variance_by_source.get(r["source"])
        assert (float(r["variance_share"]) if r["variance_share"] else None) == want


def test_sharpness_rows_are_the_reports_contraction_and_widths(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    report = _report(root)
    assert report.sensitivity is not None and report.calibration is not None
    rows = _csv(root, "F7_sharpness.csv")
    contraction = {c.component: c.life_contraction for c in report.sensitivity.components}
    got = {r["item"]: float(r["value"]) for r in rows if r["panel"] == "life_contraction"}
    assert got == {k: v for k, v in contraction.items() if v is not None}
    acc = {(a.quantity, a.estimator): a for a in report.calibration.accuracy if a.part_class == "observed"}
    for r in (r for r in rows if r["panel"] == "width_over_true_rul"):
        a = acc[(r["item"], r["estimator"])]
        assert (float(r["value"]) if r["value"] else None) == a.width_relative_median
        assert (int(r["open_count"]), int(r["pairs"])) == (a.width_censored, a.alpha_lambda_n)


def test_maintenance_curves_are_the_backends_simulation_and_the_band_holds_the_median_run(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, manifest = built
    rows = _csv(root, "F8_maintenance.csv")
    assert {r["profile"] for r in rows} == {"nominal", "documented_maintenance"}
    by = {p: [r for r in rows if r["profile"] == p] for p in ("nominal", "documented_maintenance")}
    assert all(0.0 <= float(r["p05"]) <= float(r["p95"]) <= 1.0 for r in by["documented_maintenance"])
    assert all(r["p05"] == r["p95"] == "" for r in by["nominal"])  # drawn without its band: no band numbers
    assert float(by["nominal"][0]["median_params"]) == 1.0
    # with the 4000 h replacement inside the window, maintained health ends above unmaintained health
    assert float(by["documented_maintenance"][-1]["median_params"]) > float(by["nominal"][-1]["median_params"])
    rec = next(r for r in manifest.figures if r.id == "F8")
    assert rec.notes and "derived by simulation" in rec.notes[0] and "3 prior draws" in rec.notes[0]


def test_the_grid_is_f3s_rows_for_every_unit(built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, manifest = built
    grid = _csv(root, "F11_grid.csv")
    units = sorted({r["unit"] for r in grid})
    assert len(units) == 3  # every unit of the fixture
    fan = _csv(root, "F3_fan.csv")
    assert [r for r in grid if r["unit"] == fan[0]["unit"]] == fan
    rec = next(r for r in manifest.figures if r.id == "F11")
    assert rec.selection is not None and rec.selection.startswith(f"F3's unit: {fan[0]['unit']}")


def _patched(root: Path, settings: KafulSettings, report: EvaluationReport) -> Context:
    from dataclasses import replace as dc_replace

    from kaful.figures.sources import Inputs

    ctx = Context(spec=load_spec(root / "figures" / "figures.toml"), settings=settings, inputs=Inputs(root=root))
    real = ctx.lineage

    def lineage(name: str):  # type: ignore[no-untyped-def]
        return dc_replace(real(name), report=report)

    ctx.lineage = lineage  # type: ignore[method-assign]
    return ctx


def test_cost_rows_are_the_reports_phases_and_the_m1_baseline(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    from kaful.artifacts.evaluation import FileRef, M1Comparison
    from kaful.figures import f5_cost
    from kaful.figures.spec import CostSpec

    root, settings, _ = built
    report = _report(root)
    spec = CostSpec(id="F5", slug="cost", caption="c", kind="cost", lineages=("tiny",))
    with pytest.raises(ValueError, match="no M1 baseline"):
        f5_cost.extract(spec, _patched(root, settings, report))
    m1 = M1Comparison(source=FileRef(path="repo:x", sha256="0" * 64), cold_total_usd=3.0, rerun_total_usd=2.0,
                      by_phase_usd={"ingest": 2.0, "build": 1.0}, mapping={"1": "ingest", "6b": "build"})
    with_m1 = report.model_copy(update={"cost": report.cost.model_copy(update={"m1": m1})})
    (m,) = f5_cost.extract(spec, _patched(root, settings, with_m1))
    new = {p.phase: p for p in report.cost.by_phase}
    for p in m.phases:
        assert p.legacy == m1.by_phase_usd.get(p.phase)
        assert (p.low, p.high) == ((new[p.phase].cold_low_usd, new[p.phase].cold_usd) if p.phase in new else (0, 0))
    assert {p.phase for p in m.phases} >= {"ingest", "build"} and m.legacy_total == 3.0


def test_whatif_rows_are_the_reports_and_a_report_of_another_run_is_refused(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    from kaful.figures import f9_whatif
    from kaful.figures.sources import Inputs
    from kaful.figures.spec import WhatIfSpec

    root, settings, _ = built
    producer = read_manifest(_machine(root) / "outputs" / "runs" / EVALUATE_RUN)
    est_run = next(a.run_id for p in producer.phases if p.phase == "estimate" for a in p.artifacts)
    q = {"p05": 10.0, "p50": 20.0, "p95": None, "n_already_triggered": 0, "p_before_horizon": 0.9}
    report = {"run_id": est_run, "renew_at": {"filter_element_fe": 1.5}, "time_unit": "h", "horizon": 100.0,
              "disclaimer": "priors only",
              "from_new": {"events": [{"binding": "shaft_seal", "documented": q, "delayed": {**q, "p50": 15.0}},
                                      {"binding": "filter_element_fe", "documented": q, "delayed": q}]},
              "long_run": {"filter_element_fe_to_shaft_seal": {"rate_change": {"p05": 0.01, "p50": 0.1, "p95": 0.3},
                                                               "rate_change_r_known": {"low": 0.08, "high": 0.14}}}}
    path = _machine(root) / "outputs" / "whatif" / "w.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    spec = WhatIfSpec(id="F9", slug="w", caption="c", kind="whatif", lineage="tiny", file="outputs/whatif/w.json")
    ctx = Context(spec=load_spec(root / "figures" / "figures.toml"), settings=settings, inputs=Inputs(root=root))
    rows = f9_whatif.rows(f9_whatif.extract(spec, ctx))
    assert [(r["item"], r["case"], r["p50"]) for r in rows if r["panel"] == "time_to_limit"] == \
        [("shaft_seal", "documented", 20.0), ("shaft_seal", "delayed", 15.0)]  # the renewed part itself is left out
    (rate,) = [r for r in rows if r["panel"] == "long_run_rate_change"]
    assert (rate["p05"], rate["p50"], rate["p95"], rate["low"], rate["high"]) == (0.01, 0.1, 0.3, 0.08, 0.14)
    path.write_text(json.dumps({**report, "run_id": "20260101T000000Z-aaaaaa"}))
    with pytest.raises(ValueError, match="not this lineage's estimate"):
        f9_whatif.extract(spec, ctx)


def test_the_talk_sharpness_variant_holds_the_contraction_rows_only(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    root, _, _ = built
    full, talk = _csv(root, "F7_sharpness.csv"), _csv(root, "F7t_sharpness_talk.csv")
    assert talk == [r for r in full if r["panel"] == "life_contraction"] and len(talk) < len(full)


def test_the_whatif_subtitle_says_barely_only_when_every_median_moves_less_than_5_percent() -> None:
    from kaful.figures import f9_whatif as f9

    def data(late: float) -> f9.WhatIfData:
        times = (f9.TimeToLimit("b", "documented", 1.0, 100.0, None), f9.TimeToLimit("b", "delayed", 1.0, late, None))
        return f9.WhatIfData(label="m", time_unit="h", factor=1.5, horizon=200.0, times=times, rates=(),
                             disclaimer="d")

    assert f9.median_changes(data(97.0)) == {"b": pytest.approx(-0.03)}
    assert "barely any difference" in f9._subtitle(data(97.0), {"b": "b"}) and "b -3%" in f9._subtitle(
        data(97.0), {"b": "b"})
    assert "medians move" in f9._subtitle(data(80.0), {"b": "b"})


def test_report_quotes_are_one_sentence_at_most_and_abbreviations_do_not_end_one() -> None:
    from kaful.figures.report import short_quote

    q = "Approx. 60 degrees celsius above cooling air temperature or above cooling water inlet temperature"
    assert short_quote(q, 160) == q
    assert short_quote("Remove the element (AF). Clean it with air.", 160) == "Remove the element (AF).…"
    assert short_quote("Check e.g. the filter. Then the seal.", 160) == "Check e.g. the filter.…"
    long = "word " * 60
    cut = short_quote(long, 50)
    assert len(cut) <= 51 and cut.endswith("…")


def test_the_report_embeds_every_built_figure_and_reads_its_tables_from_the_artifacts(
        built: tuple[Path, KafulSettings, FiguresManifest]) -> None:
    import re

    root, _, manifest = built
    page = (root / "figures" / "report.html").read_text(encoding="utf-8")
    assert manifest.report is not None and sha256_file(root / "figures" / "report.html") == manifest.report.sha256
    assert not re.search(r'(?:src|href)="(?!data:)', page)  # self-contained: no external request
    assert page.count('class="unit') == 3 and page.count('class="unit on"') == 1  # one panel per unit
    for fid in ("F5", "F9", "F10"):  # not in the fixture spec: said so on the page, never silently dropped
        assert f"{fid} is not in this figure spec." in page
    report = _report(root)
    assert report.summary and html.escape(report.summary[0], quote=True) in page
    ps_path = resolve_in(_machine(root), _refs(root)[("parameterize", "param_set")])
    from kaful.artifacts.param_set import ParamSet

    ps = ParamSet.model_validate_json(ps_path.read_text())
    assert all(f"<td>{p.name}</td>" in page for p in ps.parameters)
    assert any(i.path.endswith("F2_tracking.svg") for i in manifest.report.inputs)


def test_the_report_refuses_a_figure_that_changed_after_the_build(
        tmp_path_factory: pytest.TempPathFactory) -> None:
    root = tmp_path_factory.mktemp("stale")
    spec, settings = make(root)
    build(spec, settings, root=root)
    (root / "figures" / "F4_calibration.svg").write_bytes(b"<svg/>")
    with pytest.raises(ValueError, match="F4_calibration.svg does not match its manifest sha256"):
        build(spec, settings, only=("F2",), root=root)
