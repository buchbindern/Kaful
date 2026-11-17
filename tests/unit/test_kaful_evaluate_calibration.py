"""Calibration (M14 checkpoint 2): metrics on worked examples, every coverage class, the target judged on observed
parts only, an end-to-end pass over the small estimate fixture, and the guards: truth is read only by scoring code,
and evaluate reads rho from the forecasts, never from (tunable) estimate settings."""

import re
from pathlib import Path

import numpy as np
import pytest

from kaful.artifacts.evaluation import PAIR_CLASSES, CoverageCell
from kaful.artifacts.forecast import Forecast
from kaful.config import REPO_ROOT, EvaluateSettings
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.f_calibration import Collected, _cell, _verdict, collect_forecast, part_classes
from kaful.phases.p9_evaluate.helpers.metrics import (
    alpha_lambda,
    alpha_mass,
    classify,
    median_error,
    prognostic_horizon,
    triggered,
    wilson,
)
from kaful.scoring.truth import load_units

QS = (0.05, 0.95)
S = EvaluateSettings()


def test_wilson_matches_the_textbook_values() -> None:
    b = wilson(9, 10, 1.959964)
    assert b is not None and (round(b.low, 3), round(b.high, 3)) == (0.596, 0.982)
    assert wilson(0, 0, 1.96) is None
    full = wilson(20, 20, 1.959964)
    assert full is not None and full.high == 1.0 and round(full.low, 3) == 0.839


def test_every_coverage_class() -> None:
    s = np.linspace(100.0, 200.0, 101)
    censored = np.concatenate([np.linspace(100.0, 200.0, 50), np.full(51, np.inf)])
    assert classify(s, 150.0, 50.0, QS) == "covered"
    assert classify(s, 300.0, 50.0, QS) == "missed"
    assert classify(censored, None, 50.0, QS) == "censored_covered"  # no true crossing; the interval reaches past
    assert classify(s, None, 50.0, QS) == "censored_missed"  # no true crossing; the interval ends inside the horizon
    assert classify(censored, 400.0, 50.0, QS) == "covered"  # truth after the samples' finite part, interval open
    assert classify(s, 50.0, 50.0, QS) == "already_triggered"
    assert classify(s, 150.0, 50.0, QS, decidable=False) == "undecidable"


def test_alpha_lambda_mass_and_already_triggered() -> None:
    s = np.array([90.0, 100.0, 110.0, 200.0, np.inf])  # true RUL 100 from t = 0: three of five within +-20%
    assert alpha_mass(s, 100.0, 0.0, 0.2) == pytest.approx(0.6)
    assert alpha_lambda(s, 100.0, 0.0, 0.2, 0.5) and not alpha_lambda(s, 100.0, 0.0, 0.2, 0.7)
    assert triggered(np.array([5.0, 5.0, 9.0]), 5.0) and not triggered(np.array([5.0, 9.0, 9.0]), 5.0)


def test_prognostic_horizon_counts_from_the_issue_after_which_every_forecast_passes() -> None:
    issues = [(0.0, 100.0, False), (25.0, 100.0, True), (50.0, 100.0, False), (75.0, 100.0, True)]
    h = prognostic_horizon(issues)
    assert h is not None and (h.value, h.share) == (25.0, 0.25)
    always = prognostic_horizon([(0.0, 100.0, True), (50.0, 100.0, True)])
    assert always is not None and (always.value, always.share) == (100.0, 1.0)
    last_fails = prognostic_horizon([(0.0, 100.0, True), (50.0, 100.0, False)])
    assert last_fails is not None and last_fails.value == 0.0
    # a later issue aiming at the next crossing (after a renewal) is not part of this horizon
    renewed = prognostic_horizon([(0.0, 100.0, False), (50.0, 100.0, True), (120.0, 300.0, False)])
    assert renewed is not None and renewed.value == 50.0
    assert prognostic_horizon([(0.0, None, True), (50.0, 50.0, True)]) is None


def test_median_error_is_none_when_the_median_is_censored() -> None:
    assert median_error(np.array([10.0, 20.0, 30.0]), 25.0, 0.0) == (-5.0, -0.2)
    assert median_error(np.array([10.0, np.inf, np.inf]), 25.0, 0.0) is None


def _cell_with(share_covered: int, pairs: int, units: int, part: str = "observed") -> CoverageCell:
    classes = {k: 0 for k in PAIR_CLASSES}
    classes["covered"], classes["missed"] = share_covered, pairs - share_covered
    return CoverageCell(estimator="particle_filter", quantity="x", part_class=part, issue_fraction=None,
                        classes=classes, pairs=pairs, covered=share_covered, units=units, share=share_covered / pairs,
                        ci_units=wilson(share_covered / pairs * units, units, S.ci_z))


def test_the_target_is_85_to_95_percent_and_over_coverage_is_called_above() -> None:
    within, above, below = (_verdict(_cell_with(k, 100, 20)) for k in (90, 97, 80))
    assert within is not None and within.verdict == "within" and within.ci_overlaps_target
    assert above is not None and above.verdict == "above"
    assert below is not None and below.verdict == "below"
    narrow = _verdict(_cell_with(990, 1000, 1000))
    assert narrow is not None and narrow.verdict == "above" and not narrow.ci_overlaps_target


def test_calibration_end_to_end_on_the_small_estimate_fixture(tmp_path: Path) -> None:
    from tests.unit.test_kaful_estimate_phase import LAST, _estimate

    out, _ = _estimate(tmp_path)
    twin, dataset = LAST["twin"], LAST["dataset"]
    units = load_units(tmp_path, dataset)
    c = Collected(part=part_classes(twin))
    for e in out.forecast_set.forecasts:
        f = Forecast.model_validate_json(resolve_in(tmp_path, e.path).read_text())
        collect_forecast(f, units[e.source], twin, tmp_path, S, c)
    assert {est for est, _ in c.records} == {"open_loop", "particle_filter"}
    for (est, q), recs in c.records.items():
        assert len(recs) == 3 * 2  # 3 units x 2 issue times
        cell = _cell(est, q, c.part[c.binding_of[q]], None, recs, S)
        assert sum(cell.classes.values()) == 6 and cell.classes["undecidable"] == 0
        assert cell.pairs == cell.classes["covered"] + cell.classes["missed"] + cell.classes["censored_covered"] + \
            cell.classes["censored_missed"]
    assert set(c.part.values()) <= {"observed", "joined", "unobserved", "not_identifiable"}
    assert "observed" in c.part.values()


def test_only_scoring_code_reads_synthetic_truth() -> None:
    allowed = {"kaful/scoring/truth.py", "kaful/phases/p9_evaluate/f_calibration.py",
               "kaful/phases/p8_estimate/tools/score_synthetic.py"}
    readers = set()
    for path in sorted((REPO_ROOT / "kaful").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"kaful\.scoring\.truth|SyntheticTruth\.model_validate|load_units\(", text):
            readers.add(path.relative_to(REPO_ROOT).as_posix())
    assert readers - {"kaful/scoring/truth.py"} <= allowed, readers
    assert not any(p.startswith("kaful/phases/p8_estimate/") and "/tools/" not in p for p in readers)


def test_evaluate_reads_rho_from_the_forecasts_never_from_estimate_settings() -> None:
    for path in sorted((REPO_ROOT / "kaful/phases/p9_evaluate").rglob("*.py")):
        assert ".estimate." not in path.read_text(encoding="utf-8").replace("lineage.", ""), path
