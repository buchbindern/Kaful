"""The what-if tool (M13b): paired forecasts under the documented profile and a delayed renewal, the long-run view,
the conclusions and the narrowing note. The small machine of the connection tests, from new."""

import numpy as np
import pytest

from kaful.config import KafulSettings
from kaful.estimation.predict import EventSamples
from kaful.phases.p8_estimate.tools.commands import _renew_at
from kaful.phases.p8_estimate.tools.whatif import Job, _change, long_run, narrowing, run_job
from tests.harness.kaful_factories import synthetic_dataset
from tests.unit.test_kaful_connections import R, _coupled
from tests.unit.test_kaful_scope import _contract

SETTINGS = KafulSettings()


def _ev(toe: list[float], triggered: list[bool] | None = None) -> EventSamples:
    t = np.asarray(toe, dtype=float)
    return EventSamples(event="x_failure", binding="x", toe=t,
                        triggered=np.asarray(triggered if triggered else [False] * t.size))


def test_the_change_pairs_samples_and_counts_only_those_that_move() -> None:
    doc = _ev([10.0, 20.0, 30.0, np.nan, 5.0], [False, False, False, False, True])
    late = _ev([10.0, 22.0, 36.0, np.nan, 5.0], [False, False, False, False, True])
    r = np.array([0.9, 0.5, 0.1, 0.3, 0.2])
    c = _change(doc, late, {R: r})
    assert (c.n_samples, c.n_both_hit, c.n_changed) == (5, 3, 2)
    assert c.relative["p50"] == pytest.approx(0.1)
    assert c.spread_share[R] is None  # two moved samples: too few to rank


def test_the_long_run_view_comes_from_r_alone_and_narrows_when_r_is_known() -> None:
    twin, ps = _coupled()
    lr = long_run(twin.connections[0], ps, 1.5, SETTINGS.whatif.known_halfwidth)
    assert 0.0 < lr.rate_change["p05"] < lr.rate_change["p50"] < lr.rate_change["p95"] <= 1 / 3 + 1e-9
    assert lr.rate_change["p50"] == pytest.approx(1 / 9, abs=1e-3)  # r = 0.5: (1 + 0.5 * 4/3) / 1.5 - 1
    assert lr.rate_change["p05"] < lr.rate_change_r_known["low"] < lr.rate_change_r_known["high"] < lr.rate_change["p95"]
    assert "+11%" in lr.conclusion and "W-014 does not inform" in lr.conclusion
    note = narrowing(twin.connections, {twin.connections[0].id: lr}, SETTINGS.whatif.known_halfwidth)
    assert note.startswith(f"{R} (the residual hardness fraction") and "datasheet" in note and "+-0.1" in note


def test_the_delay_moves_only_the_target_samples_still_running_when_the_source_passes_its_limit() -> None:
    twin, ps = _coupled()
    profile = next(p for p in twin.profiles if p.name == "documented_maintenance")
    dataset = synthetic_dataset().model_copy(update={"profile": profile, "horizon": profile.segments[-1].end,
                                                     "time_unit": "h"})
    settings = SETTINGS.model_copy(update={"estimate": SETTINGS.estimate.model_copy(
        update={"forecast_samples": 200, "n_particles": 200})})
    job = Job(twin, ps, _contract(), dataset, settings, {"filter_element_fe": 1.5},
              ("filter_element_fe", "shaft_seal"), {"shaft_seal": ("shaft_seal.k", R)}, None)
    [new] = run_job(job)
    by = {e.binding: e for e in new.events}
    seal, filt = by["shaft_seal"], by["filter_element_fe"]
    assert seal.change.n_changed > 0 and seal.change.relative["p50"] is not None
    assert seal.change.relative["p05"] is not None and seal.change.relative["p05"] <= 0.0
    # the filter's first W-014 is the same crossing: the delay acts only after it
    assert filt.documented.p05 == filt.delayed.p05
    assert run_job(job)[0] == new  # the same seeds: the same report


def test_renew_at_arguments_parse_and_bad_ones_are_refused() -> None:
    assert _renew_at(["water_filter=1.5"]) == {"water_filter": 1.5}
    with pytest.raises(ValueError, match="BINDING=FACTOR"):
        _renew_at(["water_filter"])
