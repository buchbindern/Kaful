"""Sensitivity and summary (M14 checkpoint 3): posterior contraction and remaining shares, life contraction on the
identifiable quantity, the data-recommendation rules, variance by source, and the "calibrated but not sharp" summary
computed from the report's own numbers."""

import numpy as np
import pytest

from kaful.artifacts.evaluation import AccuracyScore, Bounds, Calibration, ComponentSensitivity, PriorEffect, Verdict
from kaful.artifacts.forecast import ParamSummary
from kaful.config import EvaluateSettings
from kaful.phases.p9_evaluate.g_sensitivity import (
    LifeParams,
    _life_contraction,
    by_source,
    contraction,
    correlation_ratio,
    recommendations,
)
from kaful.phases.p9_evaluate.helpers.summary import summary
from tests.harness import kaful_factories as f

S = EvaluateSettings()


def test_contraction_is_on_the_log_scale_for_positive_parameters() -> None:
    p = ParamSummary(param_id="a.k", role="rate", prior={"p05": 1.0, "p50": 10.0, "p95": 100.0},
                     posterior={"p05": 10.0, "p50": 20.0, "p95": 100.0})
    assert contraction(p) == pytest.approx(0.5)
    q = ParamSummary(param_id="a.y", role="nuisance", prior={"p05": -1.0, "p50": 0.0, "p95": 1.0},
                     posterior={"p05": -0.5, "p50": 0.0, "p95": 0.5})
    assert contraction(q) == pytest.approx(0.5)


def test_correlation_ratio_finds_what_drives_the_spread() -> None:
    rng = np.random.default_rng(0)
    x, noise = rng.normal(size=4000), rng.normal(size=4000)
    assert correlation_ratio(x, 2 * x + 0.1 * noise, 10) == pytest.approx(0.97, abs=0.02)
    unrelated = correlation_ratio(noise, 2 * x, 10)
    assert unrelated is not None and unrelated < 0.02
    assert correlation_ratio(np.ones(10), x[:10], 10) is None


def test_life_contraction_uses_the_identifiable_life_not_k_alone() -> None:
    rng = np.random.default_rng(1)
    life = np.exp(rng.normal(5.0, 0.1, 2000))  # posterior life tight, but k and D_limit each wide and correlated
    dl = np.exp(rng.normal(7.0, 1.0, 2000))
    rows = [{"w.k": str(d / lf), "w.damage_limit": str(d)} for d, lf in zip(dl, life)]
    lp = LifeParams(k="w.k", damage_limit="w.damage_limit", damage_limit_median=1.0,
                    prior_log_width=2 * 1.6449 * 1.0)  # a lognormal life prior with sigma 1
    assert _life_contraction(rows, lp, (0.05, 0.95)) == pytest.approx(0.1, abs=0.01)


def _effect(param: str, source: str, share: float) -> PriorEffect:
    return PriorEffect(param=param, source=source, role="rate", share=share, low_median=1.0, high_median=2.0)


def test_recommendation_rules_follow_what_observes_the_part_and_its_dominant_prior() -> None:
    from kaful.artifacts.param_set import ComponentTemplate

    twin = f.twin_definition()
    binding = twin.bindings[0]
    template = ComponentTemplate(component=binding.component, family="accumulation", decided_by=None,
                                 flags=("load ratings needed: declare bearing ratings",))
    unobserved = recommendations(binding.component, binding.id, "unobserved", (_effect("x.k", "default", 0.9),),
                                 template, "llm_judgement", twin, S)
    assert len(unobserved) == 4 and "nothing observes it" in unobserved[0] and "llm_judgement" in unobserved[1]
    assert "load ratings" in unobserved[2] and "x.k" in unobserved[3]
    observed = recommendations(binding.component, binding.id, "observed", (_effect("x.k", "calibrated", 1.0),),
                               template, "calibrated", twin, S)
    assert observed == (S.recommendations["load_ratings"].format(component=binding.component),)


def test_variance_by_source_normalises_each_component_then_averages() -> None:
    def comp(name: str, effects: tuple[PriorEffect, ...]) -> ComponentSensitivity:
        return ComponentSensitivity(component=name, part_class="observed", life_source="x", sigma_log_rul=1.0,
                                    effects=effects, remaining_spread=1.0)
    out = by_source([comp("a", (_effect("a.k", "llm_judgement", 0.6), _effect("a.l", "default", 0.2))),
                     comp("b", (_effect("b.k", "calibrated", 1.0),))])
    assert out == pytest.approx({"calibrated": 0.5, "default": 0.125, "llm_judgement": 0.375})


def _calibration(coverage: float, mass_pass: int, n: int) -> Calibration:
    ref = f.evaluation_report().lineage[3].artifacts[0]
    acc = AccuracyScore(estimator="particle_filter", quantity="a_failure", part_class="observed",
                        alpha_lambda_pass=mass_pass, alpha_lambda_n=n, rmse_n=0, rmse_excluded=0,
                        width_relative_median=1.9, width_ratio_open_loop=0.62, horizon_units=0)
    v = Verdict(quantity="a_failure", coverage=coverage, units=13, ci_units=Bounds(low=0.6, high=0.97),
                verdict="within", ci_overlaps_target=True)
    return Calibration(forecast_set=ref, dataset=ref, interval=(0.05, 0.95), alpha=0.2, beta=0.5,
                       primary_estimator="particle_filter", process_noise_cv=0.1, cells=(), accuracy=(acc,),
                       verdicts=(v,))


def test_the_summary_says_calibrated_but_not_sharp_only_when_the_numbers_do() -> None:
    sens_comp = ComponentSensitivity(component="a", part_class="unobserved", life_source="llm_judgement",
                                     sigma_log_rul=1.2, effects=(), remaining_spread=1.2)
    from kaful.artifacts.evaluation import Sensitivity
    sens = Sensitivity(param_set=f.evaluation_report().lineage[3].artifacts[0], samples=10, seed=0,
                       components=(sens_comp,), variance_by_source={"llm_judgement": 1.0})
    blunt = summary(_calibration(0.88, 0, 33), sens, [])
    assert blunt[0].startswith("Calibrated but not sharp: on 1 observed quantity") and "88% of the time" in blunt[0]
    assert "1.9x the true RUL (0.62x open-loop's" in blunt[0]
    assert blunt[1].startswith("Cause: wide priors (1 of 1") and "documented lives (a)" in blunt[2]
    sharp = summary(_calibration(0.9, 30, 33), sens, [])
    assert sharp[0].startswith("Calibrated:") and len(sharp) == 1
    assert summary(None, None, [])  == ()
