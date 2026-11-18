"""Estimation core (M13): resampling, Liu-West, the particle filter against exact grid posteriors, open-loop,
forecast censoring, and seeding. A plain-numpy Dynamics stands in for a backend."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Lognormal
from kaful.config import EstimateSettings
from kaful.data.source import Observation
from kaful.estimation.cloud import Cloud, advance
from kaful.estimation.dynamics import BindingSpec
from kaful.estimation.predict import forecast, summarize
from kaful.estimation.rejuvenation import Transform, liu_west, shrinkage
from kaful.estimation.resampling import DegenerateWeights, ess, normalized, systematic

LIMIT = 1.0


class Linear1:
    """One binding: dD/dt = k, D_limit = LIMIT; optional scheduled replacement times."""

    def __init__(self, renews: bool = False, replace_at: tuple[float, ...] = ()) -> None:
        self._spec: tuple[BindingSpec, ...] = (BindingSpec(id="b", event="b_failure", limit_param="b.limit",
                                                         renews_at_limit=renews),)
        self.replace_at = replace_at

    @property
    def bindings(self) -> tuple[BindingSpec, ...]:
        return self._spec

    @property
    def fixed(self) -> Mapping[str, float]:
        return {"b.limit": LIMIT}

    def step(self, damage: Mapping[str, FloatArray], params: Mapping[str, FloatArray], t: float,
             dt: float) -> tuple[dict[str, FloatArray], frozenset[str]]:
        d = damage["b"] + dt * params["b.k"]
        if any(t <= r < t + dt for r in self.replace_at):
            return {"b": np.zeros_like(d)}, frozenset({"b"})
        return {"b": d}, frozenset()

    def reset_times(self, binding: str) -> tuple[float, ...]:
        return self.replace_at

    def rates(self, params: Mapping[str, FloatArray], t: float,
              damage: Mapping[str, FloatArray] | None = None) -> dict[str, FloatArray]:
        return {"b": np.asarray(params["b.k"], dtype=np.float64)}


def test_weights_ess_and_systematic_resampling() -> None:
    w = normalized(np.array([0.0, 0.0, np.log(2.0), -np.inf]))
    assert w.tolist() == [0.25, 0.25, 0.5, 0.0]
    assert ess(np.full(4, 0.25)) == pytest.approx(4.0)
    assert systematic(w, 0.5).tolist() == [0, 1, 2, 2]
    with pytest.raises(DegenerateWeights):
        normalized(np.array([-np.inf, -np.inf]))


@settings(max_examples=50, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=2, max_size=50),
       st.floats(min_value=0.0, max_value=0.999))
def test_systematic_resampling_keeps_each_particle_within_one_of_its_expected_count(raw: list[float], u: float) -> None:
    w = np.asarray(raw) + 1e-9
    w = w / w.sum()
    counts = np.bincount(systematic(w, u), minlength=w.size)
    assert counts.sum() == w.size
    assert np.all(np.abs(counts - w * w.size) < 1.0 + 1e-9)


def test_liu_west_keeps_the_weighted_mean_and_covariance() -> None:
    rng = np.random.default_rng(1)
    n = 200_000
    z = rng.multivariate_normal([0.0, 2.0], [[1.0, 0.6], [0.6, 2.0]], size=n).T
    params = {"a": np.exp(z[0]), "b": z[1]}
    transforms = {"a": Transform("log"), "b": Transform("identity")}
    w = np.full(n, 1.0 / n)
    out = liu_west(params, w, np.arange(n), transforms, 0.98, rng)
    moved = np.stack([np.log(out["a"]), out["b"]])
    assert moved.mean(axis=1) == pytest.approx(z.mean(axis=1), abs=0.01)
    assert np.cov(moved) == pytest.approx(np.cov(z), abs=0.02)
    a, h = shrinkage(0.98)
    assert a == pytest.approx(0.98980, abs=1e-5) and h == pytest.approx(0.1425, abs=1e-4)


def test_a_logit_transform_keeps_a_bounded_parameter_inside_its_support() -> None:
    t = Transform("logit", 0.9, 1.0)
    x = np.array([0.9, 0.93, 0.999999, 1.0])
    back = t.backward(t.forward(x) + 50.0)
    assert np.all((back >= 0.9) & (back <= 1.0))
    assert t.backward(t.forward(np.array([0.95])))[0] == pytest.approx(0.95)


def _filter(dyn: Any, cloud: Cloud, est: EstimateSettings, seed: int = 0, weigh: bool = True) -> Any:
    from kaful.artifacts.twin_definition import ObservationBinding, TemplateBinding
    from kaful.estimation.filter import GroupFilter
    from kaful.estimation.groups import Group
    from kaful.estimation.rejuvenation import transform_for

    binding = TemplateBinding(id="b", component="b", fault_mode=None, template="accumulation", template_version="1",
                              damage_state="b_damage", damage_unit="1", threshold_param="b.limit",
                              params={"k": "b.k"}, conditions={}, justification="test")
    obs = ObservationBinding(id="o", binding="b", channel="y", form="linear", form_version="1",
                             params={"y_healthy": "m.yh", "y_limit": "b.yl", "noise_sd": "m.sd"},
                             basis="declared", ref="test", justification="test")
    transforms = {pid: transform_for(Lognormal(median=1.0, sigma_log=1.0)) for pid in cloud.params}
    return GroupFilter(Group(bindings=("b",), observations=(obs,)), dyn, cloud, {"b": binding}, transforms, est,
                       1.0, np.random.SeedSequence(seed), 0, weigh=weigh)


class LinearObserved(Linear1):
    def __init__(self, yh: float, yl: float, sd: float, **kw: Any) -> None:
        super().__init__(**kw)
        self._f = {"b.limit": LIMIT, "m.yh": yh, "b.yl": yl, "m.sd": sd}

    @property
    def fixed(self) -> Mapping[str, float]:
        return self._f


def _grid_posterior(prior: Lognormal, times: FloatArray, y: FloatArray, yh: float, yl: float,
                    sd: float) -> tuple[FloatArray, FloatArray]:
    logk = np.linspace(np.log(prior.median) - 6 * prior.sigma_log, np.log(prior.median) + 6 * prior.sigma_log, 20001)
    k = np.exp(logk)
    lp = -0.5 * ((logk - np.log(prior.median)) / prior.sigma_log) ** 2
    mean = yh + (yl - yh) * np.clip(np.outer(k, times) / LIMIT, 0.0, 1.0)
    lp = lp + (-0.5 * ((y[None, :] - mean) / sd) ** 2).sum(axis=1)
    p = np.exp(lp - lp.max())
    return logk, np.cumsum(p) / p.sum()


@pytest.mark.parametrize("true_k", [0.004, 0.012, 0.03])
def test_the_particle_filter_matches_the_exact_posterior_of_k_on_a_linear_channel(true_k: float) -> None:
    """One binding, one linear channel, k the only unknown and almost no process noise: the posterior of log k is
    exact on a grid. The filter's 5/50/95% must match it."""
    prior, yh, yl, sd = Lognormal(median=0.01, sigma_log=1.0), 0.5, 1.5, 0.05
    rng = np.random.default_rng(7)
    times = np.arange(0.0, 60.0 + 1e-9, 2.0)
    y = yh + (yl - yh) * np.clip(true_k * times / LIMIT, 0.0, 1.0) + sd * rng.standard_normal(times.size)
    est = EstimateSettings(n_particles=4000, process_noise_cv=1e-4)
    dyn = LinearObserved(yh, yl, sd)
    cloud = Cloud(damage={"b": np.zeros(4000)}, renewals={},
                  params={"b.k": prior.sample(np.random.default_rng(3), 4000)}, log_w=np.zeros(4000))
    f = _filter(dyn, cloud, est)
    for t, v in zip(times, y):
        f.update(Observation(t=float(t), values={"y": float(v)}))
    logk, cdf = _grid_posterior(prior, times, y, yh, yl, sd)
    exact = {q: float(np.interp(q, cdf, logk)) for q in (0.05, 0.5, 0.95)}
    post = np.log(f.export(f.cloud.n).params["b.k"])
    width = exact[0.95] - exact[0.05]
    assert float(np.quantile(post, 0.5)) == pytest.approx(exact[0.5], abs=0.1 * width), (exact, f.diagnostics)
    pf_width = float(np.quantile(post, 0.95) - np.quantile(post, 0.05))
    # Liu-West jitter widens the posterior a little (measured 15-30% here); it must never narrow it.
    assert 0.9 * width <= pf_width <= 1.6 * width, (pf_width, width)


def _cloud(k: FloatArray) -> Cloud:
    return Cloud(damage={"b": np.zeros(k.size)}, renewals={}, params={"b.k": np.asarray(k, dtype=np.float64)},
                 log_w=np.zeros(k.size))


def test_a_new_part_crosses_at_its_rate_life_with_coefficient_of_variation_rho() -> None:
    """Process noise: D_limit / k is the mean time to the limit, and rho its coefficient of variation."""
    n, k, rho = 20_000, 0.01, 0.1
    c = _cloud(np.full(n, k))
    first = advance(c, Linear1(), 0.0, 300.0, 0.05, rho, np.random.default_rng(2))["b"]
    assert not np.isnan(first).any()
    assert first.mean() == pytest.approx(LIMIT / k, rel=0.01)
    assert first.std() / first.mean() == pytest.approx(rho, rel=0.05)


def test_a_threshold_rule_renews_and_counts_and_a_scheduled_reset_adds_no_noise() -> None:
    c = Cloud(damage={"b": np.zeros(3)}, renewals={"b": np.zeros(3)}, params={"b.k": np.full(3, 0.1)},
              log_w=np.zeros(3))
    first = advance(c, Linear1(renews=True), 0.0, 25.0, 0.5, 1e-6, np.random.default_rng(0))["b"]
    assert c.renewals["b"].tolist() == [2.0, 2.0, 2.0] and first == pytest.approx(np.full(3, 10.0), abs=0.6)
    d = _cloud(np.full(4, 0.01))
    advance(d, Linear1(replace_at=(7.0,)), 0.0, 7.5, 0.5, 0.5, np.random.default_rng(0))
    assert np.all(d.damage["b"] == 0.0)  # reset in the last step: exactly 0, no noise added


def test_a_forecast_keeps_every_sample_with_already_triggered_and_censored_decided_per_sample() -> None:
    c = _cloud(np.array([0.1, 0.01, 1e-9, 0.01]))
    c.damage["b"] = np.array([0.2, 1.3, 0.0, 0.5])  # sample 1 is past its limit at t_issue
    p = forecast(Linear1(), c, 100.0, 300.0, 0.5, 1e-6, np.random.default_rng(0))
    e = p.events[0]
    assert e.triggered.tolist() == [False, True, False, False]
    assert e.toe[0] == pytest.approx(108.0, abs=0.6) and e.toe[1] == 100.0 and np.isnan(e.toe[2])
    assert e.toe[3] == pytest.approx(150.0, abs=0.6)
    s = summarize(e)
    assert (s.n_samples, s.n_hits, s.n_censored, s.n_triggered) == (4, 3, 1, 1)
    assert s.p_event_before_horizon == 0.75 and s.percentiles["p95"] is None and s.percentiles["p05"] == 100.0


def test_open_loop_never_weighs_or_resamples() -> None:
    est = EstimateSettings(n_particles=50)
    prior = Lognormal(median=0.01, sigma_log=1.0)
    start = prior.sample(np.random.default_rng(3), 50)
    f = _filter(LinearObserved(0.5, 1.5, 0.05), _cloud(start), est, weigh=False)
    for t in range(0, 40, 2):
        f.update(Observation(t=float(t), values={"y": 1.4}))
    assert f.diagnostics.updates == 0 and np.all(f.cloud.log_w == 0.0)
    assert np.array_equal(f.cloud.params["b.k"], start)


def test_a_row_without_the_channel_contributes_nothing_and_time_cannot_go_back() -> None:
    est = EstimateSettings(n_particles=50)
    f = _filter(LinearObserved(0.5, 1.5, 0.05), _cloud(np.full(50, 0.01)), est)
    f.update(Observation(t=3.0, values={"y": None, "other": 2.0}))
    assert f.diagnostics.updates == 0
    with pytest.raises(ValueError, match="before"):
        f.update(Observation(t=1.0, values={}))


def test_the_filter_never_reads_the_global_numpy_random_state() -> None:
    def run(global_seed: int) -> FloatArray:
        np.random.seed(global_seed)
        prior = Lognormal(median=0.01, sigma_log=1.0)
        f = _filter(LinearObserved(0.5, 1.5, 0.05), _cloud(prior.sample(np.random.default_rng(3), 300)),
                    EstimateSettings(n_particles=300), seed=5)
        for t in range(0, 60, 2):
            f.update(Observation(t=float(t), values={"y": 0.5 + 0.012 * t}))
        out: FloatArray = f.export(f.cloud.n).params["b.k"]
        return out

    assert np.array_equal(run(1), run(2))


def _alarm_filter(dyn: Any, cloud: Cloud, est: EstimateSettings, channels: dict[str, list[str]],
                  monitored: frozenset[str], seed: int = 0) -> Any:
    from kaful.artifacts.twin_definition import ObservationBinding, TemplateBinding
    from kaful.estimation.filter import GroupFilter
    from kaful.estimation.groups import Group
    from kaful.estimation.rejuvenation import transform_for

    ids = sorted({b for bs in channels.values() for b in bs})
    bindings = {b: TemplateBinding(id=b, component=b, fault_mode=None, template="accumulation", template_version="1",
                                   damage_state=f"{b}_damage", damage_unit="1", threshold_param=f"{b}.limit",
                                   params={"k": f"{b}.k"}, conditions={}, justification="test") for b in ids}
    obs = tuple(sorted((ObservationBinding(id=f"{ch}_{b}", binding=b, channel=ch, form="threshold_event",
                                           form_version="1", params={}, basis="declared", ref="test",
                                           justification="test")
                        for ch, bs in channels.items() for b in bs), key=lambda o: o.id))
    transforms = {pid: transform_for(Lognormal(median=1.0, sigma_log=1.0)) for pid in cloud.params}
    return GroupFilter(Group(bindings=tuple(ids), observations=obs), dyn, cloud, bindings, transforms, est, 0.5,
                       np.random.SeedSequence(seed), 0, weigh=True, monitored=monitored)


def test_the_particle_filter_matches_the_exact_posterior_of_k_from_alarm_reports_and_silence() -> None:
    """A part renewed at its limit, seen only through an alarm at each renewal and silence in between. Given k, the
    gaps between reports are independent inverse Gaussians, and the open gap after the last report is censored:
    the posterior of log k is exact on a grid."""
    from kaful.observation.first_passage import log_density, log_survival

    prior, rho, n = Lognormal(median=0.01, sigma_log=0.8), 0.1, 4000
    reports, t_end = [95.0, 210.0, 300.0], 360.0
    rows = sorted({*np.arange(0.0, t_end + 1e-9, 5.0).tolist(), *reports})
    est = EstimateSettings(n_particles=n, process_noise_cv=rho)
    cloud = Cloud(damage={"b": np.zeros(n)}, renewals={"b": np.zeros(n)},
                  params={"b.k": prior.sample(np.random.default_rng(3), n)}, log_w=np.zeros(n))
    f = _alarm_filter(Linear1(renews=True), cloud, est, {"alarm": ["b"]}, frozenset({"alarm"}))
    for t in rows:
        f.update(Observation(t=float(t), values={"alarm": True if t in reports else None}))
    logk = np.linspace(np.log(prior.median) - 5 * prior.sigma_log, np.log(prior.median) + 5 * prior.sigma_log, 4001)
    k = np.exp(logk)
    lp = -0.5 * ((logk - np.log(prior.median)) / prior.sigma_log) ** 2
    gaps = np.diff([0.0, *reports])
    for g in gaps:
        lp = lp + log_density(np.full(k.size, g), np.full(k.size, LIMIT), k, rho * rho * LIMIT * k)
    lp = lp + log_survival(np.full(k.size, t_end - reports[-1]), np.full(k.size, LIMIT), k, rho * rho * LIMIT * k)
    p = np.exp(lp - lp.max())
    cdf = np.cumsum(p) / p.sum()
    exact = {q: float(np.interp(q, cdf, logk)) for q in (0.05, 0.5, 0.95)}
    post = np.log(f.export(n).params["b.k"])
    width = exact[0.95] - exact[0.05]
    assert float(np.quantile(post, 0.5)) == pytest.approx(exact[0.5], abs=0.1 * width), (exact, f.diagnostics)
    pf_width = float(np.quantile(post, 0.95) - np.quantile(post, 0.05))
    assert 0.9 * width <= pf_width <= 1.6 * width, (pf_width, width)
    assert f.diagnostics.reports == {"alarm": 3} and np.all(f.cloud.renewals["b"] == 3.0)


class Two(Linear1):
    """Two bindings, a and c, each dD/dt = its k, D_limit = LIMIT; c has no renewal rule."""

    def __init__(self) -> None:
        super().__init__()
        self._spec = (BindingSpec(id="a", event="a_failure", limit_param="a.limit", renews_at_limit=True),
                      BindingSpec(id="c", event="c_failure", limit_param="c.limit", renews_at_limit=False))

    @property
    def fixed(self) -> Mapping[str, float]:
        return {"a.limit": LIMIT, "c.limit": LIMIT}

    def step(self, damage: Mapping[str, FloatArray], params: Mapping[str, FloatArray], t: float,
             dt: float) -> tuple[dict[str, FloatArray], frozenset[str]]:
        return {b: damage[b] + dt * params[f"{b}.k"] for b in ("a", "c")}, frozenset()

    def reset_times(self, binding: str) -> tuple[float, ...]:
        return ()

    def rates(self, params: Mapping[str, FloatArray], t: float,
              damage: Mapping[str, FloatArray] | None = None) -> dict[str, FloatArray]:
        return {b: np.asarray(params[f"{b}.k"], dtype=np.float64) for b in ("a", "c")}


def test_a_report_on_a_shared_channel_is_attributed_to_the_binding_due_to_cross() -> None:
    """eversys: W-014 (water filter) and W-087/88 (top piston) share one bool channel."""
    n = 1000
    cloud = Cloud(damage={"a": np.zeros(n), "c": np.zeros(n)}, renewals={"a": np.zeros(n)},
                  params={"a.k": np.full(n, 0.01), "c.k": np.full(n, 0.002)}, log_w=np.zeros(n))
    f = _alarm_filter(Two(), cloud, EstimateSettings(n_particles=n), {"alarm": ["a", "c"]}, frozenset({"alarm"}))
    for t in (0.0, 50.0, 100.0):
        f.update(Observation(t=t, values={"alarm": True if t == 100.0 else None}))
    assert f.cloud.renewals["a"].mean() > 0.99  # a is due at 100; c only at 500
    assert np.all(f.cloud.damage["c"] < LIMIT)


def test_an_unmonitored_event_channel_is_not_used_and_is_listed() -> None:
    n = 100
    cloud = Cloud(damage={"b": np.zeros(n)}, renewals={"b": np.zeros(n)}, params={"b.k": np.full(n, 0.01)},
                  log_w=np.zeros(n))
    f = _alarm_filter(Linear1(renews=True), cloud, EstimateSettings(n_particles=n), {"alarm": ["b"]}, frozenset())
    f.update(Observation(t=10.0, values={"alarm": True}))
    assert f.diagnostics.updates == 0 and f.diagnostics.unmonitored_events == ("alarm",)


def test_the_conditioned_damage_is_below_the_limit_and_matches_surviving_paths() -> None:
    from kaful.estimation.events import Anchor, conditioned_damage

    n, k, rho, t = 40_000, 0.01, 0.2, 90.0
    drawn = conditioned_damage(Anchor(t=np.zeros(n), d=np.zeros(n)), t, LIMIT, np.full(n, k), rho,
                               np.random.default_rng(4))
    assert np.all(drawn < LIMIT)
    rng = np.random.default_rng(5)
    d, alive, h = np.zeros(n), np.ones(n, dtype=bool), 0.05
    for _ in range(int(t / h)):
        d = d + k * h + np.sqrt(rho * rho * LIMIT * k * h) * rng.standard_normal(n)
        alive &= d < LIMIT
    assert drawn.mean() == pytest.approx(d[alive].mean(), abs=0.01)
    assert drawn.std() == pytest.approx(d[alive].std(), rel=0.1)


def test_a_report_on_a_row_at_the_same_time_as_the_previous_row_is_not_dropped() -> None:
    """Regression (checkpoint 2): an integer usage clock rounds a grid row and an event row to the same t; the report
    on the second row was ignored because nothing had elapsed."""
    n = 200
    cloud = Cloud(damage={"b": np.zeros(n)}, renewals={"b": np.zeros(n)}, params={"b.k": np.full(n, 0.01)},
                  log_w=np.zeros(n))
    f = _alarm_filter(Linear1(renews=True), cloud, EstimateSettings(n_particles=n), {"alarm": ["b"]},
                      frozenset({"alarm"}))
    f.update(Observation(t=0.0, values={}))
    f.update(Observation(t=100.0, values={"alarm": None}))
    f.update(Observation(t=100.0, values={"alarm": True}))
    assert f.diagnostics.reports == {"alarm": 1} and np.all(f.cloud.renewals["b"] == 1.0)


def test_the_ukf_tracks_the_exact_posterior_of_k_while_the_channel_stays_linear() -> None:
    """The UKF is a Gaussian approximation: on an unsaturated linear channel its median of log k must lie well
    inside the exact posterior and its spread must be of the right size."""
    from kaful.artifacts.twin_definition import ObservationBinding, TemplateBinding
    from kaful.estimation.groups import Group
    from kaful.estimation.rejuvenation import transform_for
    from kaful.estimation.ukf import UKFGroup

    prior, yh, yl, sd, true_k = Lognormal(median=0.01, sigma_log=1.0), 0.5, 1.5, 0.05, 0.004
    rng = np.random.default_rng(7)
    times = np.arange(0.0, 60.0 + 1e-9, 2.0)
    y = yh + (yl - yh) * np.clip(true_k * times / LIMIT, 0.0, 1.0) + sd * rng.standard_normal(times.size)
    binding = TemplateBinding(id="b", component="b", fault_mode=None, template="accumulation", template_version="1",
                              damage_state="b_damage", damage_unit="1", threshold_param="b.limit",
                              params={"k": "b.k"}, conditions={}, justification="test")
    obs = ObservationBinding(id="o", binding="b", channel="y", form="linear", form_version="1",
                             params={"y_healthy": "m.yh", "y_limit": "b.yl", "noise_sd": "m.sd"},
                             basis="declared", ref="test", justification="test")
    n = 20_000
    prior_cloud = Cloud(damage={"b": np.zeros(n)}, renewals={},
                        params={"b.k": prior.sample(np.random.default_rng(3), n)}, log_w=np.zeros(n))
    est = EstimateSettings(process_noise_cv=1e-4, forecast_samples=20_000)
    u = UKFGroup(Group(bindings=("b",), observations=(obs,)), LinearObserved(yh, yl, sd), prior_cloud, {"b": binding},
                 {"b.k": transform_for(prior)}, est, 1.0, np.random.SeedSequence(0), 0)
    for t, v in zip(times, y):
        u.update(Observation(t=float(t), values={"y": float(v)}))
    logk, cdf = _grid_posterior(prior, times, y, yh, yl, sd)
    exact = {q: float(np.interp(q, cdf, logk)) for q in (0.05, 0.5, 0.95)}
    post = np.log(u.export(20_000).params["b.k"])
    width = exact[0.95] - exact[0.05]
    assert float(np.median(post)) == pytest.approx(exact[0.5], abs=0.5 * width)
    assert 0.5 * width <= float(np.quantile(post, 0.95) - np.quantile(post, 0.05)) <= 2.0 * width
    assert u.diagnostics.updates == times.size
