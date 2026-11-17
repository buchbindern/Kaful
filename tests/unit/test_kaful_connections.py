"""Connections (M13b): a declared coupling from one binding's damage to another's rate, through every phase it
touches. The small machine of the scope, parameterize and build tests gets filter_element_fe -> shaft_seal
(hardness_passthrough), with the filter replaced on threshold or not."""

from typing import Any

import numpy as np
import pytest

from kaful.artifacts.distributions import Lognormal
from kaful.artifacts.forecast import OperatingProfile, ProfileSegment
from kaful.artifacts.twin_definition import Connection, ObservationBinding, TwinDefinition
from kaful.backends.progpy.backend import ProgPyBackend
from kaful.config import DeclaredConnection, DeclaredMaintenance, KafulSettings
from kaful.estimation.cloud import Cloud, advance
from kaful.estimation.dynamics import BindingSpec
from kaful.estimation.groups import groups, read_by
from kaful.phases.p4_parameterize.g_coupling_priors import FLAG_JUDGED_BASIS, coupling_priors
from kaful.phases.p5_build.e_bind_connections import bind_connections
from kaful.phases.p6_verify import phase as verify_phase
from kaful.phases.p6_verify.d_quarantine import quarantine, reduce
from kaful.artifacts.verification import CheckResult
from kaful.twin_checks import connection_problems
from tests.harness.kaful_factories import BASE
from tests.unit.test_kaful_build import _twin
from tests.unit.test_kaful_scope import _contract, _index, _machine

SETTINGS = KafulSettings()
LINK = DeclaredConnection(from_component="filter_element_fe", to_components=("shaft_seal",),
                          form="hardness_passthrough", reason="a spent filter lets grit through to the seal",
                          doc_id="manual", page=1, quote="Keep the area clean.")
RENEWED = (DeclaredMaintenance(component="filter_element_fe", action="replace", on_threshold=True),)
R = "filter_element_fe.residual_fraction"


def _coupled(link: DeclaredConnection = LINK, renewed: bool = True) -> tuple[TwinDefinition, Any]:
    machine = _machine(connection=(link,), maintenance=RENEWED if renewed else ())
    twin, ps, failures = _twin(machine)
    assert failures == ()
    return twin, ps


# parameterize, step g


def test_a_declared_connection_gets_one_wide_default_strength_prior_on_its_from_component() -> None:
    machine = _machine(connection=(LINK,))
    out = coupling_priors(machine, ["filter_element_fe", "shaft_seal"], ["shaft_seal"], SETTINGS.parameterize)
    [x] = out.params
    assert (x.param_id, x.role, x.unit, x.source, x.confidence) == (R, "coupling", "1", "default", 0.0)
    assert (x.prior.low, x.prior.high) == (0.0, 1.0)  # type: ignore[union-attr]
    assert "no documented value" in out.flags["filter_element_fe"][0]
    assert any("depends on filter_element_fe" in f for f in out.flags["shaft_seal"])
    assert FLAG_JUDGED_BASIS.format(source="filter_element_fe") in out.flags["shaft_seal"]
    assert out.failures == []


def test_a_connection_without_full_models_at_its_ends_gets_no_prior() -> None:
    out = coupling_priors(_machine(connection=(LINK,)), ["filter_element_fe"], [], SETTINGS.parameterize)
    assert out.params == [] and out.flags == {}


# build, step e


def test_a_quote_found_on_its_page_makes_the_link_manual() -> None:
    twin, _ = _coupled()
    bound = bind_connections(_machine(connection=(LINK,)), twin.bindings, _index(), SETTINGS.understand)
    [c] = bound.connections
    assert (c.id, c.from_binding, c.to_binding, c.source, c.ref) == (
        "filter_element_fe_to_shaft_seal", "filter_element_fe", "shaft_seal", "manual", "manual p1 (label 1)")
    assert c.params == {"residual_fraction": R} and c.evidence[0].verification == "verbatim"


def test_a_quote_not_on_its_page_downgrades_the_link_to_declared_and_flags_it() -> None:
    twin, _ = _coupled()
    wrong = LINK.model_copy(update={"quote": "The boilers will be damaged due to limescale."})
    bound = bind_connections(_machine(connection=(wrong,)), twin.bindings, _index(), SETTINGS.understand)
    [c] = bound.connections
    assert (c.source, c.downgraded_from) == ("deployment", "manual")
    assert c.ref == "machine.toml [[connection]] filter_element_fe -> shaft_seal"
    assert any("not in the cited page's raw text" in f for f in bound.flags["shaft_seal"])


def test_a_connection_to_a_component_without_a_binding_is_skipped() -> None:
    twin, _ = _coupled()
    to_motor = LINK.model_copy(update={"to_components": ("motor",)})
    bound = bind_connections(_machine(connection=(to_motor,)), twin.bindings, _index(), SETTINGS.understand)
    assert bound.connections == [] and bound.skipped == ["filter_element_fe -> motor"]


def test_the_built_twin_carries_the_connection_and_the_prior() -> None:
    twin, ps = _coupled()
    assert [c.id for c in twin.connections] == ["filter_element_fe_to_shaft_seal"]
    assert ps.get(R).role == "coupling"
    assert any("rate scaled by filter_element_fe" in f for f in twin.flags["shaft_seal"])


# static checks


def _conn(twin: TwinDefinition, **change: Any) -> TwinDefinition:
    return twin.model_copy(update={"connections": tuple(c.model_copy(update=change) for c in twin.connections)})


def test_static_checks_pass_on_the_built_connection() -> None:
    twin, ps = _coupled()
    assert connection_problems(twin, ps) == []


def test_a_self_link_or_a_cycle_fails_connection_endpoints() -> None:
    twin, ps = _coupled()
    self_link = _conn(twin, to_binding="filter_element_fe")
    assert any(p.check == "connection_endpoints" and "to itself" in p.message
               for p in connection_problems(self_link, ps))
    back = twin.connections[0].model_copy(update={"id": "shaft_seal_to_filter_element_fe",
                                                  "from_binding": "shaft_seal", "to_binding": "filter_element_fe"})
    looped = twin.model_copy(update={"connections": (*twin.connections, back)})
    assert any(p.check == "connection_endpoints" and "cycle" in p.message for p in connection_problems(looped, ps))


def test_an_unknown_version_a_missing_or_unbounded_strength_or_a_wrong_role_fails() -> None:
    twin, ps = _coupled()
    assert "version '9'" in connection_problems(_conn(twin, form_version="9"), ps)[0].message
    assert any(p.check == "params_exist" for p in connection_problems(_conn(twin, params={
        "residual_fraction": "filter_element_fe.nope"}), ps))
    wide = ps.model_copy(update={"parameters": tuple(
        x.model_copy(update={"prior": Lognormal(median=0.5, sigma_log=1.0), "role": "rate"}) if x.param_id == R else x
        for x in ps.parameters)})
    messages = [p.message for p in connection_problems(twin, wide)]
    assert any("outside the slot's [0, 1]" in m for m in messages)
    assert any("role 'rate', not coupling" in m for m in messages)


# render and groups


def _profile(end: float = 1e6) -> OperatingProfile:
    return OperatingProfile(name="flat", segments=(ProfileSegment(start=0.0, end=end, values={}),))


def test_the_target_rate_is_k_times_m_of_the_source_fraction() -> None:
    twin, ps = _coupled()
    built = ProgPyBackend().build(twin, ps, _contract())
    dyn = ProgPyBackend().dynamics(built, _profile(), ("filter_element_fe", "shaft_seal"), (R,))
    k = ps.get("shaft_seal.k").prior.quantile(0.5)
    limit = ps.get("filter_element_fe.damage_limit").prior.quantile(0.5)
    r = np.array([0.0, 0.5, 1.0])
    for fraction, expect in ((0.0, r / ((1 + r) / 2)), (1.0, 2 / (1 + r)), (3.0, 2 / (1 + r))):
        rates = dyn.rates({R: r}, 0.0, {"filter_element_fe": np.full(3, fraction * limit), "shaft_seal": np.zeros(3)})
        assert rates["shaft_seal"] == pytest.approx(k * expect)
    assert dyn.rates({R: r}, 0.0)["shaft_seal"] == pytest.approx(k * r / ((1 + r) / 2))  # no damage given: D = 0


def test_a_target_cannot_be_rendered_without_its_source() -> None:
    twin, ps = _coupled()
    built = ProgPyBackend().build(twin, ps, _contract())
    with pytest.raises(ValueError, match="fed by"):
        ProgPyBackend().dynamics(built, _profile(), ("shaft_seal",), ())


def _observed(twin: TwinDefinition, binding: str) -> TwinDefinition:
    obs = ObservationBinding(id=f"{binding}_on_alarm", binding=binding, channel="alarm", form="threshold_event",
                             form_version="1", basis="declared", ref="machine.toml [[observation]]", justification="j")
    return twin.model_copy(update={"observations": (obs,)})


def test_a_connection_joins_an_unobserved_target_to_its_observed_source() -> None:
    """The eversys boilers join the water filter's W-014 group (M13b); before, groups joined on channels only."""
    twin, _ = _coupled()
    found = groups(_observed(twin, "filter_element_fe"))
    assert [(g.bindings, g.observed) for g in found] == [
        (("filter_element_fe", "shaft_seal"), True), (("bearings",), False)]
    assert R in read_by(twin, found[0])


def test_a_connection_between_unobserved_bindings_keeps_them_in_the_unobserved_group() -> None:
    twin, _ = _coupled()
    assert [(g.bindings, g.observed) for g in groups(_observed(twin, "bearings"))] == [
        (("bearings",), True), (("filter_element_fe", "shaft_seal"), False)]


# verify


def test_a_coupled_twin_passes_with_its_coupled_crossing_and_an_unchanged_life_check() -> None:
    for renewed in (True, False):
        twin, ps = _coupled(renewed=renewed)
        done = verify_phase.verify({**BASE}, {**BASE}, twin, ps, _contract(), SETTINGS, False)
        assert done.status == "passed", done.failures
        by = {(r.target, r.check): r for r in done.report.results}
        coupled = by[("shaft_seal", "coupled_life_consistency")]
        assert coupled.status == "pass" and coupled.measured["expected_time"] != coupled.measured["uncoupled_life"]
        assert by[("shaft_seal", "life_consistency")].status == "pass"
        assert by[("bearings", "coupled_life_consistency")].status == "skipped"
        assert "nominal_uncoupled@q0.5" in [r.name for r in done.report.runs]


def test_a_twin_without_connections_runs_no_uncoupled_simulation() -> None:
    twin, ps, _ = _twin()
    done = verify_phase.verify({**BASE}, {**BASE}, twin, ps, _contract(), SETTINGS, False)
    assert "nominal_uncoupled@q0.5" not in [r.name for r in done.report.runs]


def test_the_component_a_quarantined_source_feeds_is_quarantined_with_it_but_not_the_reverse() -> None:
    """Logged in M11: dependents of a quarantined component, now that connections exist."""
    twin, _ = _coupled()
    held = quarantine(twin, [CheckResult(check="monotone", target="filter_element_fe", status="fail", message="m")])
    assert [(q.component, q.checks, q.dependents) for q in held] == [
        ("filter_element_fe", ("monotone",), ("shaft_seal",)), ("shaft_seal", ("depends_on_quarantined",), ())]
    assert reduce(twin, held, {**BASE}).connections == ()
    held = quarantine(twin, [CheckResult(check="monotone", target="shaft_seal", status="fail", message="m")])
    assert [q.component for q in held] == ["shaft_seal"]
    assert reduce(twin, held, {**BASE}).connections == ()


# the what-if: a renewal delayed past the limit


class _Constant:
    """One renewing binding gaining 1 per time unit, D_limit 10 (a Dynamics for cloud.advance)."""

    def __init__(self, factor: float) -> None:
        self.bindings = (BindingSpec(id="f", event="f_failure", limit_param="f.limit", renews_at_limit=True,
                                     renew_factor=factor),)
        self.fixed = {"f.limit": 10.0}

    def step(self, damage: Any, params: Any, t: float, dt: float) -> Any:
        return {"f": damage["f"] + dt}, frozenset()

    def reset_times(self, binding: str) -> tuple[float, ...]:
        return ()

    def rates(self, params: Any, t: float, damage: Any = None) -> dict[str, Any]:
        return {"f": np.ones(1)}


@pytest.mark.parametrize(("factor", "renewals"), [(1.0, 3.0), (1.5, 2.0)])
def test_a_delayed_renewal_still_crosses_at_the_limit_and_renews_later(factor: float, renewals: float) -> None:
    cloud = Cloud(damage={"f": np.zeros(1)}, renewals={"f": np.zeros(1)}, params={}, log_w=np.zeros(1))
    first = advance(cloud, _Constant(factor), 0.0, 30.0, 1.0, 0.0, np.random.default_rng(0))  # type: ignore[arg-type]
    assert first["f"][0] == 10.0 and cloud.renewals["f"][0] == renewals and cloud.damage["f"][0] == 0.0


def test_a_profile_without_a_renewal_what_if_serializes_as_before() -> None:
    plain = _profile()
    assert "renew_at" not in plain.model_dump(mode="json")
    delayed = plain.model_copy(update={"renew_at": {"filter_element_fe": 1.5}})
    assert delayed.model_dump(mode="json")["renew_at"] == {"filter_element_fe": 1.5}


def test_a_renewal_what_if_is_refused_while_filtering_an_alarm_observed_binding() -> None:
    twin, ps = _coupled()
    twin = _observed(twin, "filter_element_fe")
    built = ProgPyBackend().build(twin, ps, _contract())
    est = SETTINGS.estimate.model_copy(update={"n_particles": 10})
    with pytest.raises(ValueError, match="applies to forecasts"):
        ProgPyBackend().estimate(built, ps, _profile().model_copy(update={"renew_at": {"filter_element_fe": 1.5}}),
                                 method="particle_filter", dt=1.0, settings=est, seed=np.random.SeedSequence(1),
                                 monitored_events=("alarm",))


def test_connection_model_rejects_an_unknown_binding() -> None:
    twin, _ = _coupled()
    with pytest.raises(ValueError, match="unknown binding 'nope'"):
        TwinDefinition.model_validate({**twin.model_dump(), "connections": (
            {**twin.connections[0].model_dump(), "to_binding": "nope"},)})
    assert isinstance(twin.connections[0], Connection)


# scoring: a part joined only through a connection


def test_a_part_joined_only_through_a_connection_is_listed_for_the_prior_check() -> None:
    from kaful.scoring.groups import joined_bindings
    twin, _ = _coupled()
    assert joined_bindings(_observed(twin, "filter_element_fe")) == ("shaft_seal",)
    assert joined_bindings(twin) == ()


def test_the_ks_check_accepts_draws_from_the_prior_and_rejects_a_moved_posterior() -> None:
    from kaful.phases.p8_estimate.tools.score_synthetic import ks_critical, ks_distance
    rng = np.random.default_rng(0)
    prior, crit = rng.uniform(0, 1, 20000), ks_critical(1000, 20000, 0.01)
    assert crit == pytest.approx(1.6276 * np.sqrt(21000 / 2e7), rel=1e-3)
    assert ks_distance(rng.uniform(0, 1, 1000), prior) < crit
    assert ks_distance(rng.uniform(0.1, 1, 1000), prior) > crit
