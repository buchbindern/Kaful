"""Unread bindings (M13b): in an observed group, a binding no observation's likelihood depends on (the eversys boilers,
joined to W-014's group only through the water filter's connection) is not carried through resampling. The filter
tracks the source's path integrals per particle, and at export the binding's parameters are drawn exactly from the
prior and its damage from the path (decided 2026-11-17: particle impoverishment had made their posterior a clump of
a few ancestors' prior draws). Small machine of the connection tests: filter_element_fe observed by an alarm,
shaft_seal fed by it."""

from typing import Any

import numpy as np
import pytest

from kaful.backends.progpy.backend import ProgPyBackend
from kaful.config import DeclaredMaintenance, KafulSettings
from kaful.data.source import Observation
from kaful.estimation import cloud as clouds
from kaful.estimation.groups import groups
from kaful.phases.p8_estimate.tools.score_synthetic import ks_critical, ks_distance
from tests.unit.test_kaful_connections import LINK, R, _coupled, _observed
from tests.unit.test_kaful_scope import _contract, _machine
from tests.unit.test_kaful_build import _twin

SETTINGS = KafulSettings()
SEAL_K = "shaft_seal.k"


def split(*args: Any) -> Any:
    from kaful.estimation.unread import split as run
    return run(*args)


def _documented(twin: Any) -> Any:
    return next(p for p in twin.profiles if p.name == "documented_maintenance")


def _group(twin: Any) -> Any:
    return next(g for g in groups(twin) if g.observed)


def test_a_binding_no_likelihood_reads_is_split_off_structurally() -> None:
    twin, ps = _coupled()
    twin = _observed(twin, "filter_element_fe")
    s = split(twin, ps, _group(twin))
    assert s.simulated == ("filter_element_fe",)
    assert [r.binding for r in s.reconstructed] == ["shaft_seal"]
    assert s.unread_params == (R, SEAL_K)  # uncertain parameters only; D_limit is a delta


def test_an_observed_target_or_an_unobserved_group_is_simulated_as_before() -> None:
    twin, ps = _coupled()
    both = _observed(twin, "filter_element_fe")
    both = both.model_copy(update={"observations": (*both.observations, both.observations[0].model_copy(
        update={"id": "shaft_seal_on_alarm", "binding": "shaft_seal"}))})
    s = split(both, ps, _group(both))
    assert s.reconstructed == () and s.simulated == ("filter_element_fe", "shaft_seal")
    unobserved = next(g for g in groups(twin) if not g.observed)
    assert split(twin, ps, unobserved).reconstructed == ()


def test_a_target_with_a_clean_stays_simulated_and_says_why() -> None:
    machine = _machine(connection=(LINK,), maintenance=(
        DeclaredMaintenance(component="filter_element_fe", action="replace", on_threshold=True),
        DeclaredMaintenance(component="shaft_seal", action="clean", interval=1000, unit="h")))
    twin, ps, failures = _twin(machine)
    assert failures == ()
    twin = _observed(twin, "filter_element_fe")
    s = split(twin, ps, _group(twin))
    assert s.reconstructed == () and "shaft_seal" in s.simulated
    assert "clean" in s.reasons["shaft_seal"]


def _estimator(twin: Any, ps: Any, method: str, n: int = 2000) -> Any:
    built = ProgPyBackend().build(twin, ps, _contract())
    est = SETTINGS.estimate.model_copy(update={"n_particles": n, "forecast_samples": 1000})
    return ProgPyBackend().estimate(built, ps, _documented(twin), method=method, dt=10.0,  # type: ignore[arg-type]
                                    settings=est, seed=np.random.SeedSequence(7), monitored_events=("alarm",))


def test_after_resampling_on_the_source_the_unread_parameters_are_still_their_prior() -> None:
    """The KS check of the scoring tool, as written: it failed on the boilers in run fd6bf0 (2026-11-16)."""
    twin, ps = _coupled()
    twin = _observed(twin, "filter_element_fe")
    est = _estimator(twin, ps, "particle_filter")
    f = est.filters[0]
    log_k = np.log(f.cloud.params["filter_element_fe.k"])
    for _ in range(3):  # weights on the source's rate only, sharp enough to keep a few dozen ancestors
        f._reweigh(f.cloud.log_w - 0.5 * ((log_k - log_k.mean()) / 0.01) ** 2)
        log_k = np.log(f.cloud.params["filter_element_fe.k"])
    belief = est.belief
    rng = np.random.default_rng(1)
    crit = ks_critical(1000, 20000, 0.01)
    for pid in (R, SEAL_K):
        reference = np.asarray(ps.get(pid).prior.sample(rng, 20000))
        assert ks_distance(np.asarray(belief.particles[pid]), reference) < crit, pid
    assert ks_distance(np.log(belief.particles["filter_element_fe.k"]),
                       np.log(ps.get("filter_element_fe.k").prior.sample(rng, 20000))) > crit  # the read one moved


def test_the_reconstructed_damage_matches_a_forward_simulation_of_the_whole_group() -> None:
    """Exact in distribution: D = k S + sqrt(rho^2 L k S) z given the source's path, S its weighted integrals since
    the last replace (the seal is replaced every 2000 h)."""
    twin, ps = _coupled()
    twin = _observed(twin, "filter_element_fe")
    t_end = 3000.0  # past one scheduled replace of the seal
    est = _estimator(twin, ps, "open_loop")
    est.update(Observation(t=t_end, values={}))
    new = np.asarray(est.belief.particles["shaft_seal_damage"] if "shaft_seal_damage" in est.belief.particles
                     else est.belief.particles[next(b.damage_state for b in twin.bindings if b.id == "shaft_seal")])
    built = ProgPyBackend().build(twin, ps, _contract())
    g = _group(twin)
    from kaful.estimation.groups import carried
    per = carried(twin, ps, g)
    dyn = ProgPyBackend().dynamics(built, _documented(twin), g.bindings, per)
    rng = np.random.default_rng(11)
    full = clouds.draw(ps, per, dyn.bindings, 2000, rng)
    clouds.advance(full, dyn, 0.0, t_end, 10.0, SETTINGS.estimate.process_noise_cv, rng)
    old = full.damage["shaft_seal"]
    assert ks_distance(new, old) < ks_critical(new.size, old.size, 0.01)
    assert np.mean(new) == pytest.approx(np.mean(old), rel=0.05)
