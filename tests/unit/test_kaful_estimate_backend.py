"""The ProgPy adapter's estimation side (M13) on the small machine of the synthesize tests: the vectorized step
matches the scalar simulation, the unobserved group is open-loop under every estimator, and runs are reproducible
under any hash seed."""

import hashlib
import os
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from kaful.backends.progpy.backend import ProgPyBackend, ProgPyTwin
from kaful.config import EstimateSettings
from kaful.data.source import Observation
from kaful.estimation.groups import groups
from tests.unit.test_kaful_synthesize import _twin

BACKEND = ProgPyBackend()
DT = 10.0


def _built() -> tuple[Any, Any, ProgPyTwin, Any]:
    twin, ps, contract = _twin()
    built = BACKEND.build(twin, ps, contract)
    profile = next(p for p in twin.profiles if p.name == "documented_maintenance")
    return twin, ps, built, profile


def _rows(n: int = 30) -> list[Observation]:
    return [Observation(t=float(200 * i), values={"hours": 200 * i, "leak_flow": 1.0 + 0.004 * 200 * i / 10})
            for i in range(n)]


def test_the_vectorized_step_matches_the_scalar_simulation_particle_by_particle() -> None:
    twin, ps, built, profile = _built()
    seal = next(b for b in twin.bindings if b.component == "shaft_seal")
    k_id = seal.params["k"]
    ks = np.array([0.5, 1.0, 2.0]) * ps.get(k_id).prior.quantile(0.5)
    dyn = BACKEND.dynamics(built, profile, [seal.id], [k_id])
    damage, t = {seal.id: np.zeros(3)}, 0.0
    steps = 400
    for _ in range(steps):
        damage = dyn.step(damage, {k_id: ks}, t, DT)[0]
        t += DT
    compared = 0
    for i, k in enumerate(ks):
        sim = BACKEND.simulate(built, profile, {k_id: float(k)}, dt=DT, horizon=steps * DT, seed=0)
        renewals = sim.states.get(f"{seal.id}_renewals")
        if renewals is not None and renewals[-1] > 0:
            continue  # the scalar run renews at D_limit; the estimators renew themselves (cloud.advance)
        assert damage[seal.id][i] == pytest.approx(sim.states[seal.damage_state][-1], rel=1e-12)
        compared += 1
    assert compared >= 2


def test_the_unobserved_group_is_open_loop_under_the_particle_filter() -> None:
    twin, ps, built, profile = _built()
    est = EstimateSettings(n_particles=200)
    beliefs = {}
    for method in ("particle_filter", "open_loop"):
        e = BACKEND.estimate(built, ps, profile, method=method, dt=DT, settings=est,
                             seed=np.random.SeedSequence(est.seed, spawn_key=(0,)))
        for row in _rows():
            e.update(row)
        beliefs[method] = e.belief
    unobserved = next(g for g in groups(twin) if not g.observed)
    for bid in unobserved.bindings:
        state = next(b.damage_state for b in twin.bindings if b.id == bid)
        assert np.array_equal(beliefs["particle_filter"].particles[state], beliefs["open_loop"].particles[state])
    observed = next(g for g in groups(twin) if g.observed)
    seal_state = next(b.damage_state for b in twin.bindings if b.id == observed.bindings[0])
    assert not np.array_equal(beliefs["particle_filter"].particles[seal_state],
                              beliefs["open_loop"].particles[seal_state])


def _digest() -> str:
    twin, ps, built, profile = _built()
    est = EstimateSettings(n_particles=100)
    e = BACKEND.estimate(built, ps, profile, method="particle_filter", dt=DT, settings=est,
                         seed=np.random.SeedSequence(est.seed, spawn_key=(0,)))
    for row in _rows():
        e.update(row)
    p = BACKEND.predict(built, e.belief, profile, dt=DT, horizon=20000.0, settings=est,
                        seed=np.random.SeedSequence(est.seed, spawn_key=(0, 9)))
    h = hashlib.sha256()
    for name, v in sorted(e.belief.particles.items()):
        h.update(name.encode() + np.ascontiguousarray(v).tobytes())
    for ev in p.events:
        h.update(ev.event.encode() + np.ascontiguousarray(ev.toe).tobytes())
    return h.hexdigest()


def test_estimate_and_predict_do_not_depend_on_hash_order() -> None:
    code = "from tests.unit.test_kaful_estimate_backend import _digest; print(_digest())"
    run = [sys.executable, "-B", "-W", "ignore", "-c", code]
    outs = {subprocess.run(run, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("0", "1")}
    assert len(outs) == 1 and len(next(iter(outs)).strip()) == 64
