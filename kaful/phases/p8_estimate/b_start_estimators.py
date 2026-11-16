"""Step b: start each estimator for one source from the priors.

Open-loop runs every group. The particle filter and the UKF run only the observed groups: an unobserved group's
posterior is its prior moved forward, identical under every estimator by construction (same seed streams), so it is
filtered and forecast once, in the open-loop run, and shared (approved 2026-11-16). Seeds: the source's estimators
draw from SeedSequence(seed, spawn_key=(source,)), so particle filter and open-loop start from the same particles;
forecasts from SeedSequence(seed, spawn_key=(source, PREDICT, issue)).
"""

from collections.abc import Sequence

import numpy as np

from kaful.artifacts.forecast import OperatingProfile
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import BuiltTwin, Estimator, TwinBackend
from kaful.config import EstimateSettings, EstimatorName
from kaful.estimation.groups import groups

PREDICT = 999_999  # spawn-key marker for forecasts (group indices never reach it)


def observed_bindings(twin: TwinDefinition) -> tuple[str, ...]:
    return tuple(sorted(b for g in groups(twin) if g.observed for b in g.bindings))


def unobserved_bindings(twin: TwinDefinition) -> tuple[str, ...]:
    return tuple(sorted(b for g in groups(twin) if not g.observed for b in g.bindings))


def source_seed(settings: EstimateSettings, spawn_key: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(settings.seed, spawn_key=(spawn_key,))


def predict_seed(settings: EstimateSettings, spawn_key: int, issue: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(settings.seed, spawn_key=(spawn_key, PREDICT, issue))


def start(backend: TwinBackend, built: BuiltTwin, ps: ParamSet, profile: OperatingProfile, method: EstimatorName,
          dt: float, settings: EstimateSettings, spawn_key: int, monitored: Sequence[str]) -> Estimator:
    twin = built.definition
    only = None if method == "open_loop" else observed_bindings(twin)
    return backend.estimate(built, ps, profile, method=method, dt=dt, settings=settings,
                            seed=source_seed(settings, spawn_key), monitored_events=monitored, bindings=only)


def ukf_skip_reason(estimator: Estimator) -> str | None:
    """Why the UKF ran nowhere (every observed group fell back to open-loop), or None."""
    notes = [f.diagnostics.note for f in getattr(estimator, "filters", ()) if f.group.observed]
    if notes and all(notes):
        return "; ".join(sorted(set(notes)))
    return None
