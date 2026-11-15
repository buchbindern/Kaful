"""The first observation forms (M12): linear and threshold_event. Pure numpy.

- linear: y = y_healthy + delta * clip(D / D_limit, 0, 1) + N(0, noise_sd^2). The reading moves from its healthy
  value to y_healthy + delta at the damage limit and stays there (a failed part reads at its limit value). Delta
  carries the sign: clogging raises a pressure drop (delta > 0) and lowers the outlet pressure (delta < 0).
- threshold_event: the channel reports the event exactly where D reached D_limit (`crossed`), e.g. the eversys
  W-014 "Predefined water amount has been reached" at each water-filter renewal. No noise: no missed or false
  alarms in M12; the log-likelihood is 0 where y agrees and -inf where it does not (M13 may soften it).
"""

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from kaful.arrays import FloatArray
from kaful.observation.protocol import ChannelKind, DamageView, ObservationForm, ObservationSlot

LOG_2PI = float(np.log(2.0 * np.pi))


class Linear:
    name: ClassVar[str] = "linear"
    version: ClassVar[str] = "1"
    channel_kind: ClassVar[ChannelKind] = "numeric"
    params: ClassVar[tuple[ObservationSlot, ...]] = (
        ObservationSlot("y_healthy", "observation", "The reading of a healthy part (D = 0)."),
        ObservationSlot("delta", "observation", "Change of the reading from D = 0 to D = D_limit (signed)."),
        ObservationSlot("noise_sd", "noise", "Standard deviation of the Gaussian measurement noise."),
    )

    def expected(self, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        fraction = np.clip(np.asarray(view.fraction, dtype=np.float64), 0.0, 1.0)
        return np.asarray(params["y_healthy"] + params["delta"] * fraction, dtype=np.float64)

    def sample(self, view: DamageView, params: Mapping[str, float], rng: np.random.Generator) -> FloatArray:
        mean = self.expected(view, params)
        return mean + _sd(params) * rng.standard_normal(mean.shape)

    def log_likelihood(self, y: FloatArray, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        sd = _sd(params)
        z = (np.asarray(y, dtype=np.float64) - self.expected(view, params)) / sd
        return np.asarray(-0.5 * z * z - np.log(sd) - 0.5 * LOG_2PI, dtype=np.float64)


class ThresholdEvent:
    name: ClassVar[str] = "threshold_event"
    version: ClassVar[str] = "1"
    channel_kind: ClassVar[ChannelKind] = "event"
    params: ClassVar[tuple[ObservationSlot, ...]] = ()

    def expected(self, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        return np.asarray(view.crossed, dtype=np.float64)

    def sample(self, view: DamageView, params: Mapping[str, float], rng: np.random.Generator) -> FloatArray:
        return self.expected(view, params)

    def log_likelihood(self, y: FloatArray, view: DamageView, params: Mapping[str, float]) -> FloatArray:
        agree = (np.asarray(y, dtype=np.float64) > 0.5) == np.asarray(view.crossed, dtype=bool)
        return np.where(agree, 0.0, -np.inf).astype(np.float64)


def _sd(params: Mapping[str, float]) -> float:
    sd = float(params["noise_sd"])
    if not sd > 0.0:
        raise ValueError(f"noise_sd must be > 0, got {sd}")
    return sd


FORMS: dict[str, type[ObservationForm]] = {cls.name: cls for cls in (Linear, ThresholdEvent)}
