"""The first observation forms (M12): linear and threshold_event. Pure numpy.

- linear: y = y_healthy + (y_limit - y_healthy) * clip(D / D_limit, 0, 1) + N(0, noise_sd^2). The reading moves from
  its healthy value to y_limit, the reading documents state at the damage limit (e.g. "when the pressure drop over
  the separator exceeds 1 bar"), and stays there: a failed part reads at its limit value. y_limit may be above or
  below y_healthy, so the direction comes from the two values, not a sign parameter.
  A channel several bindings act on (one housing accelerometer, two bearings) reads the healthy value plus every
  binding's change: y = y_healthy + sum_i (y_limit_i - y_healthy) * clip(f_i, 0, 1) (`linear_channel`), so y_healthy
  and noise_sd belong to the channel and y_limit to each binding.
- threshold_event: the channel reports the event exactly where D reached D_limit (`crossed`), e.g. the eversys
  W-014 "Predefined water amount has been reached" at each water-filter renewal. No noise: no missed or false
  alarms in M12; the log-likelihood is 0 where y agrees and -inf where it does not (M13 may soften it).
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import numpy as np

from kaful.arrays import FloatArray
from kaful.observation.protocol import ChannelKind, DamageView, ObservationForm, ObservationSlot

LOG_2PI = float(np.log(2.0 * np.pi))


Value = float | FloatArray  # a scalar, or one value per particle (M13)


def linear_channel(y_healthy: Value, terms: Sequence[tuple[DamageView, Value]]) -> FloatArray:
    """Noise-free reading of a linear channel: `terms` are (binding's view, its y_limit), at least one.

    Parameters may be scalars or one value per particle; they broadcast against the damage fractions.
    """
    if not terms:
        raise ValueError("a linear channel needs at least one binding")
    healthy = np.asarray(y_healthy, dtype=np.float64)
    shape = np.broadcast_shapes(np.shape(terms[0][0].fraction), healthy.shape)
    out = np.broadcast_to(healthy, shape).astype(np.float64)
    for view, y_limit in terms:
        delta = np.asarray(y_limit, dtype=np.float64) - healthy
        out = out + delta * np.clip(np.asarray(view.fraction, dtype=np.float64), 0.0, 1.0)
    return out


def gaussian_log_likelihood(y: FloatArray | float, mean: FloatArray, sd: Value) -> FloatArray:
    sd_arr = np.asarray(sd, dtype=np.float64)
    if not np.all(sd_arr > 0.0):
        raise ValueError(f"noise_sd must be > 0, got {sd}")
    z = (np.asarray(y, dtype=np.float64) - mean) / sd_arr
    return np.asarray(-0.5 * z * z - np.log(sd_arr) - 0.5 * LOG_2PI, dtype=np.float64)


class Linear:
    """One binding on a linear channel (a channel with several bindings goes through `linear_channel`)."""

    name: ClassVar[str] = "linear"
    version: ClassVar[str] = "1"
    channel_kind: ClassVar[ChannelKind] = "numeric"
    params: ClassVar[tuple[ObservationSlot, ...]] = (
        ObservationSlot("y_healthy", "observation", "The channel's reading on a healthy machine (D = 0); per channel."),
        ObservationSlot("y_limit", "observation", "The reading when this binding alone is at D = D_limit."),
        ObservationSlot("noise_sd", "noise", "Standard deviation of the Gaussian measurement noise; per channel."),
    )

    def expected(self, view: DamageView, params: Mapping[str, Value]) -> FloatArray:
        return linear_channel(params["y_healthy"], [(view, params["y_limit"])])

    def sample(self, view: DamageView, params: Mapping[str, Value], rng: np.random.Generator) -> FloatArray:
        mean = self.expected(view, params)
        sd = np.asarray(params["noise_sd"], dtype=np.float64)
        gaussian_log_likelihood(mean, mean, sd)  # refuses a non-positive sd before drawing
        return mean + sd * rng.standard_normal(mean.shape)

    def log_likelihood(self, y: FloatArray, view: DamageView, params: Mapping[str, Value]) -> FloatArray:
        return gaussian_log_likelihood(y, self.expected(view, params), params["noise_sd"])


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


FORMS: dict[str, type[ObservationForm]] = {cls.name: cls for cls in (Linear, ThresholdEvent)}
