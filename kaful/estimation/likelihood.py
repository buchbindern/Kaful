"""log p(row | particle) for one group: what the particle filter weighs particles by.

- A linear channel reads y_healthy plus each of its bindings' change (kaful.observation.forms.linear_channel), with
  Gaussian noise; every parameter is the particle's own value when carried.
- A channel with no value in the row contributes nothing.
- Threshold-event channels: kaful.estimation.events (first-passage likelihood).
"""

from collections.abc import Mapping

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.twin_definition import TemplateBinding
from kaful.data.source import Observation
from kaful.estimation.cloud import Cloud, value
from kaful.estimation.groups import Group
from kaful.observation.forms import gaussian_log_likelihood, linear_channel
from kaful.observation.protocol import DamageView


def linear_mean(obs: Observation, ch: str, group: Group, bindings: Mapping[str, TemplateBinding], cloud: Cloud,
                fixed: Mapping[str, float]) -> tuple[FloatArray, FloatArray | float]:
    """The noise-free reading of linear channel `ch` per particle, and its noise sd (per particle or fixed)."""
    on = [o for o in group.observations if o.channel == ch and o.form == "linear"]
    no_crossing = np.zeros(cloud.n, dtype=bool)
    terms = []
    for o in on:
        b = bindings[o.binding]
        fraction = cloud.damage[b.id] / value(cloud, fixed, b.threshold_param)
        terms.append((DamageView(np.asarray(fraction, dtype=np.float64), no_crossing),
                      value(cloud, fixed, o.params["y_limit"])))
    mean = linear_channel(value(cloud, fixed, on[0].params["y_healthy"]), terms)
    return mean, value(cloud, fixed, on[0].params["noise_sd"])


def linear(obs: Observation, group: Group, bindings: Mapping[str, TemplateBinding], cloud: Cloud,
           fixed: Mapping[str, float]) -> tuple[FloatArray, tuple[str, ...]]:
    """Summed log-likelihood over the group's linear channels that report in `obs`, and those channels."""
    total = np.zeros(cloud.n)
    used = []
    for ch in group.channels:
        y = obs.values.get(ch)
        if not any(o.channel == ch and o.form == "linear" for o in group.observations):
            continue
        if y is None or isinstance(y, (bool, str)):
            continue
        mean, sd = linear_mean(obs, ch, group, bindings, cloud, fixed)
        total = total + gaussian_log_likelihood(float(y), mean, sd)
        used.append(ch)
    return total, tuple(used)
