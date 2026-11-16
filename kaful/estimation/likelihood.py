"""log p(row | particle) for one group: what the particle filter weighs particles by.

- A linear channel reads y_healthy plus each of its bindings' change (kaful.observation.forms.linear_channel), with
  Gaussian noise; every parameter is the particle's own value when carried.
- A channel with no value in the row contributes nothing.
- Threshold-event channels: checkpoint 2 (first-passage likelihood).
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


def linear(obs: Observation, group: Group, bindings: Mapping[str, TemplateBinding], cloud: Cloud,
           fixed: Mapping[str, float]) -> tuple[FloatArray, tuple[str, ...]]:
    """Summed log-likelihood over the group's linear channels that report in `obs`, and those channels."""
    total = np.zeros(cloud.n)
    used = []
    for ch in group.channels:
        on = [o for o in group.observations if o.channel == ch and o.form == "linear"]
        y = obs.values.get(ch)
        if not on or y is None or isinstance(y, (bool, str)):
            continue
        no_crossing = np.zeros(cloud.n, dtype=bool)
        terms = []
        for o in on:
            b = bindings[o.binding]
            fraction = cloud.damage[b.id] / value(cloud, fixed, b.threshold_param)
            terms.append((DamageView(np.asarray(fraction, dtype=np.float64), no_crossing),
                          value(cloud, fixed, o.params["y_limit"])))
        mean = linear_channel(value(cloud, fixed, on[0].params["y_healthy"]), terms)
        total = total + gaussian_log_likelihood(float(y), mean, value(cloud, fixed, on[0].params["noise_sd"]))
        used.append(ch)
    return total, tuple(used)
