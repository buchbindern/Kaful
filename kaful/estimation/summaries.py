"""Summaries of a belief for forecasts and traces: health quantiles, a linear channel's expected reading, and
parameter quantiles. Quantiles are 5/50/95% over the belief's equally weighted samples."""

from collections.abc import Mapping

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TemplateBinding, TwinDefinition
from kaful.backends.base import StateBelief
from kaful.data.source import Observation
from kaful.estimation.cloud import Cloud
from kaful.estimation.groups import groups
from kaful.estimation.likelihood import linear_mean

QUANTILES = {"p05": 0.05, "p50": 0.50, "p95": 0.95}


def quantiles(x: FloatArray) -> dict[str, float]:
    return {k: float(np.quantile(x, q, method="inverted_cdf")) for k, q in QUANTILES.items()}


def medians(ps: ParamSet) -> dict[str, float]:
    return {x.param_id: (x.prior.value if isinstance(x.prior, Delta) else x.prior.quantile(0.5))
            for x in sorted(ps.parameters, key=lambda x: x.param_id)}


def _value(belief: StateBelief, fixed: Mapping[str, float], pid: str) -> FloatArray:
    if pid in belief.particles:
        return np.asarray(belief.particles[pid], dtype=np.float64)
    return np.full(belief.weights.size, fixed[pid])


def health(belief: StateBelief, binding: TemplateBinding, fixed: Mapping[str, float]) -> FloatArray:
    """1 - D / D_limit per sample, clipped to [0, 1]."""
    d = np.asarray(belief.particles[binding.damage_state], dtype=np.float64)
    return np.asarray(np.clip(1.0 - d / _value(belief, fixed, binding.threshold_param), 0.0, 1.0), dtype=np.float64)


def channel_readings(twin: TwinDefinition, belief: StateBelief, fixed: Mapping[str, float]) -> dict[str, FloatArray]:
    """Each linear channel's noise-free reading per sample, for channels whose bindings are all in the belief."""
    by_id = {b.id: b for b in twin.bindings}
    out = {}
    for g in groups(twin):
        if not all(by_id[b].damage_state in belief.particles for b in g.bindings):
            continue
        n = belief.weights.size
        params = {p: _value(belief, fixed, p) for o in g.observations for p in o.params.values()}
        params |= {by_id[b].threshold_param: _value(belief, fixed, by_id[b].threshold_param) for b in g.bindings}
        cloud = Cloud(damage={b: np.asarray(belief.particles[by_id[b].damage_state]) for b in g.bindings},
                      renewals={}, params=params, log_w=np.zeros(n))
        for ch in g.channels:
            if any(o.channel == ch and o.form == "linear" for o in g.observations):
                out[ch] = linear_mean(Observation(t=belief.t, values={}), ch, g, by_id, cloud, {})[0]
    return dict(sorted(out.items()))
