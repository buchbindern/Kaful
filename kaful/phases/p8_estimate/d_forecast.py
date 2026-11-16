"""Step d: the forecasts issued at each issue time, and what they say.

Per issue time, the observed groups are forecast from the estimator's belief, and the unobserved groups once from
the open-loop belief, shared by every estimator (identical by construction: same particles, same seeds). Per event:
counts (hits, censored, already triggered), P(event before horizon) and 5/50/95% of the next time the part reaches
its limit (kaful.estimation.predict). Per carried parameter: prior and posterior 5/50/95% (k: the reported
posterior; the others: nuisance). Per binding: health quantiles. Every sample goes to a CSV sidecar.
"""

import csv
import io
from collections.abc import Mapping, Sequence

import numpy as np

from kaful.artifacts.forecast import EventForecast, HealthSummary, OperatingProfile, ParamSummary
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends.base import BuiltTwin, StateBelief, TwinBackend
from kaful.config import EstimateSettings
from kaful.estimation.predict import Prediction, summarize
from kaful.estimation.summaries import QUANTILES, health, quantiles


def forecast_issue(backend: TwinBackend, built: BuiltTwin, belief: StateBelief, profile: OperatingProfile, dt: float,
                   horizon: float, settings: EstimateSettings, seed: np.random.SeedSequence,
                   bindings: Sequence[str]) -> Prediction | None:
    if not bindings:
        return None
    return backend.predict(built, belief, profile, dt=dt, horizon=horizon, settings=settings, seed=seed,
                           bindings=bindings)


def merge(parts: Sequence[Prediction | None]) -> Prediction:
    found = [p for p in parts if p is not None]
    events = tuple(sorted((e for p in found for e in p.events), key=lambda e: e.event))
    params = dict(sorted(((k, v) for p in found for k, v in p.params.items())))
    return Prediction(t_issue=found[0].t_issue, horizon=found[0].horizon, events=events, params=params)


def merge_beliefs(own: StateBelief, shared: StateBelief | None) -> StateBelief:
    """The estimator's belief plus the shared open-loop belief's other states (independent groups pair by index)."""
    if shared is None:
        return own
    particles = {**{k: v for k, v in shared.particles.items() if k not in own.particles}, **own.particles}
    return StateBelief(t=own.t, particles=dict(sorted(particles.items())), weights=own.weights)


def summarize_issue(pred: Prediction, belief: StateBelief, twin: TwinDefinition, ps: ParamSet,
                    fixed: Mapping[str, float]) -> tuple[list[EventForecast], list[ParamSummary], list[HealthSummary]]:
    events = []
    for e in pred.events:
        s = summarize(e)
        events.append(EventForecast(event=e.event, binding=e.binding, n_samples=s.n_samples, n_hits=s.n_hits,
                                    n_censored=s.n_censored, n_already_triggered=s.n_triggered,
                                    p_event_before_horizon=s.p_event_before_horizon,
                                    percentiles=s.percentiles))
    rates = {b.params["k"] for b in twin.bindings if "k" in b.params}
    params = [ParamSummary(param_id=pid, role="rate" if pid in rates else "nuisance",
                           prior={k: ps.get(pid).prior.quantile(q) for k, q in QUANTILES.items()},
                           posterior=quantiles(values))
              for pid, values in sorted(pred.params.items())]
    states = [HealthSummary(binding=b.id, quantiles=quantiles(health(belief, b, fixed)))
              for b in sorted(twin.bindings, key=lambda b: b.id) if b.damage_state in belief.particles]
    return events, params, states


def _cell(x: float) -> str:
    return "" if np.isnan(x) else repr(float(x))


def samples_csv(pred: Prediction) -> str:
    """One row per sample: each event's time of event (empty: censored) and whether it was already triggered,
    then each carried parameter."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["sample", *(f"{e.event}.toe" for e in pred.events), *(f"{e.event}.triggered" for e in pred.events),
                     *pred.params])
    n = pred.events[0].toe.size if pred.events else 0
    for i in range(n):
        writer.writerow([i, *(_cell(e.toe[i]) for e in pred.events),
                         *("true" if e.triggered[i] else "false" for e in pred.events),
                         *(repr(float(v[i])) for v in pred.params.values())])
    return buf.getvalue()
