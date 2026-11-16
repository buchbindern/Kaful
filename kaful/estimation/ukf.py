"""The unscented Kalman filter, for comparison with the particle filter (M13; numpy, approved 2026-11-16).

Per observed group with a linear channel, the state is each binding's damage D plus every carried parameter on an
unconstrained scale (kaful.estimation.rejuvenation transforms), so a Gaussian over it keeps parameters in their
support. Between rows, the sigma points move through the backend's step (drift and scheduled maintenance, which
moves all of them alike), and each binding's process noise, rho^2 * D_limit * dD/dt per time unit (cloud.py),
is added to its D variance at the mean. A row updates on the group's linear channels that report; threshold-event
channels are not used (their likelihood is not Gaussian), so a group seen only through alarms, and a group with a
binding renewed at its limit (a discontinuity a Gaussian cannot follow), run open-loop under the UKF instead, which
the estimate records. Sigma points: Merwe's scaled set (alpha, beta, kappa in settings).

An exported belief draws settings.forecast_samples points from the Gaussian (D clipped at 0) from the group's own
seed stream.
"""

from collections.abc import Mapping

import numpy as np

from kaful.arrays import FloatArray
from kaful.artifacts.twin_definition import TemplateBinding
from kaful.config import EstimateSettings
from kaful.data.source import Observation
from kaful.estimation import likelihood
from kaful.estimation.cloud import Cloud
from kaful.estimation.dynamics import Dynamics
from kaful.estimation.filter import EXPORT, Diagnostics, _rng
from kaful.estimation.groups import Group
from kaful.estimation.rejuvenation import Transform


def sigma_weights(n: int, alpha: float, beta: float, kappa: float) -> tuple[float, FloatArray, FloatArray]:
    """(lambda, mean weights, covariance weights) of Merwe's scaled sigma points for dimension n."""
    lam = alpha * alpha * (n + kappa) - n
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha * alpha + beta)
    return lam, wm, wc


def _sqrt(cov: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    return np.asarray(vectors * np.sqrt(np.clip(values, 0.0, None)), dtype=np.float64)


def ukf_ready(group: Group, dyn: Dynamics) -> str | None:
    """Why the UKF cannot run this group (None: it can)."""
    if not any(o.form == "linear" for o in group.observations):
        return "no linear channel (alarm channels are not Gaussian)"
    if any(b.renews_at_limit for b in dyn.bindings):
        return "a binding is renewed at its limit (a discontinuity)"
    return None


class UKFGroup:
    """One observed group under the UKF; the same interface as GroupFilter."""

    def __init__(self, group: Group, dyn: Dynamics, prior: Cloud, bindings: Mapping[str, TemplateBinding],
                 transforms: Mapping[str, Transform], settings: EstimateSettings, dt: float,
                 seed: np.random.SeedSequence, index: int) -> None:
        self.group, self.dyn, self.bindings, self.transforms = group, dyn, bindings, transforms
        self.settings, self.dt, self.seed, self.index = settings, dt, seed, index
        self.damage_ids = tuple(b.id for b in dyn.bindings)
        self.param_ids = tuple(sorted(prior.params))
        z = np.stack([prior.damage[b] for b in self.damage_ids]
                     + [transforms[p].forward(prior.params[p]) for p in self.param_ids])
        self.mean = z.mean(axis=1)
        self.cov = np.cov(z) + np.diag(np.full(z.shape[0], 1e-12))
        self.lam, self.wm, self.wc = sigma_weights(z.shape[0], settings.ukf_alpha, settings.ukf_beta,
                                                   settings.ukf_kappa)
        self.t = 0.0
        self.rows = 0
        self.diagnostics = Diagnostics()
        self.limits = {b.id: b.limit_param for b in dyn.bindings}

    def _points(self) -> FloatArray:
        n = self.mean.size
        root = _sqrt((n + self.lam) * self.cov)
        return np.concatenate([self.mean[:, None], self.mean[:, None] + root, self.mean[:, None] - root], axis=1)

    def _cloud(self, points: FloatArray) -> Cloud:
        k = len(self.damage_ids)
        return Cloud(damage={b: np.maximum(points[i], 0.0) for i, b in enumerate(self.damage_ids)}, renewals={},
                     params={p: self.transforms[p].backward(points[k + j]) for j, p in enumerate(self.param_ids)},
                     log_w=np.zeros(points.shape[1]))

    def _limit(self, cloud: Cloud, b: str) -> FloatArray:
        pid = self.limits[b]
        return np.asarray(cloud.params[pid] if pid in cloud.params else self.dyn.fixed[pid], dtype=np.float64)

    def _predict(self, t1: float) -> None:
        points = self._points()
        cloud = self._cloud(points)
        t = self.t
        while t < t1:
            h = min(self.dt, t1 - t)
            cloud.damage = {b: np.maximum(d, 0.0) for b, d in self.dyn.step(cloud.damage, cloud.params, t, h)[0].items()}
            t = t1 if h >= t1 - t else t + h
        for i, b in enumerate(self.damage_ids):
            points[i] = cloud.damage[b]
        mean = points @ self.wm
        centred = points - mean[:, None]
        cov = (centred * self.wc) @ centred.T
        at_mean = self._cloud(self.mean[:, None])
        rates = self.dyn.rates(at_mean.params, self.t)
        rho = self.settings.process_noise_cv
        for i, b in enumerate(self.damage_ids):
            cov[i, i] += rho * rho * float(self._limit(at_mean, b).ravel()[0]) * max(float(rates[b][0]), 0.0) * (t1 - self.t)
        self.mean, self.cov = mean, 0.5 * (cov + cov.T)

    def update(self, obs: Observation) -> None:
        if obs.t < self.t:
            raise ValueError(f"observation at t={obs.t} is before the filter's time {self.t}")
        self.rows += 1
        if obs.t > self.t:
            self._predict(obs.t)
        self.t = obs.t
        points = self._points()
        cloud = self._cloud(points)
        channels = [ch for ch in self.group.channels
                    if any(o.channel == ch and o.form == "linear" for o in self.group.observations)
                    and isinstance(obs.values.get(ch), (int, float)) and not isinstance(obs.values.get(ch), bool)]
        if not channels:
            return
        predicted, noise, y = [], [], []
        for ch in channels:
            one = Observation(t=obs.t, values={ch: obs.values[ch]})
            mean, sd = likelihood.linear_mean(one, ch, self.group, self.bindings, cloud, self.dyn.fixed)
            predicted.append(mean)
            noise.append(float(np.asarray(sd, dtype=np.float64).ravel()[0]))
            y.append(float(obs.values[ch]))  # type: ignore[arg-type]
        zs = np.stack(predicted)  # (m, 2n+1)
        z_mean = zs @ self.wm
        dz = zs - z_mean[:, None]
        dx = points - self.mean[:, None]
        s = (dz * self.wc) @ dz.T + np.diag(np.square(noise))
        cross = (dx * self.wc) @ dz.T
        gain = cross @ np.linalg.inv(s)
        self.mean = self.mean + gain @ (np.asarray(y) - z_mean)
        self.cov = self.cov - gain @ s @ gain.T
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.diagnostics.updates += 1
        for ch in channels:
            self.diagnostics.channels_used[ch] = self.diagnostics.channels_used.get(ch, 0) + 1

    def export(self, m: int) -> Cloud:
        rng = _rng(self.seed, self.index, EXPORT, self.rows)
        z = self.mean[:, None] + _sqrt(self.cov) @ rng.standard_normal((self.mean.size, m))
        return self._cloud(z)
