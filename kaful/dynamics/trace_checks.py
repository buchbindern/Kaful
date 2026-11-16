"""The dynamic checks on one binding's trace from one simulation. Shared by verify (c_check_dynamics) and synthesize
(c_check_units: every synthetic unit's truth, M12).

`tol` is verify.damage_tolerance, a fraction of D_limit. Health is checked on raw D, not only on the output:
the backend clips health to [0, 1], so a negative D (health above 1) would not show in the output (PLAN M11 b).
- starts_healthy: D = 0, health = event_state = 1, no event and no renewal at t = 0;
- finite_bounded: D, health and event_state finite; D >= 0; health in [0, 1] and equal to clip(1 - D/D_limit);
  a renewal count is a whole number >= 0;
- monotone: D falls only at a scheduled reset or a renewal;
- event_agreement: without a renewal rule, threshold_met holds exactly when event_state is 0. With one (a part
  replaced when it reaches its limit, e.g. the eversys water filter), D returns to 0 at each renewal while the
  event stays met, so: the event first holds at the first renewal and holds after it, and one step before it
  event_state is within one step's wear of 0 (PLAN M11 c, Natalie 2026-11-13);
- maintenance_resets: a replace leaves D at 0, a clean or descale leaves D * (1 - recovery), each within one
  step's growth; a renewal adds exactly 1 to the count, at a step where D reached D_limit, and leaves D at 0;
- life_consistency: the first event is within `band` of the median life (too fast and too slow both fail).
"""

from dataclasses import dataclass, field

import numpy as np

from kaful.arrays import BoolArray
from kaful.dynamics.traces import BindingTrace


@dataclass(frozen=True)
class Finding:
    ok: bool
    message: str = ""
    measured: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=dict)


def _first(mask: BoolArray) -> int | None:
    return int(np.argmax(mask)) if mask.any() else None


def _verdict(problems: list[str], measured: dict[str, float]) -> Finding:
    return Finding(ok=not problems, message="; ".join(problems), measured=measured)


def starts_healthy(tr: BindingTrace, tol: float) -> Finding:
    problems = []
    if abs(tr.damage[0]) > tol * tr.limit:
        problems.append(f"D starts at {tr.damage[0]:.6g}, not 0")
    if abs(tr.health[0] - 1.0) > tol or abs(tr.event_state[0] - 1.0) > tol:
        problems.append(f"health starts at {tr.health[0]:.6g} (event_state {tr.event_state[0]:.6g}), not 1.0")
    if tr.met[0]:
        problems.append("the event is already triggered at t=0")
    if tr.renewals is not None and tr.renewals[0] != 0.0:
        problems.append(f"the renewal count starts at {tr.renewals[0]:g}, not 0")
    return _verdict(problems, {"damage_t0": float(tr.damage[0]), "health_t0": float(tr.health[0])})


def finite_bounded(tr: BindingTrace, tol: float) -> Finding:
    series = {"D": tr.damage, "health": tr.health, "event_state": tr.event_state}
    if tr.renewals is not None:
        series["renewals"] = tr.renewals
    for name, values in series.items():
        bad = _first(~np.isfinite(values))
        if bad is not None:
            return Finding(ok=False, message=f"{name} is not finite from t={tr.times[bad]:.6g}")
    problems = []
    low = int(np.argmin(tr.damage))
    if tr.damage[low] < -tol * tr.limit:
        problems.append(f"D falls to {tr.damage[low]:.6g} < 0 at t={tr.times[low]:.6g}")
    if tr.health.min() < 0.0 or tr.health.max() > 1.0:
        problems.append(f"health leaves [0, 1]: {tr.health.min():.6g} to {tr.health.max():.6g}")
    off = np.abs(tr.health - np.clip(1.0 - tr.damage / tr.limit, 0.0, 1.0))
    if off.max() > tol:
        problems.append(f"health is not 1 - D/D_limit: off by {off.max():.3g} at t={tr.times[int(np.argmax(off))]:.6g}")
    if tr.renewals is not None and (tr.renewals.min() < 0 or np.any(tr.renewals != np.round(tr.renewals))):
        problems.append("the renewal count is not a whole number >= 0")
    return _verdict(problems, {"damage_min": float(tr.damage.min() / tr.limit), "health_min": float(tr.health.min()),
                               "health_max": float(tr.health.max())})


def monotone(tr: BindingTrace, tol: float) -> Finding:
    allowed = tr.reset_steps
    drops = [int(i) + 1 for i in np.flatnonzero(np.diff(tr.damage) < -tol * tr.limit) if int(i) + 1 not in allowed]
    problems = []
    if drops:
        i = drops[0]
        problems.append(f"D falls by {tr.damage[i - 1] - tr.damage[i]:.6g} at t={tr.times[i]:.6g} with no maintenance "
                        f"({len(drops)} such step{'s' if len(drops) > 1 else ''})")
    return _verdict(problems, {"unexplained_drops": float(len(drops))})


def _agreement_without_renewal(tr: BindingTrace, tol: float, upto: int) -> list[str]:
    met, es = tr.met[:upto], tr.event_state[:upto]
    bad = _first((met & (es > tol)) | (~met & (es <= 0.0)))
    if bad is None:
        return []
    return [f"threshold_met is {bool(met[bad])} while event_state is {es[bad]:.6g} at t={tr.times[bad]:.6g}"]


def event_agreement(tr: BindingTrace, tol: float) -> Finding:
    if tr.renewals is None:
        return _verdict(_agreement_without_renewal(tr, tol, tr.met.size), {})
    first_met, first_renewal = _first(tr.met), _first(tr.renewals >= 1.0)
    if first_met is None and first_renewal is None:
        return _verdict(_agreement_without_renewal(tr, tol, tr.met.size), {})
    if first_met != first_renewal or first_met is None:
        when = {i: f"t={tr.times[i]:.6g}" if i is not None else "never" for i in (first_met, first_renewal)}
        return Finding(ok=False, message=f"the event first holds at {when[first_met]} but the first renewal is at "
                                         f"{when[first_renewal]}")
    problems = _agreement_without_renewal(tr, tol, first_met)
    if not tr.met[first_met:].all():
        problems.append("the event stops holding after the first renewal")
    before = float(tr.event_state[first_met - 1]) if first_met > 0 else 1.0
    step_wear = tr.increment() / tr.limit
    if before > step_wear * (1.0 + 1e-6) + tol:
        problems.append(f"event_state is {before:.6g} one step before the first renewal (one step's wear is "
                        f"{step_wear:.3g})")
    return _verdict(problems, {"event_state_before_renewal": before, "step_wear": step_wear})


def maintenance_resets(tr: BindingTrace, tol: float) -> Finding | None:
    """None when the binding has no scheduled reset in this run and no renewal rule."""
    if not tr.resets and tr.renewals is None:
        return None
    slack, d, problems = tr.increment() + tol * tr.limit, tr.damage, []
    for r in tr.resets:
        i = min(r.window, key=lambda j: d[j] - d[j - 1])  # the step where D fell most
        before = float(d[i - 1])
        expected = 0.0 if r.rule.action == "replace" else before * (1.0 - tr.recovery[r.rule.id])
        if abs(d[i] - expected) > slack and before > slack:
            problems.append(f"{r.rule.id} at t={r.t:.6g}: D is {d[i]:.6g} after the reset, expected {expected:.6g}")
    renewed = 0
    if tr.renewals is not None:
        for j in np.flatnonzero(np.diff(tr.renewals)):
            i, added = int(j) + 1, float(tr.renewals[int(j) + 1] - tr.renewals[int(j)])
            renewed += 1
            if added != 1.0 or d[i] > slack or d[i - 1] < tr.limit - slack:
                problems.append(f"renewal at t={tr.times[i]:.6g}: count +{added:g}, D {d[i - 1]:.6g} -> {d[i]:.6g} "
                                f"(D_limit {tr.limit:.6g})")
    return _verdict(problems[:3] + ([f"and {len(problems) - 3} more"] if len(problems) > 3 else []),
                    {"resets": float(len(tr.resets)), "renewals": float(renewed)})


def life_consistency(tr: BindingTrace, median: float, band: float) -> Finding:
    first = _first(tr.met)
    limits = {"band": band}
    if first is None:
        return Finding(ok=False, message=f"no event before t={tr.times[-1]:.6g} (median life {median:.6g}): too slow",
                       measured={"median_life": median}, limits=limits)
    t, ratio = float(tr.times[first]), float(tr.times[first]) / median
    ok = abs(ratio - 1.0) <= band
    message = "" if ok else (f"the event fires at {t:.6g}, {100 * (ratio - 1.0):+.2f}% from the median life "
                             f"{median:.6g} ({'too fast' if ratio < 1 else 'too slow'}; band ±{100 * band:g}%)")
    return Finding(ok=ok, message=message, measured={"event_time": t, "median_life": median, "ratio": ratio},
                   limits=limits)
