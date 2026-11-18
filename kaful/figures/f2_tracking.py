"""F2: what each estimator believes over time, for one unit chosen by rule (the same rule and quantity as F3).

Top panel: the channel's observations (as a FileSource yields them) and each estimator's expected reading (mean).
One panel per binding below: each estimator's health (mean and 5-95%) and the true health 1 - D / D_limit.
Dotted verticals mark the forecast issue times.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from kaful.data.file_source import FileSource
from kaful.figures.drawn import Context, Drawn, pretty_estimator
from kaful.figures.forecasts import read_trace
from kaful.figures.select import choose_unit
from kaful.figures.spec import Style, TrackingSpec
from kaful.figures.style import Cell, mark_synthetic, new_figure
from kaful.figures.truth_view import Series, load_truth, rul_records, true_health

COLUMNS = ("series", "estimator", "binding", "t", "value", "low", "high")
CHANNEL_PANEL_RATIO = 1.0
HEALTH_PANEL_RATIO = 0.75
OBS_MARKER_SIZE = 3.0
ISSUE_COLOR = "#999999"


@dataclass(frozen=True)
class Belief:
    estimator: str
    t: tuple[float, ...]
    expected: tuple[float, ...]
    health: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]]  # binding -> mean, p05, p95


@dataclass(frozen=True)
class TrackingData:
    label: str
    time_unit: str
    unit: str
    selection: str
    channel: str
    labels: dict[str, str]  # channel and bindings -> display names
    observations: Series
    beliefs: tuple[Belief, ...]
    truth: dict[str, Series]
    issues: tuple[float, ...]


def _observations(ctx: Context, spec: TrackingSpec, unit: str) -> Series:
    ll = ctx.lineage(spec.lineage)
    dataset = ll.dataset()
    u = next(x for x in dataset.units if x.unit_id == unit)
    src = FileSource(ll.file(u.observations.path), ll.contract(), dataset.time_unit, dataset.timestamp_origin)
    pts = [(o.t, float(v)) for o in src for k, v in o.values.items()
           if k == spec.channel and isinstance(v, (int, float)) and not isinstance(v, bool)]
    return tuple(t for t, _ in pts), tuple(v for _, v in pts)


def extract(spec: TrackingSpec, ctx: Context) -> TrackingData:
    ll = ctx.lineage(spec.lineage)
    units = load_truth(ll)
    records = rul_records(ll, units, spec.estimators, ctx.settings.evaluate)
    choice = choose_unit(records.get((spec.estimators[0], spec.quantity), []), spec.unit, spec.estimators[0],
                         spec.quantity, spec.unit_rule)
    beliefs: list[Belief] = []
    issues: tuple[float, ...] = ()
    for f in ll.forecasts(spec.estimators):
        if f.source != choice.unit:
            continue
        tr = read_trace(ll, f)
        col = f"{spec.channel}.expected"
        if col not in tr:
            raise ValueError(f"{spec.id}: trace {f.trace} has no {col}")
        health = {b: tuple(tuple(float(x) for x in tr[f"{b}.health{s}"]) for s in ("", "_p05", "_p95"))
                  for b in spec.bindings}
        beliefs.append(Belief(estimator=f.estimator, t=tuple(float(x) for x in tr["t"]),
                              expected=tuple(float(x) for x in tr[col]), health=health))  # type: ignore[arg-type]
        issues = tuple(i.t_issue for i in f.issues)
    order = {e: k for k, e in enumerate(spec.estimators)}
    return TrackingData(label=ll.spec.label, time_unit=ll.time_unit, unit=choice.unit, selection=choice.note,
                        channel=spec.channel,
                        labels={n: ctx.label(n) for n in (spec.channel, *spec.bindings)}, observations=_observations(ctx, spec, choice.unit),
                        beliefs=tuple(sorted(beliefs, key=lambda b: order[b.estimator])),
                        truth=true_health(ll, units[choice.unit], spec.bindings), issues=issues)


def rows(data: TrackingData) -> list[dict[str, Cell]]:
    out: list[dict[str, Cell]] = [{"series": "observation", "t": t, "value": v}
                                  for t, v in zip(*data.observations)]
    for b in data.beliefs:
        out += [{"series": "expected", "estimator": b.estimator, "t": t, "value": v} for t, v in zip(b.t, b.expected)]
        for binding, (mean, lo, hi) in sorted(b.health.items()):
            out += [{"series": "health", "estimator": b.estimator, "binding": binding, "t": t, "value": m,
                     "low": lo_, "high": hi_} for t, m, lo_, hi_ in zip(b.t, mean, lo, hi)]
    for binding, (t_, h) in sorted(data.truth.items()):
        out += [{"series": "true_health", "binding": binding, "t": t, "value": v} for t, v in zip(t_, h)]
    return out


def render(data: TrackingData, style: Style, width: str) -> Figure:
    bindings = sorted(data.truth)
    fig, axes = new_figure(style, width, rows=1 + len(bindings),
                           height_ratios=[CHANNEL_PANEL_RATIO] + [HEALTH_PANEL_RATIO] * len(bindings))
    ax = axes[0]
    ax.plot(*data.observations, ls="none", marker=".", ms=OBS_MARKER_SIZE, color=style.colors["observation"],
            label="observations")
    for b in data.beliefs:
        ax.plot(b.t, b.expected, color=style.colors[b.estimator], lw=1.2,
                ls="-" if b is data.beliefs[0] else "--", label=f"{pretty_estimator(b.estimator)}: expected")
    ax.set_ylabel(data.labels[data.channel])
    ax.set_title(f"{data.label} · {data.unit}", loc="left")
    ax.legend(loc="upper left", frameon=False)
    for a, binding in zip(axes[1:], bindings):
        for b in data.beliefs:
            mean, lo, hi = b.health[binding]
            c = style.colors[b.estimator]
            if b is data.beliefs[0]:
                a.fill_between(b.t, lo, hi, color=c, alpha=style.band_alpha, lw=0)
                a.plot(b.t, mean, color=c, lw=1.0, label=f"{pretty_estimator(b.estimator)} (mean, 5–95% band)")
            else:  # comparison estimators: the mean only (their spread is in F3 and F4)
                a.plot(b.t, mean, color=c, lw=1.0, ls="--", label=f"{pretty_estimator(b.estimator)} (mean)")
        a.plot(*data.truth[binding], color=style.colors["truth"], lw=1.2, label="true health")
        a.set_ylim(-0.02, 1.02)
        a.set_ylabel(f"{data.labels[binding]}\nhealth")
    axes[1].legend(loc="lower left", frameon=False)
    for a in axes:
        for t in data.issues:
            a.axvline(t, color=ISSUE_COLOR, lw=0.6, ls=":")
    axes[-1].set_xlabel(f"time ({data.time_unit})")
    axes[-1].set_xlim(0, float(np.max(data.beliefs[0].t)) if data.beliefs else None)
    mark_synthetic(axes[0], style, "grey dotted verticals: forecast issue times")
    return fig


def draw(spec: TrackingSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, ctx.style, spec.width), columns=COLUMNS, rows=rows(data),
                 selection=data.selection)
