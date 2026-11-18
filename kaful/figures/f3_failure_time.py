"""F3: the predicted failure time at each issue time, against the true failure time, for one unit chosen by rule.

One row per issue time, on a time axis. The primary estimator's 90% (5-95%) and 50% (25-75%) bars and median of the
sampled time of event; comparison estimators as a faint thin 90% bar below it. A grey tick marks when the forecast
was issued. Samples with no crossing before the forecast horizon are censored: a bar reaching them ends at the
horizon with an open arrow, never at the horizon as if it were a value. The true failure time is the next true
crossing after each issue, as calibration scores it (one vertical line when every row shares it).
"""

from dataclasses import dataclass, replace

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn, pretty_estimator
from kaful.figures.forecasts import FAN_QUANTILES, issue_rul, rul_quantiles
from kaful.figures.select import choose_unit
from kaful.figures.sources import LoadedLineage
from kaful.figures.spec import FailureTimeSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, mark_synthetic, new_figure
from kaful.figures.truth_view import load_truth, rul_records
from kaful.phases.p9_evaluate.f_calibration import Record

COLUMNS = ("unit", "estimator", "issue", "t_issue", "horizon", "true_time_of_event", "toe_p05", "toe_p25",
           "toe_p50", "toe_p75", "toe_p95", "p_event_before_horizon", "n_samples")
SECONDARY_OFFSET = 0.28  # rows below the primary bar
ROW_HEIGHT_IN = 0.55
X_MARGIN = 1.06  # the x axis ends past the horizon, room for the open arrows
BAR90_LW, BAR50_LW, SECONDARY_LW = 1.6, 6.0, 1.0
SECONDARY_ALPHA = 0.55


@dataclass(frozen=True)
class FailurePoint:
    estimator: str
    issue: int
    t_issue: float
    horizon: float
    toe: tuple[float, ...]  # FAN_QUANTILES of the time of event; +inf: past the horizon (censored)
    p_before_horizon: float
    n_samples: int


@dataclass(frozen=True)
class FailureData:
    label: str
    time_unit: str
    quantity: str
    quantity_label: str
    unit: str
    selection: str
    truths: tuple[tuple[int, float, float | None], ...]  # (issue, t_issue, true time of event or None)
    points: tuple[FailurePoint, ...]  # by issue, then the spec's estimator order


def extract_units(ctx: Context, ll: LoadedLineage, records: dict[tuple[str, str], list[Record]], quantity: str,
                  estimators: tuple[str, ...], only: str | None = None) -> list[FailureData]:
    """Each unit's failure-time data (only one unit when `only` is given); truths are calibration's `records`."""
    by_unit: dict[str, list[FailurePoint]] = {}
    for f in ll.forecasts(estimators):
        if only is not None and f.source != only:
            continue
        for i, issue in enumerate(f.issues):
            rul = issue_rul(ll, issue, quantity)
            by_unit.setdefault(f.source, []).append(FailurePoint(
                estimator=f.estimator, issue=i, t_issue=issue.t_issue, horizon=f.horizon,
                toe=tuple(issue.t_issue + q for q in rul_quantiles(rul)),
                p_before_horizon=float(np.isfinite(rul).mean()), n_samples=int(rul.size)))
    order = {e: k for k, e in enumerate(estimators)}
    out = []
    for unit, points in sorted(by_unit.items()):
        mine = sorted((r for r in records[(estimators[0], quantity)] if r.unit == unit), key=lambda r: r.issue)
        out.append(FailureData(label=ll.spec.label, time_unit=ll.time_unit, quantity=quantity,
                               quantity_label=ctx.label(quantity), unit=unit, selection="",
                               truths=tuple((r.issue, r.t, r.truth) for r in mine),
                               points=tuple(sorted(points, key=lambda p: (p.issue, order[p.estimator])))))
    return out


def extract(spec: FailureTimeSpec, ctx: Context) -> FailureData:
    ll = ctx.lineage(spec.lineage)
    records = rul_records(ll, load_truth(ll), spec.estimators, ctx.settings.evaluate)
    primary = spec.estimators[0]
    choice = choose_unit(records.get((primary, spec.quantity), []), spec.unit, primary, spec.quantity,
                         spec.unit_rule)
    (data,) = extract_units(ctx, ll, records, spec.quantity, spec.estimators, only=choice.unit)
    return replace(data, selection=choice.note)


def rows(data: FailureData) -> list[dict[str, Cell]]:
    truth = {i: toe for i, _, toe in data.truths}
    out: list[dict[str, Cell]] = []
    for p in data.points:
        row: dict[str, Cell] = {"unit": data.unit, "estimator": p.estimator, "issue": p.issue, "t_issue": p.t_issue,
                                "horizon": p.horizon, "true_time_of_event": truth.get(p.issue),
                                "p_event_before_horizon": p.p_before_horizon, "n_samples": p.n_samples}
        row.update({f"toe_p{round(q * 100):02d}": v for q, v in zip(FAN_QUANTILES, p.toe)})
        out.append(row)
    return out


def shared_truth(data: FailureData) -> float | None:
    """The true failure time when every issue shares it (one vertical line), else None."""
    toes = {toe for _, _, toe in data.truths}
    return next(iter(toes)) if len(toes) == 1 else None


def _bar(ax: Axes, lo: float, hi: float, y: float, horizon: float, color: str, lw: float, alpha: float = 1.0,
         label: str | None = None) -> None:
    """A bar from lo to hi; past the horizon it stops there with an open arrow (censored)."""
    end = min(hi, horizon)
    ax.plot([min(lo, horizon), end], [y, y], color=color, lw=lw, alpha=alpha, solid_capstyle="butt", label=label)
    if not np.isfinite(hi):
        ax.annotate("", xy=(horizon * X_MARGIN * 0.995, y), xytext=(end, y), alpha=alpha,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": max(lw * 0.6, 1.0),
                                "shrinkA": 0, "shrinkB": 0})


def _draw_row(ax: Axes, p: FailurePoint, y: float, primary: bool, style: Style, labelled: set[str]) -> None:
    c = style.colors[p.estimator]
    lo, q25, med, q75, hi = p.toe
    first = p.estimator not in labelled
    labelled.add(p.estimator)
    if not primary:
        name = f"{pretty_estimator(p.estimator)} 90%" if first else None
        _bar(ax, lo, hi, y, p.horizon, c, SECONDARY_LW, SECONDARY_ALPHA, name)
        return
    _bar(ax, lo, hi, y, p.horizon, c, BAR90_LW, label=f"{pretty_estimator(p.estimator)} 90%" if first else None)
    _bar(ax, q25, q75, y, p.horizon, c, BAR50_LW, label=f"{pretty_estimator(p.estimator)} 50%" if first else None)
    if np.isfinite(med):
        ax.plot([med], [y], marker="o", ms=4.5, mfc="white", mec=c, mew=1.2, ls="none",
                label="median" if first else None)


def _draw_truth(ax: Axes, data: FailureData, y_of: dict[int, float], style: Style) -> None:
    common = shared_truth(data)
    if common is not None:
        ax.axvline(common, color=style.colors["truth"], lw=1.2, label="true failure time")
        return
    for i, _, toe in data.truths:
        if toe is not None:
            ax.plot([toe], [y_of[i]], marker="|", ms=14, mew=1.6, color=style.colors["truth"], ls="none",
                    label="true failure time" if i == data.truths[0][0] else None)


def draw_axes(ax: Axes, data: FailureData, primary: str, style: Style) -> None:
    """One unit's rows on `ax` (shared by F3 and the all-units grid F11)."""
    issues = sorted({(p.issue, p.t_issue) for p in data.points})
    horizon = max(p.horizon for p in data.points)
    y_of = {i: -float(k) for k, (i, _) in enumerate(issues)}
    ax.axvline(horizon, color=NOTE_COLOR, lw=0.8, ls="--", label="forecast horizon")
    ax.plot([t for _, t in issues], [y_of[i] for i, _ in issues], marker="|", ms=10, mew=1.2, ls="none",
            color=NOTE_COLOR, label="issued")
    _draw_truth(ax, data, y_of, style)
    labelled: set[str] = set()
    for p in data.points:
        secondary = p.estimator != primary
        _draw_row(ax, p, y_of[p.issue] - (SECONDARY_OFFSET if secondary else 0.0), not secondary, style, labelled)
    ax.set_yticks([y_of[i] for i, _ in issues])
    ax.set_yticklabels([f"issued at {t:,.0f}" for _, t in issues])
    ax.set_ylim(-len(issues) + 0.4, 0.6)
    ax.set_xlim(0, horizon * X_MARGIN)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def render(data: FailureData, primary: str, style: Style, width: str) -> Figure:
    fig, (ax,) = new_figure(style, width)
    rows_ = len({p.issue for p in data.points})
    fig.set_figheight(max(style.panel_height_in, ROW_HEIGHT_IN * (rows_ + 2)))
    draw_axes(ax, data, primary, style)
    ax.set_xlabel(f"predicted failure time ({data.time_unit})")
    ax.set_title(f"{data.label} · {data.quantity_label} · {data.unit}", loc="left")
    fig.legend(loc="outside lower center", ncols=4, frameon=False)
    mark_synthetic(ax, style, "open arrow: interval reaches past the forecast horizon")
    return fig


def draw(spec: FailureTimeSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, spec.estimators[0], ctx.style, spec.width), columns=COLUMNS, rows=rows(data),
                 selection=data.selection)
