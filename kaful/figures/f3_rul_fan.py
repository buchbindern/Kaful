"""F3: the RUL forecast at each issue time, against the true RUL, for one unit chosen by rule.

At each issue time and estimator: the 5-95% and 25-75% ranges and the median of the sampled RUL (time of event -
t_issue). A range open past the horizon (censored samples) is drawn to the top edge with an arrow, never at the
horizon as if it were a value. The true RUL is the next true crossing after each issue, as calibration scores it
(a sawtooth when the part renews), with the alpha-lambda cone of +-alpha around it.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn, pretty_estimator
from kaful.figures.forecasts import FAN_QUANTILES, issue_rul, rul_quantiles
from kaful.figures.select import choose_unit
from kaful.figures.spec import RulFanSpec, Style
from kaful.figures.style import Cell, mark_synthetic, new_figure
from kaful.figures.truth_view import load_truth, rul_records

COLUMNS = ("unit", "estimator", "issue", "t_issue", "true_time_of_event", "true_rul", "rul_p05", "rul_p25",
           "rul_p50", "rul_p75", "rul_p95", "p_event_before_horizon", "n_samples")
OFFSET_SHARE = 0.018  # of the x range, between estimators at one issue time
TOP_MARGIN = 1.12


@dataclass(frozen=True)
class FanPoint:
    estimator: str
    issue: int
    t_issue: float
    quantiles: tuple[float, ...]  # FAN_QUANTILES; +inf past the horizon
    p_before_horizon: float
    n_samples: int


@dataclass(frozen=True)
class FanData:
    label: str
    time_unit: str
    quantity: str
    quantity_label: str
    unit: str
    selection: str
    alpha: float
    truths: tuple[tuple[int, float, float | None], ...]  # (issue, t_issue, true time of event or None)
    points: tuple[FanPoint, ...]


def extract(spec: RulFanSpec, ctx: Context) -> FanData:
    ll = ctx.lineage(spec.lineage)
    units = load_truth(ll)
    records = rul_records(ll, units, spec.estimators, ctx.settings.evaluate)
    primary = spec.estimators[0]
    choice = choose_unit(records.get((primary, spec.quantity), []), spec.unit, primary, spec.quantity,
                         spec.unit_rule)
    mine = sorted((r for r in records[(primary, spec.quantity)] if r.unit == choice.unit), key=lambda r: r.issue)
    points = []
    for f in ll.forecasts(spec.estimators):
        if f.source != choice.unit:
            continue
        for i, issue in enumerate(f.issues):
            rul = issue_rul(ll, issue, spec.quantity)
            points.append(FanPoint(estimator=f.estimator, issue=i, t_issue=issue.t_issue,
                                   quantiles=rul_quantiles(rul), p_before_horizon=float(np.isfinite(rul).mean()),
                                   n_samples=int(rul.size)))
    if ll.report.calibration is None:
        raise ValueError(f"{spec.id}: {spec.lineage} has no calibration section")
    order = {e: k for k, e in enumerate(spec.estimators)}
    return FanData(label=ll.spec.label, time_unit=ll.time_unit, quantity=spec.quantity,
                   quantity_label=ctx.label(spec.quantity), unit=choice.unit,
                   selection=choice.note, alpha=ll.report.calibration.alpha,
                   truths=tuple((r.issue, r.t, r.truth) for r in mine),
                   points=tuple(sorted(points, key=lambda p: (p.issue, order[p.estimator]))))


def rows(data: FanData) -> list[dict[str, Cell]]:
    truth = {i: toe for i, _, toe in data.truths}
    out: list[dict[str, Cell]] = []
    for p in data.points:
        toe = truth.get(p.issue)
        row: dict[str, Cell] = {"unit": data.unit, "estimator": p.estimator, "issue": p.issue, "t_issue": p.t_issue,
                                "true_time_of_event": toe, "true_rul": toe - p.t_issue if toe is not None else None,
                                "p_event_before_horizon": p.p_before_horizon, "n_samples": p.n_samples}
        row.update({f"rul_p{round(q * 100):02d}": v for q, v in zip(FAN_QUANTILES, p.quantiles)})
        out.append(row)
    return out


def _truth_segments(data: FanData) -> list[tuple[float, float]]:
    """(start, true time of event) per distinct next event still ahead at its issue: the true RUL is toe - t over
    [start, toe]. An issue at or past a failure with no renewal (toe == t_issue) has true RUL 0: no segment."""
    segs: list[tuple[float, float]] = []
    for _, t, toe in data.truths:
        if toe is not None and toe > t and (not segs or segs[-1][1] != toe):
            segs.append((segs[-1][1] if segs else 0.0, toe))
    return segs


def _triggered(data: FanData) -> list[float]:
    """Issue times at which the part had already reached its limit (true RUL 0)."""
    return [t for _, t, toe in data.truths if toe is not None and toe <= t]


def _draw_truth(ax: Axes, data: FanData, style: Style) -> None:
    for k, (start, toe) in enumerate(_truth_segments(data)):
        x = np.linspace(start, toe, 2)
        ax.fill_between(x, (1 - data.alpha) * (toe - x), (1 + data.alpha) * (toe - x), color=style.colors["cone"],
                        lw=0, label=f"±{data.alpha:.0%} of true RUL (α-λ)" if k == 0 else None)
        ax.plot(x, toe - x, color=style.colors["truth"], lw=1.2, label="true RUL" if k == 0 else None)
    done = _triggered(data)
    if done:
        ax.plot(done, [0.0] * len(done), ls="none", marker="x", ms=6, mew=1.4, color=style.colors["truth"],
                label="already failed at issue (true RUL 0)", clip_on=False, zorder=5)


def _draw_point(ax: Axes, p: FanPoint, x: float, top: float, style: Style, labelled: set[str]) -> None:
    c = style.colors[p.estimator]
    lo, q25, med, q75, hi = (min(v, top) for v in p.quantiles)
    label = None if p.estimator in labelled else f"{pretty_estimator(p.estimator)} (5–95%, 25–75%, median)"
    labelled.add(p.estimator)
    ax.plot([x, x], [lo, hi], color=c, lw=1.0, solid_capstyle="butt", label=label)
    ax.plot([x, x], [q25, q75], color=c, lw=4.0, solid_capstyle="butt")
    if np.isfinite(p.quantiles[-1]):
        ax.plot([x], [med], marker="o", ms=4, mfc="white", mec=c, mew=1.0, ls="none")
    else:
        ax.annotate("", xy=(x, top), xytext=(x, top * 0.94), arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.0})
        if np.isfinite(p.quantiles[2]):
            ax.plot([x], [med], marker="o", ms=4, mfc="white", mec=c, mew=1.0, ls="none")


def render(data: FanData, style: Style, width: str) -> Figure:
    fig, (ax,) = new_figure(style, width)
    segs = _truth_segments(data)
    finite = [v for p in data.points for v in p.quantiles if np.isfinite(v)] + [toe - s for s, toe in segs]
    top = TOP_MARGIN * max(finite) if finite else 1.0
    x_end = max([toe for _, toe in segs] + [p.t_issue for p in data.points]) * 1.04
    _draw_truth(ax, data, style)
    n = len({p.estimator for p in data.points})
    labelled: set[str] = set()
    for k, p in enumerate(data.points):
        x = p.t_issue + (k % n - (n - 1) / 2) * OFFSET_SHARE * x_end
        _draw_point(ax, p, x, top, style, labelled)
    ax.set_xlim(0, x_end)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"time ({data.time_unit})")
    ax.set_ylabel(f"remaining useful life ({data.time_unit})")
    ax.set_title(f"{data.label} · {data.quantity_label} · {data.unit}", loc="left")
    ax.legend(loc="upper right", frameon=False)
    mark_synthetic(ax, style, "arrow: interval open past the horizon")
    return fig


def draw(spec: RulFanSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, ctx.style, spec.width), columns=COLUMNS, rows=rows(data),
                 selection=data.selection)
