"""F7: how much the data sharpens the forecast ("calibrated but not sharp"), from the evaluation reports.

Left: each part's life contraction, the posterior 90% width of its life (D_limit / k) over its prior width, median
over units at the particle filter's last issue (1 = the data taught nothing). Filled: parts an observation reaches
(observed, or in a shared-channel group); hollow: joined or unobserved parts.
Right: per observed quantity and estimator, the median 90% interval width over the true RUL. When more than half of
those intervals reach past the horizon the median is open: drawn as an arrow to the edge, labelled with the count.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn, pretty_estimator
from kaful.figures.spec import SharpnessSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, mark_synthetic, new_figure

COLUMNS = ("panel", "machine", "item", "estimator", "part_class", "value", "open_count", "pairs")
REACHED = ("observed", "not_identifiable")
EST_STEP = 0.26
OPEN_EDGE = 0.55  # share of the right panel's x range where an open median's arrow ends
ROW_HEIGHT_IN = 0.22
WIDTH_ROWS = 1.6  # a width row holds two estimators: this many contraction rows high


@dataclass(frozen=True)
class Contraction:
    machine: str
    component: str
    part_class: str
    value: float


@dataclass(frozen=True)
class Width:
    machine: str
    quantity: str
    estimator: str
    value: float | None  # None: the median interval reaches past the horizon
    open_count: int
    pairs: int


def extract(spec: SharpnessSpec, ctx: Context) -> tuple[tuple[Contraction, ...], tuple[Width, ...]]:
    contractions, widths = [], []
    for name in spec.lineages:
        ll = ctx.lineage(name)
        sens, cal = ll.report.sensitivity, ll.report.calibration
        if sens is None or cal is None:
            raise ValueError(f"{spec.id}: {name} lacks sensitivity or calibration")
        contractions += [Contraction(ll.spec.label, c.component, c.part_class, c.life_contraction)
                         for c in sorted(sens.components, key=lambda c: (c.part_class not in REACHED, c.component))
                         if c.life_contraction is not None]
        widths += [Width(ll.spec.label, a.quantity, a.estimator, a.width_relative_median, a.width_censored,
                         a.alpha_lambda_n)
                   for a in cal.accuracy if a.part_class == "observed" and a.estimator in spec.estimators]
    order = {e: k for k, e in enumerate(spec.estimators)}
    lin = {ctx.spec.lineages[n].label: k for k, n in enumerate(spec.lineages)}
    widths.sort(key=lambda w: (lin[w.machine], w.quantity, order[w.estimator]))
    return tuple(contractions), tuple(widths)


def rows(data: tuple[tuple[Contraction, ...], tuple[Width, ...]]) -> list[dict[str, Cell]]:
    cs, ws = data
    out: list[dict[str, Cell]] = [{"panel": "life_contraction", "machine": c.machine, "item": c.component,
                                   "part_class": c.part_class, "value": c.value} for c in cs]
    out += [{"panel": "width_over_true_rul", "machine": w.machine, "item": w.quantity, "estimator": w.estimator,
             "value": w.value, "open_count": w.open_count, "pairs": w.pairs} for w in ws]
    return out


def _contraction(ax: Axes, cs: tuple[Contraction, ...], labels: dict[str, str], style: Style) -> None:
    y = -np.arange(len(cs), dtype=float)
    c = style.colors["particle_filter"]
    for yi, item in zip(y, cs):
        filled = item.part_class in REACHED
        ax.plot([item.value], [yi], marker="o", ms=4.5, ls="none", mec=c, mfc=c if filled else "white")
    ax.axvline(1.0, color=NOTE_COLOR, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{i.machine} · {labels[i.component]}" for i in cs])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("posterior / prior 90% width of life")
    ax.set_title("what the data taught (1 = nothing)", loc="left")


def _widths(ax: Axes, ws: tuple[Width, ...], estimators: tuple[str, ...], labels: dict[str, str],
            style: Style) -> None:
    keys = list(dict.fromkeys((w.machine, w.quantity) for w in ws))
    finite = [w.value for w in ws if w.value is not None]
    x_max = max(finite) * 1.25 if finite else 1.0
    y_of = {k: -float(i) for i, k in enumerate(keys)}
    for w in ws:
        k = estimators.index(w.estimator)
        y = y_of[(w.machine, w.quantity)] - (k - (len(estimators) - 1) / 2) * EST_STEP
        c = style.colors[w.estimator]
        if w.value is not None:
            ax.plot([0, w.value], [y, y], color=c, lw=3, solid_capstyle="butt")
        else:
            ax.annotate("", xy=(x_max * OPEN_EDGE, y), xytext=(0, y),
                        arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.2, "shrinkA": 0, "shrinkB": 0})
            ax.text(x_max * OPEN_EDGE, y, f" open past the horizon in {w.open_count}/{w.pairs}", ha="left",
                    va="center", fontsize=style.font_size - 2, color=NOTE_COLOR, clip_on=False)
    for e in estimators:
        ax.plot([], [], color=style.colors[e], lw=3, label=pretty_estimator(e))
    ax.axvline(1.0, color=NOTE_COLOR, lw=0.8, ls=":")
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([f"{m} · {labels[q]}" for m, q in keys])
    ax.set_xlim(0, x_max)
    ax.set_ylim(-(len(keys) - 1) - 0.5, 0.5)
    ax.set_xlabel("median 90% width / true RUL")
    ax.set_title("interval width (observed quantities)", loc="left")


def render(data: tuple[tuple[Contraction, ...], tuple[Width, ...]], estimators: tuple[str, ...],
           labels: dict[str, str], style: Style, width: str) -> Figure:
    cs, ws = data
    n_w = len({(w.machine, w.quantity) for w in ws})
    fig, (top, bottom) = new_figure(style, width, rows=2, sharex=False, height_ratios=[len(cs), n_w * WIDTH_ROWS])
    fig.set_figheight(ROW_HEIGHT_IN * (len(cs) + n_w * WIDTH_ROWS + 4))
    _contraction(top, cs, labels, style)
    _widths(bottom, ws, estimators, labels, style)
    for ax in (top, bottom):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    mark_synthetic(top, style)
    fig.legend(*bottom.get_legend_handles_labels(), loc="outside lower center", ncols=len(estimators), frameon=False)
    return fig


def draw(spec: SharpnessSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    names = {c.component for c in data[0]} | {w.quantity for w in data[1]}
    labels = {n: ctx.label(n) for n in names}
    return Drawn(figure=render(data, spec.estimators, labels, ctx.style, spec.width), columns=COLUMNS,
                 rows=rows(data))
