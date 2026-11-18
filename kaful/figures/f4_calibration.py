"""F4: 90% interval coverage per quantity, from each lineage's evaluation report (no recomputation).

Per quantity and estimator: the pooled coverage and its Wilson 95% interval on units (the honest one, since pairs in
a unit are correlated), against the 85-95% target band. Observed quantities (judged on the primary estimator) are
drawn above the line; joined and unobserved parts below it, labelled circular (the prior predictive scored against
truth drawn from that prior). Per-part cells of a non-identifiable group are left out: their group's first failure
is the judged quantity.
"""

import math
from dataclasses import dataclass

from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from kaful.artifacts.evaluation import CALIBRATION_TARGET_90
from kaful.figures.drawn import Context, Drawn, pretty_estimator
from kaful.figures.spec import CalibrationSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, mark_synthetic, new_figure

COLUMNS = ("machine", "quantity", "estimator", "part_class", "coverage", "ci_units_low", "ci_units_high", "units",
           "pairs", "verdict")
CIRCULAR = ("joined", "unobserved")
ROW_STEP = 1.0
EST_STEP = 0.28
X_FLOOR_STEP = 0.1  # the x axis starts at the lowest CI bound, rounded down to this step
X_MAX = 1.0
ROW_HEIGHT_IN = 0.26
CIRCULAR_GAP = 0.8  # extra rows between the observed and the circular groups
CIRCULAR_NOTE = "below: circular (no observation reaches the part; prior predictive vs truth drawn from that prior)"


@dataclass(frozen=True)
class CoveragePoint:
    machine: str
    quantity: str
    estimator: str
    part_class: str
    coverage: float
    ci: tuple[float, float]
    units: int
    pairs: int
    verdict: str | None


def extract(spec: CalibrationSpec, ctx: Context) -> tuple[CoveragePoint, ...]:
    out = []
    for name in spec.lineages:
        ll = ctx.lineage(name)
        cal = ll.report.calibration
        if cal is None:
            raise ValueError(f"{spec.id}: {name} has no calibration section")
        verdicts = {v.quantity: v.verdict for v in cal.verdicts}
        for c in cal.cells:
            if c.issue_fraction is not None or c.estimator not in spec.estimators or c.share is None \
                    or c.ci_units is None or c.part_class == "not_identifiable" \
                    or (spec.observed_only and c.part_class in CIRCULAR):
                continue
            judged = c.estimator == cal.primary_estimator and c.part_class == "observed"
            out.append(CoveragePoint(machine=ll.spec.label, quantity=c.quantity, estimator=c.estimator,
                                     part_class=c.part_class, coverage=c.share, ci=(c.ci_units.low, c.ci_units.high),
                                     units=c.units, pairs=c.pairs,
                                     verdict=verdicts.get(c.quantity) if judged else None))
    order = {e: k for k, e in enumerate(spec.estimators)}
    lin = {ctx.spec.lineages[n].label: k for k, n in enumerate(spec.lineages)}
    return tuple(sorted(out, key=lambda p: (p.part_class in CIRCULAR, lin[p.machine], p.quantity,
                                            order[p.estimator])))


def rows(points: tuple[CoveragePoint, ...]) -> list[dict[str, Cell]]:
    return [{"machine": p.machine, "quantity": p.quantity, "estimator": p.estimator, "part_class": p.part_class,
             "coverage": p.coverage, "ci_units_low": p.ci[0], "ci_units_high": p.ci[1], "units": p.units,
             "pairs": p.pairs, "verdict": p.verdict} for p in points]


def render(points: tuple[CoveragePoint, ...], estimators: tuple[str, ...], labels: dict[str, str], style: Style,
           width: str, x_min: float | None = None) -> Figure:
    keys = list(dict.fromkeys((p.machine, p.quantity, p.part_class) for p in points))
    fig, (ax,) = new_figure(style, width)
    fig.set_figheight(max(style.panel_height_in, ROW_HEIGHT_IN * (len(keys) + 2)))
    lo, hi = CALIBRATION_TARGET_90
    ax.axvspan(lo, hi, color=style.colors["target"], alpha=style.band_alpha, lw=0, label="target 85–95%")
    ax.axvline(0.9, color=style.colors["target"], lw=0.8, ls=":")
    if x_min is None:
        x_min = math.floor(min(p.ci[0] for p in points) / X_FLOOR_STEP) * X_FLOOR_STEP
    y_of = {k: -(i + (CIRCULAR_GAP if k[2] in CIRCULAR else 0.0)) * ROW_STEP for i, k in enumerate(keys)}
    labelled: set[str] = set()
    for p in points:
        k = estimators.index(p.estimator)
        y = y_of[(p.machine, p.quantity, p.part_class)] - (k - (len(estimators) - 1) / 2) * EST_STEP
        c = style.colors[p.estimator]
        filled = p.part_class not in CIRCULAR
        ax.plot([max(p.ci[0], x_min), p.ci[1]], [y, y], color=c, lw=1.2)
        if p.ci[0] < x_min:  # the CI runs past the axis start: an arrowhead at the edge (the CSV keeps the value)
            ax.plot([x_min], [y], marker="<", ms=4, color=c, ls="none", clip_on=False)
        ax.plot([p.coverage], [y], marker="o" if k == 0 else "s", ms=4.5, ls="none", mec=c,
                mfc=c if filled else "white",
                label=None if p.estimator in labelled else f"{pretty_estimator(p.estimator)} (95% CI on units)")
        labelled.add(p.estimator)
    ax.set_yticks([y_of[k] for k in keys])
    ax.set_yticklabels([f"{m} · {labels[q]} (n={_units(points, m, q)})" for m, q, _ in keys])
    first_circ = next((i for i, k in enumerate(keys) if k[2] in CIRCULAR), None)
    if first_circ is not None and first_circ > 0:
        y = -(first_circ - 1 + (1 + CIRCULAR_GAP) / 2) * ROW_STEP
        ax.axhline(y + 0.18 * ROW_STEP, color=NOTE_COLOR, lw=0.6)
        ax.text(x_min + 0.005, y, CIRCULAR_NOTE, va="center", fontsize=style.font_size - 2, color=NOTE_COLOR)
    ax.set_xlim(x_min, X_MAX + 0.01)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("share of true failure times inside the 90% interval")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.legend(loc="outside lower center", ncols=len(estimators) + 1, frameon=False)
    mark_synthetic(ax, style)
    return fig


def _units(points: tuple[CoveragePoint, ...], machine: str, quantity: str) -> int:
    return max(p.units for p in points if p.machine == machine and p.quantity == quantity)


def draw(spec: CalibrationSpec, ctx: Context) -> Drawn:
    points = extract(spec, ctx)
    labels = {p.quantity: ctx.label(p.quantity) for p in points}
    fig = render(points, spec.estimators, labels, ctx.style, spec.width, spec.x_min)
    return Drawn(figure=fig, columns=COLUMNS, rows=rows(points))
