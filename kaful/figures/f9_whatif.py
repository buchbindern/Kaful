"""F9: the water-filter → boiler what-if, from a saved `kaful estimate whatif` report (M13b).

Left: on a new machine, each boiler's time to its scaling limit with the filter replaced at its documented point vs
later (5%, median, 95% over the forecast samples; a 95% past the horizon is an open arrow). Right: the long-run change
in each boiler's scaling rate, 5-95% with its median, against the range if the coupling strength r were known
(the report's `rate_change_r_known`). The report's disclaimer travels with the figure: the boilers' k and r are
priors only (W-014 does not identify them).
"""

import json
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn
from kaful.figures.spec import Style, WhatIfSpec
from kaful.figures.style import NOTE_COLOR, Cell, mark_synthetic, new_figure

COLUMNS = ("panel", "item", "case", "p05", "p50", "p95", "low", "high")
ROW_STEP, CASE_STEP, PAD = 1.0, 0.3, 0.45
BARELY = 0.05  # every median moves less than this share: "barely any difference"
X_MARGIN = 1.06


@dataclass(frozen=True)
class TimeToLimit:
    binding: str
    case: str  # documented | delayed
    p05: float | None
    p50: float | None
    p95: float | None  # None: past the horizon


@dataclass(frozen=True)
class RateChange:
    connection: str
    p05: float
    p50: float
    p95: float
    known_low: float
    known_high: float


@dataclass(frozen=True)
class WhatIfData:
    label: str
    time_unit: str
    factor: float
    horizon: float
    times: tuple[TimeToLimit, ...]
    rates: tuple[RateChange, ...]
    disclaimer: str


def extract(spec: WhatIfSpec, ctx: Context) -> WhatIfData:
    ll = ctx.lineage(spec.lineage)
    report = json.loads(ll.file(spec.file).read_text(encoding="utf-8"))
    if report["run_id"] != ll.lineage.get("estimate").produced_by:
        raise ValueError(f"{spec.id}: {spec.file} was made from run {report['run_id']}, not this lineage's estimate")
    (binding, factor), = report["renew_at"].items()
    times = [TimeToLimit(e["binding"], case, e[case]["p05"], e[case]["p50"], e[case]["p95"])
             for e in sorted(report["from_new"]["events"], key=lambda e: e["binding"]) if e["binding"] != binding
             for case in ("documented", "delayed")]
    rates = [RateChange(name, r["rate_change"]["p05"], r["rate_change"]["p50"], r["rate_change"]["p95"],
                        r["rate_change_r_known"]["low"], r["rate_change_r_known"]["high"])
             for name, r in sorted(report["long_run"].items())]
    return WhatIfData(label=ll.spec.label, time_unit=report["time_unit"], factor=float(factor),
                      horizon=float(report["horizon"]), times=tuple(times), rates=tuple(rates),
                      disclaimer=report["disclaimer"])


def rows(d: WhatIfData) -> list[dict[str, Cell]]:
    out: list[dict[str, Cell]] = [{"panel": "time_to_limit", "item": t.binding, "case": t.case, "p05": t.p05,
                                   "p50": t.p50, "p95": t.p95} for t in d.times]
    out += [{"panel": "long_run_rate_change", "item": r.connection, "case": "r_unknown", "p05": r.p05, "p50": r.p50,
             "p95": r.p95, "low": r.known_low, "high": r.known_high} for r in d.rates]
    return out


def median_changes(d: WhatIfData) -> dict[str, float]:
    """Per binding, the relative change of the median time to limit (delayed vs documented)."""
    by = {(t.binding, t.case): t.p50 for t in d.times}
    out = {}
    for b in dict.fromkeys(t.binding for t in d.times):
        doc, late = by.get((b, "documented")), by.get((b, "delayed"))
        if doc and late is not None:
            out[b] = late / doc - 1.0
    return out


def _subtitle(d: WhatIfData, labels: dict[str, str]) -> str:
    changes = median_changes(d)
    word = "barely any difference" if all(abs(c) < BARELY for c in changes.values()) else "medians move"
    moved = ", ".join(f"{labels[b]} {c:+.0%}" for b, c in changes.items())
    return f"on a new machine, {word}\n(median time to limit: {moved})"


def _times(ax: Axes, d: WhatIfData, labels: dict[str, str], style: Style) -> None:
    items = list(dict.fromkeys(t.binding for t in d.times))
    colors = {"documented": style.colors["with_maintenance"], "delayed": style.colors["without_maintenance"]}
    names = {"documented": "filter replaced at W-014", "delayed": f"filter replaced at {d.factor:g}× W-014"}
    seen: set[str] = set()
    for t in d.times:
        y = -items.index(t.binding) * ROW_STEP - (CASE_STEP if t.case == "delayed" else 0.0)
        lo, hi = t.p05 or 0.0, t.p95 if t.p95 is not None else d.horizon
        ax.plot([lo, hi], [y, y], color=colors[t.case], lw=2, solid_capstyle="butt",
                label=None if t.case in seen else names[t.case])
        seen.add(t.case)
        if t.p95 is None:
            ax.annotate("", xy=(d.horizon * X_MARGIN, y), xytext=(d.horizon, y),
                        arrowprops={"arrowstyle": "-|>", "color": colors[t.case], "lw": 1.2, "shrinkA": 0,
                                    "shrinkB": 0})
        if t.p50 is not None:
            ax.plot([t.p50], [y], marker="o", ms=4.5, mfc="white", mec=colors[t.case], ls="none")
    ax.axvline(d.horizon, color=NOTE_COLOR, lw=0.8, ls="--")
    ax.set_yticks([-i * ROW_STEP - CASE_STEP / 2 for i in range(len(items))])
    ax.set_yticklabels([labels[b] for b in items])
    ax.set_xlim(0, d.horizon * X_MARGIN * 1.01)
    ax.set_ylim(-(len(items) - 1) * ROW_STEP - CASE_STEP - PAD, PAD)
    ax.set_xlabel(f"time to scaling limit on a new machine ({d.time_unit})")
    ax.set_title(_subtitle(d, labels), loc="left")


def _rates(ax: Axes, d: WhatIfData, labels: dict[str, str], style: Style) -> None:
    c = style.colors["without_maintenance"]
    for i, r in enumerate(d.rates):
        y = -i * ROW_STEP
        ax.plot([r.p05, r.p95], [y, y], color=c, lw=1.4, label="5–95% (r unknown)" if i == 0 else None)
        ax.plot([r.known_low, r.known_high], [y, y], color=c, lw=6, alpha=0.5, solid_capstyle="butt",
                label="if r were known to ±0.1" if i == 0 else None)
        ax.plot([r.p50], [y], marker="o", ms=4.5, mfc="white", mec=c, ls="none")
    ax.axvline(0.0, color=NOTE_COLOR, lw=0.8)
    ax.set_yticks([-i * ROW_STEP for i in range(len(d.rates))])
    ax.set_yticklabels([labels[r.connection] for r in d.rates])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.set_ylim(-(len(d.rates) - 1) * ROW_STEP - PAD, PAD)
    ax.set_xlabel("long-run change in scaling rate")
    ax.set_title("over many filter cycles", loc="left")


def render(d: WhatIfData, labels: dict[str, str], style: Style, width: str) -> Figure:
    fig, (left, right) = new_figure(style, width, sharex=False, cols=2)
    _times(left, d, labels, style)
    _rates(right, d, labels, style)
    for ax in (left, right):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    (h_left, l_left), (h_right, l_right) = left.get_legend_handles_labels(), right.get_legend_handles_labels()
    fig.legend([*h_left, *h_right], [*l_left, *l_right], loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(f"{d.label}: what if the water filter is replaced late ({d.factor:g}× its W-014 point)?", x=0.01,
                 ha="left", fontsize=style.font_size + 1)
    mark_synthetic(right, style.model_copy(update={"synthetic_label": "boiler k and r are priors only"}))
    return fig


def draw(spec: WhatIfSpec, ctx: Context) -> Drawn:
    d = extract(spec, ctx)
    labels = {n: ctx.label(n) for n in {t.binding for t in d.times} | {r.connection for r in d.rates}}
    return Drawn(figure=render(d, labels, ctx.style, spec.width), columns=COLUMNS, rows=rows(d),
                 notes=(d.disclaimer,))

