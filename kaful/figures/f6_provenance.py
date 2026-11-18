"""F6: where each machine's parameters come from, and which sources carry the RUL uncertainty.

Left: the number of parameters by provenance source (evaluate's grounding section). Right: each source's share of
the forecast uncertainty, precisely var(log RUL) at t = 0 under the priors (evaluate's sensitivity section). Both read from the evaluation reports; a
source with no parameter on any machine is left out of the legend.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn
from kaful.figures.spec import ProvenanceSpec, Style
from kaful.figures.style import Cell, new_figure

COLUMNS = ("machine", "source", "parameters", "variance_share")
BAR_HEIGHT = 0.6


@dataclass(frozen=True)
class SourceShare:
    machine: str
    source: str
    parameters: int
    variance_share: float | None  # None: no variance figure for this source


def extract(spec: ProvenanceSpec, ctx: Context) -> tuple[SourceShare, ...]:
    out = []
    for name in spec.lineages:
        ll = ctx.lineage(name)
        g, sens = ll.report.grounding, ll.report.sensitivity
        if g is None or sens is None:
            raise ValueError(f"{spec.id}: {name} lacks grounding or sensitivity")
        for src in spec.sources:
            out.append(SourceShare(machine=ll.spec.label, source=src, parameters=g.params_by_source.get(src, 0),
                                   variance_share=sens.variance_by_source.get(src)))
    return tuple(out)


def rows(data: tuple[SourceShare, ...]) -> list[dict[str, Cell]]:
    return [{"machine": d.machine, "source": d.source, "parameters": d.parameters,
             "variance_share": d.variance_share} for d in data]


def render(data: tuple[SourceShare, ...], sources: tuple[str, ...], colors: dict[str, str], style: Style,
           width: str) -> Figure:
    machines = list(dict.fromkeys(d.machine for d in data))
    used = [s for s in sources if any(d.source == s and (d.parameters or d.variance_share) for d in data)]
    fig, (left, right) = new_figure(style, width, sharex=False, cols=2, sharey=True)
    y = np.arange(len(machines), dtype=float)[::-1]
    for ax, value, title in ((left, "parameters", "parameters by source"),
                             (right, "variance_share", "share of forecast uncertainty")):
        start = np.zeros(len(machines))
        for src in used:
            vals = np.array([getattr(next(d for d in data if d.machine == m and d.source == src), value) or 0.0
                             for m in machines], dtype=float)
            ax.barh(y, vals, BAR_HEIGHT, left=start, color=colors[src], label=src.replace("_", " "))
            for yi, s0, v in zip(y, start, vals):
                if v > 0 and value == "parameters":
                    ax.text(s0 + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=style.font_size - 1,
                            color="white")
            start += vals
        ax.set_title(title, loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    right.set_xlim(0, 1)
    right.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    left.set_yticks(y)
    left.set_yticklabels(machines)
    fig.legend(*left.get_legend_handles_labels(), loc="outside lower center", ncols=len(used), frameon=False)
    return fig


def draw(spec: ProvenanceSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    missing = sorted(set(spec.sources) - set(ctx.style.source_colors))
    if missing:
        raise ValueError(f"{spec.id}: style.source_colors lacks {missing}")
    fig = render(data, spec.sources, ctx.style.source_colors, ctx.style, spec.width)
    return Drawn(figure=fig, columns=COLUMNS, rows=rows(data))
