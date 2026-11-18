"""F11: F3 for every unit (small multiples), so the one unit F3 shows can be judged against all of them.

Each panel is F3's drawing for one unit, in unit order, on a shared time axis. The unit F3's rule picks is marked
in its title. Rows are written exactly as F3 writes them, one block per unit.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from kaful.figures import f3_failure_time as f3
from kaful.figures.drawn import Context, Drawn
from kaful.figures.select import choose_unit
from kaful.figures.spec import FailureGridSpec, Style
from kaful.figures.style import Cell, fresh
from kaful.figures.truth_view import load_truth, rul_records

PANEL_HEIGHT_IN = 0.95


def extract(spec: FailureGridSpec, ctx: Context) -> tuple[list[f3.FailureData], str]:
    ll = ctx.lineage(spec.lineage)
    records = rul_records(ll, load_truth(ll), spec.estimators, ctx.settings.evaluate)
    choice = choose_unit(records.get((spec.estimators[0], spec.quantity), []), spec.unit, spec.estimators[0],
                         spec.quantity, spec.unit_rule)
    return f3.extract_units(ctx, ll, records, spec.quantity, spec.estimators), choice.note


def rows(data: list[f3.FailureData]) -> list[dict[str, Cell]]:
    return [r for d in data for r in f3.rows(d)]


def render(data: list[f3.FailureData], chosen: str, spec: FailureGridSpec, style: Style) -> Figure:
    fresh(style)
    n_rows = -(-len(data) // spec.columns)
    w = style.double_width_in if spec.width == "double" else style.single_width_in
    fig, axes = plt.subplots(n_rows, spec.columns, figsize=(w, PANEL_HEIGHT_IN * n_rows + 0.8), sharex=True,
                             squeeze=False, layout="constrained")
    flat = [a for row in axes for a in row]
    for ax, d in zip(flat, data):
        f3.draw_axes(ax, d, spec.estimators[0], style)
        ax.set_yticklabels([])
        ax.set_title(d.unit + (" (F3)" if d.unit == chosen else ""), loc="left", fontsize=style.font_size - 1,
                     fontweight="bold" if d.unit == chosen else "normal")
        ax.tick_params(axis="x", labelsize=style.font_size - 2)
    for ax in flat[len(data):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(f"{data[0].label} · {data[0].quantity_label} · all {len(data)} units · x: predicted failure time "
                 f"({data[0].time_unit}); rows: issue times, earliest at the top\n{style.synthetic_label}", x=0.01,
                 ha="left", fontsize=style.font_size)
    return fig


def draw(spec: FailureGridSpec, ctx: Context) -> Drawn:
    data, note = extract(spec, ctx)
    chosen = note.split(":")[0]
    return Drawn(figure=render(data, chosen, spec, ctx.style), columns=f3.COLUMNS, rows=rows(data),
                 selection=f"F3's unit: {note}")
