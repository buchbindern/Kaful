"""Shared look and byte-stable output for every figure (used by figures/build.py and each figure's render).

- `rc(style)`: the matplotlib settings, applied inside a context so nothing leaks into other code;
- `new_figure`: a figure at the spec's single- or double-column width;
- `encode`: SVG (text as paths, fixed hash salt, no date or creator) and PNG (no software tag) bytes, so the same
  data gives the same files under any PYTHONHASHSEED;
- `csv_bytes`: the plotted numbers, floats written with repr (exact), '' for none or not finite.
"""

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from kaful.figures.spec import Style  # noqa: E402

Cell = str | int | float | None


def rc(style: Style) -> dict[str, Any]:
    return {
        "font.family": style.font_family,
        "font.size": style.font_size,
        "axes.titlesize": style.font_size,
        "axes.labelsize": style.font_size,
        "legend.fontsize": style.font_size - 1,
        "xtick.labelsize": style.font_size - 1,
        "ytick.labelsize": style.font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "path",
        "svg.hashsalt": style.svg_hashsalt,
        "path.simplify": False,
        "figure.dpi": style.dpi,
        "savefig.dpi": style.dpi,
    }


def new_figure(style: Style, width: str, rows: int = 1, height_ratios: Sequence[float] | None = None,
               sharex: bool = True) -> tuple[Figure, list[Any]]:
    w = style.double_width_in if width == "double" else style.single_width_in
    ratios = list(height_ratios) if height_ratios is not None else [1.0] * rows
    fig, axes = plt.subplots(rows, 1, figsize=(w, style.panel_height_in * sum(ratios)), sharex=sharex,
                             gridspec_kw={"height_ratios": ratios}, squeeze=False, layout="constrained")
    return fig, [a for row in axes for a in row]


NOTE_COLOR = "#555555"


def mark_synthetic(ax: Any, style: Style, extra: str = "") -> None:
    """The synthetic-data note, right-aligned above the (top) axes, so it never collides with axis labels."""
    text = style.synthetic_label + (f"\n{extra}" if extra else "")
    ax.set_title(text, loc="right", fontsize=style.font_size - 2, color=NOTE_COLOR)


def encode(fig: Figure, style: Style) -> dict[str, bytes]:
    out = {}
    with io.BytesIO() as buf:
        fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None})
        out["svg"] = buf.getvalue()
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=style.dpi, metadata={"Software": None})
        out["png"] = buf.getvalue()
    plt.close(fig)
    return out


def cell(value: Cell) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    return str(value)


def csv_bytes(columns: Sequence[str], rows: Sequence[Mapping[str, Cell]]) -> bytes:
    with io.StringIO() as buf:
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([cell(r.get(c)) for c in columns])
        return buf.getvalue().encode("utf-8")
