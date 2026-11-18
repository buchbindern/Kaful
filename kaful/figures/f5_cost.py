"""F5: cold-run API cost per phase against the M1 legacy baseline, from each lineage's evaluation report.

Per machine and phase: the legacy cold-run cost (M1, its steps mapped onto the new phases) and the new pipeline's
cold-run cost as a range, from every cached LLM call batched (low) to none batched (high), as evaluate computes it
(billed + unbilled cache hits). A phase at $0 in both is left out; a new phase at $0 is labelled "$0".
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from kaful.figures.drawn import Context, Drawn
from kaful.figures.spec import CostSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, new_figure
from kaful.phases import PHASE_ORDER

COLUMNS = ("machine", "phase", "legacy_usd", "new_low_usd", "new_high_usd")
BAR_WIDTH = 0.38
ZERO_LABEL = "$0"
CENT = 0.01


@dataclass(frozen=True)
class PhaseCost:
    phase: str
    legacy: float | None  # None: no M1 figure for this phase
    low: float
    high: float


@dataclass(frozen=True)
class MachineCost:
    label: str
    phases: tuple[PhaseCost, ...]
    legacy_total: float
    low_total: float
    high_total: float


def extract(spec: CostSpec, ctx: Context) -> tuple[MachineCost, ...]:
    out = []
    for name in spec.lineages:
        ll = ctx.lineage(name)
        cost = ll.report.cost
        if cost.m1 is None:
            raise ValueError(f"{spec.id}: {name} has no M1 baseline")
        new = {p.phase: p for p in cost.by_phase}
        phases = []
        for ph in PHASE_ORDER:
            legacy = cost.m1.by_phase_usd.get(ph)
            p = new.get(ph)
            low, high = (p.cold_low_usd, p.cold_usd) if p is not None else (0.0, 0.0)
            if (legacy or 0.0) > 0.0 or high > 0.0:
                phases.append(PhaseCost(phase=ph, legacy=legacy, low=low, high=high))
        out.append(MachineCost(label=ll.spec.label, phases=tuple(phases), legacy_total=cost.m1.cold_total_usd,
                               low_total=cost.cold_low_usd, high_total=cost.cold_usd))
    return tuple(out)


def rows(data: tuple[MachineCost, ...]) -> list[dict[str, Cell]]:
    out: list[dict[str, Cell]] = []
    for m in data:
        out += [{"machine": m.label, "phase": p.phase, "legacy_usd": p.legacy, "new_low_usd": p.low,
                 "new_high_usd": p.high} for p in m.phases]
        out.append({"machine": m.label, "phase": "total", "legacy_usd": m.legacy_total, "new_low_usd": m.low_total,
                    "new_high_usd": m.high_total})
    return out


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def render(data: tuple[MachineCost, ...], style: Style, width: str) -> Figure:
    fig, axes = new_figure(style, width, rows=len(data), sharex=False)
    for ax, m in zip(axes, data):
        x = np.arange(len(m.phases), dtype=float)
        legacy = [p.legacy or 0.0 for p in m.phases]
        ax.bar(x - BAR_WIDTH / 2, legacy, BAR_WIDTH, color=style.colors["legacy"], label="legacy (M1 cold run)")
        highs = [p.high for p in m.phases]
        ax.bar(x + BAR_WIDTH / 2, highs, BAR_WIDTH, color=style.colors["cost_high"],
               label="new, extra if none batched")
        ax.bar(x + BAR_WIDTH / 2, [p.low for p in m.phases], BAR_WIDTH, color=style.colors["particle_filter"],
               label="new, all cached calls batched")
        top = max(legacy + highs)
        for xi, p in zip(x, m.phases):
            if p.high < CENT:
                ax.text(xi + BAR_WIDTH / 2, top * 0.02, ZERO_LABEL if p.high == 0.0 else "<$0.01", ha="center",
                        va="bottom", fontsize=style.font_size - 1, color=NOTE_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([p.phase for p in m.phases])
        ax.set_ylabel("USD per cold run")
        ax.set_title(f"{m.label}: {_usd(m.low_total)}–{_usd(m.high_total)} vs legacy {_usd(m.legacy_total)}",
                     loc="left")
    axes[0].legend(loc="upper right", frameon=False)
    return fig


def draw(spec: CostSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, ctx.style, spec.width), columns=COLUMNS, rows=rows(data))
