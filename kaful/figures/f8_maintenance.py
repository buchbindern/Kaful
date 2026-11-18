"""F8: a part's health with and without its documented maintenance, by simulation of the saved twin (approved
2026-11-18: derived by simulation, not a saved artifact; recorded in the manifest's notes).

For each part: the backend simulates the verified TwinDefinition and ParamSet under the twin's two stored profiles
(`documented_maintenance`, `nominal`): once at the prior medians (the line), and once per prior draw (the 5-95%
band). Draws are synthesize's own (`draw_unit`: every parameter the twin reads, life derived from k), from
SeedSequence(seed, spawn_key=(i,)). Deterministic, offline; no truth is read.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.backends import load_backend
from kaful.config import SynthesizeSettings
from kaful.figures.drawn import Context, Drawn
from kaful.figures.sources import LoadedLineage
from kaful.figures.spec import MaintenancePart, MaintenanceSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, new_figure
from kaful.phases.p7_synthesize.a_draw_units import draw_unit

COLUMNS = ("machine", "binding", "profile", "t", "median_params", "p05", "p95")
PROFILES = ("nominal", "documented_maintenance")
BACKEND = "progpy"
BAND = (0.05, 0.95)
NOTE = ("derived by simulation: {draws} prior draws per profile (synthesize's draw_unit, seed {seed}) and one run at "
        "the prior medians, on the lineage's verified twin and ParamSet")


@dataclass(frozen=True)
class Curve:
    profile: str
    t: tuple[float, ...]
    median_params: tuple[float, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]


@dataclass(frozen=True)
class PartHealth:
    machine: str
    binding: str
    label: str
    time_unit: str
    rules: tuple[str, ...]
    curves: tuple[Curve, ...]


def _rules(twin: TwinDefinition, binding: str) -> tuple[str, ...]:
    out = []
    for m in sorted(twin.maintenance, key=lambda m: m.id):
        if m.binding == binding:
            unit = twin.time_unit + ("s" if m.interval and round(m.interval) != 1 else "")
            when = f"every {m.interval:,.0f} {unit}" if m.interval else f"on {m.trigger}"
            out.append(f"{m.action} {when} ({m.source})")
    return tuple(out)


def _part(spec: MaintenanceSpec, part: MaintenancePart, ll: LoadedLineage, label: str) -> PartHealth:
    twin = ll.twin()
    ps = ll.load(ll._ref(("parameterize", "param_set")), ParamSet)
    backend = load_backend(BACKEND)
    built = backend.build(twin, ps, ll.contract())
    b = next(x for x in twin.bindings if x.id == part.binding)
    life = ps.get(f"{b.component}.life_nominal").prior.quantile(0.5)
    intervals = [m.interval for m in twin.maintenance if m.binding == b.id and m.interval is not None]
    dt = min([life, *intervals]) / spec.steps_per_shortest
    profiles = {p.name: p for p in twin.profiles}
    horizon = min(spec.lives_shown * life, profiles[PROFILES[0]].segments[-1].end)
    synth = SynthesizeSettings(seed=spec.seed, n_units=spec.draws)
    draws = [draw_unit(ps, twin, synth, i).true_params for i in range(spec.draws)]
    out = f"{b.component}_health"
    curves = []
    for name in PROFILES:
        mid = backend.simulate(built, profiles[name], {}, dt=dt, horizon=horizon, seed=0)
        runs = np.stack([backend.simulate(built, profiles[name], d, dt=dt, horizon=horizon, seed=0).outputs[out]
                         for d in draws])
        lo, hi = (np.quantile(runs, q, axis=0, method="inverted_cdf") for q in BAND)
        curves.append(Curve(profile=name, t=tuple(float(x) for x in mid.times),
                            median_params=tuple(float(x) for x in mid.outputs[out]),
                            low=tuple(float(x) for x in lo), high=tuple(float(x) for x in hi)))
    return PartHealth(machine=ll.spec.label, binding=b.id, label=label, time_unit=twin.time_unit,
                      rules=_rules(twin, b.id), curves=tuple(curves))


def extract(spec: MaintenanceSpec, ctx: Context) -> tuple[PartHealth, ...]:
    return tuple(_part(spec, p, ctx.lineage(p.lineage), ctx.label(p.binding)) for p in spec.parts)


def rows(data: tuple[PartHealth, ...]) -> list[dict[str, Cell]]:
    return [{"machine": p.machine, "binding": p.binding, "profile": c.profile, "t": t, "median_params": m,
             "p05": lo, "p95": hi}
            for p in data for c in p.curves for t, m, lo, hi in zip(c.t, c.median_params, c.low, c.high)]


def _panel(ax: Axes, p: PartHealth, style: Style) -> None:
    for c in p.curves:
        color = style.colors["with_maintenance" if c.profile == "documented_maintenance" else "without_maintenance"]
        name = "with documented maintenance" if c.profile == "documented_maintenance" else "without maintenance"
        ax.fill_between(c.t, c.low, c.high, color=color, alpha=style.band_alpha, lw=0)
        ax.plot(c.t, c.median_params, color=color, lw=1.3, ls="-" if c.profile != "nominal" else "--", label=name)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, max(c.t[-1] for c in p.curves))
    ax.set_ylabel("health")
    ax.set_xlabel(f"time ({p.time_unit})")
    ax.set_title(f"{p.machine} · {p.label}", loc="left")
    ax.set_title("; ".join(p.rules), loc="right", fontsize=style.font_size - 1, color=NOTE_COLOR)


def render(data: tuple[PartHealth, ...], draws: int, style: Style, width: str) -> Figure:
    fig, axes = new_figure(style, width, rows=len(data), sharex=False)
    for ax, p in zip(axes, data):
        _panel(ax, p, style)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle(f"simulated on the twin: lines at the prior medians, bands 5–95% over {draws} prior draws",
                 x=0.995, ha="right", fontsize=style.font_size - 2, color=NOTE_COLOR)
    return fig


def draw(spec: MaintenanceSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, spec.draws, ctx.style, spec.width), columns=COLUMNS, rows=rows(data),
                 notes=(NOTE.format(draws=spec.draws, seed=spec.seed),))
