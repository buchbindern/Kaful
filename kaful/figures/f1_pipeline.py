"""F1: the pipeline, PDF to twin to forecast, with one machine's real numbers in every box.

Nine phases in execution order (two rows, read left to right then right to left). Each box: the phase, its job, and
what this lineage's artifact holds (counts read from the artifacts and the evaluation report), with the phase's
cold-run API cost range. Nothing in a box is typed by hand: the CSV lists every number and where it came from.
"""

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.verification import VerificationReport
from kaful.figures.drawn import Context, Drawn
from kaful.figures.sources import LoadedLineage
from kaful.figures.spec import PipelineSpec, Style
from kaful.figures.style import NOTE_COLOR, Cell, mark_synthetic, new_figure
from kaful.phases import PHASE_ORDER

COLUMNS = ("phase", "job", "line", "cost_low_usd", "cost_high_usd")
JOBS = {"ingest": "read the documents", "understand": "parts, flows, faults, sensors",
        "scope": "which parts get a model", "parameterize": "every number a prior", "build": "templates → twin",
        "verify": "checks that gate", "synthesize": "test data from the twin", "estimate": "filter + forecast",
        "evaluate": "grounding, calibration, cost"}
PER_ROW = 5
BOX_W, BOX_H, GAP_X, GAP_Y = 1.18, 0.8, 0.16, 0.3
FIGURE_HEIGHT_IN = 2.75
CENT = 0.01


@dataclass(frozen=True)
class Box:
    phase: str
    job: str
    lines: tuple[str, ...]
    cost: tuple[float, float] | None


def _lines(ll: LoadedLineage) -> dict[str, tuple[str, ...]]:
    r = ll.report
    index = ll.load(ll._ref(("ingest", "document_index")), DocumentIndex)
    model = ll.load(ll._ref(("understand", "machine_model")), MachineModel)
    contract = ll.contract()
    scope = ll.load(ll._ref(("scope", "scope")), Scope)
    ps = ll.load(ll._ref(("parameterize", "param_set")), ParamSet)
    twin = ll.twin()
    ver = ll.load(ll._ref(("verify", "verification_report")), VerificationReport)
    ds, fs = ll.dataset(), ll.forecast_set()
    full = sum(d.decision == "full_model" for d in scope.decisions)
    src = r.grounding.params_by_source if r.grounding else {}
    top = sorted(((n, s) for s, n in src.items() if n), reverse=True)[:2]
    tq, cal = r.twin_quality, r.calibration
    verdicts = ", ".join(f"{v.coverage:.0%}" for v in cal.verdicts) if cal else "–"
    return {
        "ingest": (f"{len(index.documents)} document{'s' if len(index.documents) != 1 else ''}",
                   f"{len(index.pages)} pages, {len(index.chunks)} chunks"),
        "understand": (f"{len(model.components)} components", f"{len(model.fault_modes)} fault modes",
                       f"{len(contract.channels)} sensor channels"),
        "scope": (f"{full} of {len(scope.decisions)} components", "get a full model"),
        "parameterize": (f"{len(ps.parameters)} parameters", ", ".join(f"{n} {s.replace('_', ' ')}" for n, s in top)),
        "build": (f"{len(twin.bindings)} degradation models", f"{len(twin.maintenance)} maintenance rules"),
        "verify": (f"{len(ver.results)} checks", f"dynamic pass {tq.dynamic_pass_rate:.0%}" if tq and
                   tq.dynamic_pass_rate is not None else "–"),
        "synthesize": (f"{len(ds.units)} synthetic units", "truth hidden from estimators"),
        "estimate": (f"{len(fs.forecasts)} forecasts", "particle filter, open-loop"),
        "evaluate": ("90% coverage, observed", f"parts: {verdicts}"),
    }


def extract(spec: PipelineSpec, ctx: Context) -> tuple[str, tuple[Box, ...], tuple[float, float]]:
    ll = ctx.lineage(spec.lineage)
    lines = _lines(ll)
    cost = {p.phase: (p.cold_low_usd, p.cold_usd) for p in ll.report.cost.by_phase}
    boxes = tuple(Box(phase=p, job=JOBS[p], lines=lines[p], cost=cost.get(p)) for p in PHASE_ORDER)
    return ll.spec.label, boxes, (ll.report.cost.cold_low_usd, ll.report.cost.cold_usd)


def rows(data: tuple[str, tuple[Box, ...], tuple[float, float]]) -> list[dict[str, Cell]]:
    return [{"phase": b.phase, "job": b.job, "line": line, "cost_low_usd": b.cost[0] if b.cost else None,
             "cost_high_usd": b.cost[1] if b.cost else None} for b in data[1] for line in b.lines]


def _xy(i: int) -> tuple[float, float]:
    row, k = divmod(i, PER_ROW)
    col = k if row == 0 else PER_ROW - 1 - k
    return col * (BOX_W + GAP_X), -row * (BOX_H + GAP_Y)


def _box(ax: Axes, b: Box, x: float, y: float, style: Style) -> None:
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=style.colors["cone"], ec="none"))
    ax.text(x + 0.06, y + BOX_H - 0.07, b.phase, fontsize=style.font_size, fontweight="bold", va="top")
    ax.text(x + 0.06, y + BOX_H - 0.22, b.job, fontsize=style.font_size - 2, va="top", color=NOTE_COLOR)
    ax.text(x + 0.06, y + BOX_H - 0.38, "\n".join(b.lines), fontsize=style.font_size - 1.5, va="top",
            linespacing=1.3)
    if b.cost is not None:
        lo, hi = b.cost
        text = "$0" if hi == 0 else "<$0.01" if hi < CENT else (f"${hi:.2f}" if abs(hi - lo) < CENT / 2
                                                                else f"${lo:.2f}–{hi:.2f}")
        ax.text(x + BOX_W - 0.05, y + BOX_H - 0.08, text, fontsize=style.font_size - 2, ha="right", va="top",
                color=style.colors["particle_filter"])


def render(data: tuple[str, tuple[Box, ...], tuple[float, float]], style: Style, width: str) -> Figure:
    label, boxes, total = data
    fig, (ax,) = new_figure(style, width)
    fig.set_figheight(FIGURE_HEIGHT_IN)
    for i, b in enumerate(boxes):
        _box(ax, b, *_xy(i), style)
        if i + 1 < len(boxes):
            (x0, y0), (x1, y1) = _xy(i), _xy(i + 1)
            if y0 == y1:
                a, c = ((x0 + BOX_W, y0 + BOX_H / 2), (x1, y1 + BOX_H / 2)) if x1 > x0 else \
                    ((x0, y0 + BOX_H / 2), (x1 + BOX_W, y1 + BOX_H / 2))
            else:
                a, c = (x0 + BOX_W / 2, y0), (x1 + BOX_W / 2, y1 + BOX_H)
            ax.add_patch(FancyArrowPatch(a, c, arrowstyle="-|>", mutation_scale=9, color=NOTE_COLOR, lw=1.0))
    ax.set_xlim(-0.05, PER_ROW * (BOX_W + GAP_X) - GAP_X + 0.05)
    ax.set_ylim(-(BOX_H + GAP_Y) - 0.05, BOX_H + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{label}: manual (PDF) → twin → forecast · cold run ${total[0]:.2f}–{total[1]:.2f} in API calls",
                 loc="left")
    mark_synthetic(ax, style.model_copy(update={"synthetic_label": "estimate/evaluate on synthetic data"}))
    return fig


def draw(spec: PipelineSpec, ctx: Context) -> Drawn:
    data = extract(spec, ctx)
    return Drawn(figure=render(data, ctx.style, spec.width), columns=COLUMNS, rows=rows(data))
