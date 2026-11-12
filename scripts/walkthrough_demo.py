"""
scripts/walkthrough_demo.py
---------------------------
The demo in docs/WALKTHROUGH.md (Part 3). Offline and $0: no API call, nothing written under machines/.

1. tiny_pump end to end: ingest -> understand -> scope -> parameterize -> build in a scratch machines
   directory, with the test suite's scripted LLM answers (tests/harness/kaful_understand.py) and hashing
   embeddings, so the run is exactly what the offline tests exercise. The scratch machine.toml adds one
   [[maintenance]] declaration (the manual's "Replace the cartridge every 2000 operating hours"), the way an
   engineer's review would, so the "with maintenance" profile has something to do.
2. eversys and CNC: the TwinDefinition, ParamSet and SensorContract of the M10 build runs, built by the
   ProgPy backend and simulated at prior medians under the two stored profiles: `nominal` (no maintenance)
   and `documented_maintenance`.

For each machine it saves docs/walkthrough/<machine>_health.png (one row per component: health without
and with maintenance) and prints when each failure event fires.

Usage:
    python -B scripts/walkthrough_demo.py --out <scratch dir>
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kaful import pipeline  # noqa: E402
from kaful.artifacts.manifest import RunManifest  # noqa: E402
from kaful.artifacts.param_set import ParamSet  # noqa: E402
from kaful.artifacts.sensor_contract import SensorContract  # noqa: E402
from kaful.artifacts.twin_definition import TwinDefinition  # noqa: E402
from kaful.backends import load_backend  # noqa: E402
from kaful.backends.base import SimulationResult  # noqa: E402
from kaful.phases import PhaseName  # noqa: E402
from kaful.runs import RunContext, load_artifact  # noqa: E402
from tests.harness.kaful_ingest import FakeEmbeddings  # noqa: E402
from tests.harness.kaful_llm import ScriptedTransport  # noqa: E402
from tests.harness.kaful_understand import MACHINE_ID, ScriptedUnderstanding, make_tiny_pump, understand_settings  # noqa: E402

PLOTS = ROOT / "docs" / "walkthrough"
BUILD_RUNS = {  # the M10 final build runs (docs/milestones/M10.md)
    "eversys_coffee_machine": "20261110T160740Z-de19f9",
    "cnc_spindle": "20261110T160827Z-8c2f37",
}
TINY_PHASES: tuple[PhaseName, ...] = ("ingest", "understand", "scope", "parameterize", "build")
TINY_MAINTENANCE = """
# Walkthrough demo only (scratch copy): p2 "Replace the cartridge every 2000 operating hours", declared because
# scope's naming rule does not link "the cartridge" to "inlet cartridge filter (F1)" (M9 naming gap).
[[maintenance]]
component = "inlet_cartridge_filter_f1"
action = "replace"
interval = 2000
unit = "hour"
"""
# Scripted parameterize.life_judgement replies (5th-95th percentile ranges), one per component.
JUDGED = {
    "impeller": '{"life": {"unit": "hour", "low": 5000, "high": 40000}, '
                '"basis": "centrifugal pump impellers in clean water"}',
    "inlet_cartridge_filter_f1": '{"life": {"unit": "hour", "low": 1500, "high": 12000}, '
                                 '"basis": "cartridge filters on a clean-water supply"}',
}
STEPS_PER_SHORTEST = 200  # time step = the shortest life or maintenance interval / this
ROW_LIVES = 3.0  # each row's x axis spans this many of the component's median lives (capped at the horizon)
WITHOUT, WITH = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 (validated: CVD dE 24.7, contrast >= 3:1)
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#fcfcfb"


def tiny_pump_responder() -> Callable[[dict[str, Any]], str]:
    understanding = ScriptedUnderstanding()

    def respond(request: dict[str, Any]) -> str:
        if "You estimate how long" not in request["system"][0]["text"]:
            return understanding(request)
        text = json.dumps(request["messages"])
        return next(reply for component, reply in sorted(JUDGED.items()) if component in text)

    return respond


def run_tiny_pump(out: Path) -> RunContext:
    make_tiny_pump(out, TINY_MAINTENANCE)
    request = pipeline.RunRequest(machine_id=MACHINE_ID, argv=("run", MACHINE_ID), only=TINY_PHASES, budget_usd=1.0)
    ctx, base = pipeline.start_run(understand_settings(out), request, transport=ScriptedTransport(tiny_pump_responder()),
                                   embedding_transport=FakeEmbeddings())
    pipeline.execute(ctx, TINY_PHASES, base)
    sys.stdout.write(f"tiny_pump run {ctx.manifest.run_id}: {ctx.manifest.status}\n")
    for record in ctx.manifest.phases:
        files = ", ".join(r.path for r in record.artifacts) or "-"
        sys.stdout.write(f"  {record.phase:<13} {record.status:<15} ${record.cost_usd:.4f}  {files}\n")
        for failure in record.failures:
            sys.stdout.write(f"      failure: {failure}\n")
    return ctx


def twin_of(machine_dir: Path, manifest: RunManifest) -> tuple[TwinDefinition, ParamSet, SensorContract]:
    refs = {r.kind: r for p in ("understand", "parameterize", "build") for r in manifest.phase(p).artifacts}
    return (load_artifact(machine_dir, refs["twin_definition"], TwinDefinition),
            load_artifact(machine_dir, refs["param_set"], ParamSet),
            load_artifact(machine_dir, refs["sensor_contract"], SensorContract))


def simulate(twin: TwinDefinition, ps: ParamSet, contract: SensorContract) -> dict[str, SimulationResult]:
    backend = load_backend("progpy")
    built = backend.build(twin, ps, contract)
    lives = [ps.get(f"{b.component}.life_nominal").prior.quantile(0.5) for b in twin.bindings]
    intervals = [m.interval for m in twin.maintenance if m.interval is not None]
    dt = min(lives + intervals) / STEPS_PER_SHORTEST
    horizon = twin.profiles[0].segments[-1].end
    return {p.name: backend.simulate(built, p, {}, dt=dt, horizon=horizon, seed=0) for p in twin.profiles}


def _fmt(t: float | None) -> str:
    return "never (within the horizon)" if t is None else f"{t:,.0f}"


def report(machine: str, twin: TwinDefinition, ps: ParamSet, results: dict[str, SimulationResult]) -> None:
    sys.stdout.write(f"{machine} (time unit {twin.time_unit}, horizon {results['nominal'].times[-1]:,.0f}):\n")
    for b in twin.bindings:
        life = ps.get(f"{b.component}.life_nominal")
        event = f"{b.component}_failure"
        line = (f"  {b.component:<28} life median {life.prior.quantile(0.5):>9,.0f} ({life.source:<13}) "
                f"no maintenance: {_fmt(results['nominal'].event_times[event]):<27} "
                f"documented maintenance: {_fmt(results['documented_maintenance'].event_times[event])}")
        state = f"{b.component}_renewals"
        if state in results["documented_maintenance"].states:
            line += f" ({results['documented_maintenance'].states[state][-1]:.0f} threshold renewals)"
        sys.stdout.write(line + "\n")


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_ylim(-0.03, 1.05)


def plot(machine: str, twin: TwinDefinition, ps: ParamSet, results: dict[str, SimulationResult], path: Path) -> None:
    bindings = sorted(twin.bindings, key=lambda b: b.component)
    fig, axes = plt.subplots(len(bindings), 1, figsize=(8, 2.2 * len(bindings) + 0.6), squeeze=False,
                             facecolor=SURFACE)
    for ax, b in zip(axes[:, 0], bindings, strict=True):
        life = ps.get(f"{b.component}.life_nominal")
        end = min(results["nominal"].times[-1], ROW_LIVES * life.prior.quantile(0.5))
        for name, color, style, label in (("documented_maintenance", WITH, "solid", "with documented maintenance"),
                                          ("nominal", WITHOUT, (0, (4, 2)), "without maintenance")):
            r = results[name]
            keep = r.times <= end
            ax.plot(r.times[keep], r.outputs[f"{b.component}_health"][keep], color=color, linestyle=style,
                    linewidth=1.6, label=label)
        failed = results["nominal"].event_times[f"{b.component}_failure"]
        if failed is not None and failed <= end:
            ax.plot([failed], [0.0], "o", color=WITHOUT, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(f"fails at {failed:,.0f}", (failed, 0.0), xytext=(6, 8), textcoords="offset points",
                        fontsize=8, color=INK)
        rules = [m for m in twin.maintenance if m.binding == b.id]
        how = "; ".join(f"{m.action} " + (f"every {m.interval:,.0f}" if m.interval else
                                          f"on {m.trigger}, inside the model: both profiles") for m in rules)
        how = how or "none documented (the curves coincide)"
        ax.set_title(f"{b.component}: {b.template}, life median {life.prior.quantile(0.5):,.0f} {twin.time_unit}s "
                     f"({life.source})\nmaintenance: {how}", fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, end)
        ax.set_ylabel("health", fontsize=8, color=MUTED)
        _style(ax)
    axes[-1, 0].set_xlabel(f"time ({twin.time_unit}s)", fontsize=8, color=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper left", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"{machine}: health at prior medians, no noise", x=0.99, ha="right", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()), h_pad=1.2)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    sys.stdout.write(f"  plot: {path.relative_to(ROOT)}\n")


def demo_machine(name: str, machine_dir: Path, manifest: RunManifest) -> None:
    twin, ps, contract = twin_of(machine_dir, manifest)
    results = simulate(twin, ps, contract)
    report(name, twin, ps, results)
    plot(name, twin, ps, results, PLOTS / f"{name}_health.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="scratch directory for the tiny_pump run")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    ctx = run_tiny_pump(args.out)
    if ctx.manifest.status != "passed":
        return 1
    demo_machine(MACHINE_ID, ctx.machine_dir, ctx.manifest)
    for name, run_id in BUILD_RUNS.items():
        machine_dir = ROOT / "machines" / name
        manifest = RunManifest.model_validate_json((machine_dir / "outputs" / "runs" / run_id / "manifest.json").read_text())
        demo_machine(name, machine_dir, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
