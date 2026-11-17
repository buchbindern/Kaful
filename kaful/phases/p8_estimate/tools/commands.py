"""Estimate tools (offline; no API calls).

- `kaful estimate score <machine> [--run RUN_ID]` scores an estimate run on synthetic data against the hidden truth
  (tools/score_synthetic) and saves it to <machine>/outputs/estimate_scores/<run_id>.json.
- `kaful estimate whatif <machine> --renew-at BINDING=FACTOR [--run RUN_ID]` forecasts what a delayed renewal does to
  the parts it feeds (tools/whatif) and saves it to <machine>/outputs/whatif/<run_id>__<binding>_x<factor>.json.
"""

import argparse
import sys

from kaful.artifacts.base import ArtifactRef
from kaful.artifacts.forecast import ForecastSet
from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.synthetic import SyntheticDataset
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.config import KafulSettings
from kaful.phases import PhaseName
from kaful.artifacts.sensor_contract import SensorContract
from kaful.phases.p8_estimate.tools.score_synthetic import EstimateScore, score
from kaful.phases.p8_estimate.tools.whatif import WhatIfReport, whatif
from kaful.runs import dump_json, list_manifests, load_artifact, write_text_atomic

SCORES_DIR = ("outputs", "estimate_scores")
WHATIF_DIR = ("outputs", "whatif")
EXIT_OK = 0
SCORED = ("passed",)


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("estimate", help="estimate tools (offline)").add_subparsers(
        dest="estimate_command", required=True)
    scored = sub.add_parser("score", help="score an estimate run on synthetic data against its hidden truth")
    scored.add_argument("machine")
    scored.add_argument("--run", help="run id (default: the latest run whose estimate phase passed)")
    what = sub.add_parser("whatif", help="forecast a delayed renewal's effect on the parts it feeds (offline)")
    what.add_argument("machine")
    what.add_argument("--renew-at", required=True, action="append", metavar="BINDING=FACTOR",
                      help="replace this threshold-renewed binding at FACTOR x its limit (repeatable)")
    what.add_argument("--run", help="run id (default: the latest run whose estimate phase passed)")


def _pick(manifests: list[RunManifest], run_id: str | None) -> RunManifest:
    passed = [m for m in manifests if m.phase("estimate").status in SCORED]
    picked = [m for m in passed if m.run_id == run_id] if run_id else passed[-1:]
    if not picked:
        raise ValueError(f"no run {run_id!r} with a passed estimate phase" if run_id
                         else "no run with a passed estimate phase")
    return picked[0]


def summary(result: EstimateScore) -> str:
    lines = [f"{result.machine_id} {result.run_id} ({result.disclaimer})"]
    for s in result.scores:
        lines.append(f"== {s.estimator}")
        for c in s.comparisons:
            parts = [f"{c.kind:14s} {c.target:44s} n={c.units:2d}"]
            if c.estimator_mean is not None:
                parts.append(f"est {c.estimator_mean:.4f}")
            if c.open_loop_mean is not None:
                parts.append(f"open-loop {c.open_loop_mean:.4f} wins {c.wins}")
            if c.prior_mean is not None:
                parts.append(f"prior {c.prior_mean:.4f} closer {c.closer}")
            if c.covers is not None:
                parts.append(f"90% covers {c.covers}")
            lines.append("   " + "  ".join(parts))
        lines.append("   forecast preview (5-95% covers the next limit time): "
                     + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in s.forecast_cover.items()))
    for j in result.joined:
        lines.append(f"== joined through a connection: {j.binding} (width PF/open-loop {j.width_ratio}, "
                     f"n={j.width_pairs}; spread share {j.spread_share})")
        for pc in j.priors:
            lines.append(f"   posterior = prior? {pc.param}: KS mean {pc.ks_mean:.4f} max {pc.ks_max:.4f} critical "
                         f"{pc.ks_critical:.4f}; rejected {pc.rejected}/{pc.units} at alpha {pc.alpha:g}")
    return "\n".join(lines) + "\n"


def estimate_score(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    manifest = _pick(list_manifests(machine_dir), args.run)

    def ref(phase: PhaseName, kind: str) -> ArtifactRef:
        return next(r for r in manifest.phase(phase).artifacts if r.kind == kind)

    fs = load_artifact(machine_dir, ref("estimate", "forecast_set"), ForecastSet)
    dataset = load_artifact(machine_dir, ref("synthesize", "synthetic_dataset"), SyntheticDataset)
    twin = load_artifact(machine_dir, ref("verify", "twin_definition"), TwinDefinition)
    ps = load_artifact(machine_dir, ref("parameterize", "param_set"), ParamSet)
    result = score(machine_dir, manifest.run_id, fs, dataset, twin, ps, settings.scoring)
    notes = manifest.phase("estimate").notes
    result = result.model_copy(update={"runtime_note": next((n for n in notes if "forecasts in" in n), "")})
    path = machine_dir.joinpath(*SCORES_DIR, f"{manifest.run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(result))
    sys.stdout.write(summary(result) + f"wrote {path.relative_to(machine_dir)}\n")
    return EXIT_OK


def _renew_at(values: list[str]) -> dict[str, float]:
    out = {}
    for v in values:
        name, _, factor = v.partition("=")
        if not name or not factor:
            raise ValueError(f"--renew-at {v!r}: expected BINDING=FACTOR")
        out[name] = float(factor)
    return dict(sorted(out.items()))


def whatif_summary(r: WhatIfReport) -> str:
    lines = [f"{r.machine_id} {r.run_id} renew_at {r.renew_at}", r.disclaimer, r.from_new.conclusion,
             r.per_unit.conclusion, *(lr.conclusion for lr in r.long_run.values()), "Would narrow this: " + r.narrowing]
    return "\n".join(lines) + "\n"


def estimate_whatif(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    manifest = _pick(list_manifests(machine_dir), args.run)

    def ref(phase: PhaseName, kind: str) -> ArtifactRef:
        return next(r for r in manifest.phase(phase).artifacts if r.kind == kind)

    renew_at = _renew_at(args.renew_at)
    report = whatif(manifest.run_id, load_artifact(machine_dir, ref("verify", "twin_definition"), TwinDefinition),
                    load_artifact(machine_dir, ref("parameterize", "param_set"), ParamSet),
                    load_artifact(machine_dir, ref("understand", "sensor_contract"), SensorContract),
                    load_artifact(machine_dir, ref("synthesize", "synthetic_dataset"), SyntheticDataset),
                    settings, machine_dir, renew_at, settings.whatif.known_halfwidth)
    label = "_".join(f"{b}_x{f:g}" for b, f in renew_at.items())
    path = machine_dir.joinpath(*WHATIF_DIR, f"{manifest.run_id}__{label}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(report))
    sys.stdout.write(whatif_summary(report) + f"wrote {path.relative_to(machine_dir)}\n")
    return EXIT_OK
