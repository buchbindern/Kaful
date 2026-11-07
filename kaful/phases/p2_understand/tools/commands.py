"""Understand tools (offline; no API calls).

- `kaful understand golden-review <machine>` writes outputs/understand_review/<machine>.html
  (gitignored: it embeds manual pages), the page for filling in
  tests/fixtures/understand/golden/<machine>.toml.
- `kaful understand score <machine> [--run RUN_ID]` scores a run's understand artifacts against
  those labels and saves the score to <machine>/outputs/understand_scores/<run_id>.json, with
  the disagreements as a table to adjudicate (<run_id>.md).
"""

import argparse
import sys
from pathlib import Path

from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.config import REPO_ROOT, KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p2_understand.tools.golden_review import write_review
from kaful.phases.p2_understand.tools.golden_score import GoldenScore, score
from kaful.runs import dump_json, list_manifests, load_artifact, write_text_atomic

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "understand" / "golden"
REVIEW_DIR = REPO_ROOT / "outputs" / "understand_review"
SCORES_DIR = ("outputs", "understand_scores")
PHASE: PhaseName = "understand"
SCORED = ("passed", "reused")  # phase statuses whose artifacts can be scored
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    understand = commands.add_parser("understand", help="understand tools (offline)").add_subparsers(
        dest="understand_command", required=True)
    review = understand.add_parser("golden-review", help="write the golden-label review page for a machine")
    review.add_argument("machine")
    scored = understand.add_parser("score", help="score a run's understand artifacts against the golden labels")
    scored.add_argument("machine")
    scored.add_argument("--run", help="run id (default: the latest run whose understand phase passed)")


def understand_golden_review(args: argparse.Namespace, settings: KafulSettings) -> int:
    golden = _golden(args.machine)
    path = write_review(REVIEW_DIR, golden, settings.machine_dir(args.machine), settings.ingest)
    sys.stdout.write(f"wrote {path.relative_to(REPO_ROOT)} (local only: it embeds manual pages)\n"
                     f"save the downloaded file as {golden.relative_to(REPO_ROOT)}\n")
    return EXIT_OK


def _golden(machine: str) -> Path:
    golden = GOLDEN_DIR / f"{machine}.toml"
    if not golden.is_file():
        raise ValueError(f"no golden label file {golden.relative_to(REPO_ROOT)}")
    return golden


def _run(manifests: list[RunManifest], run_id: str | None) -> RunManifest:
    passed = [m for m in manifests if m.phase(PHASE).status in SCORED]
    picked = [m for m in passed if m.run_id == run_id] if run_id else passed[-1:]
    if not picked:
        raise ValueError(f"no run {run_id!r} with a passed understand phase" if run_id
                         else "no run with a passed understand phase")
    return picked[0]


def _summary(result: GoldenScore) -> str:
    def pr(name: str, p: float | None, r: float | None, extra: str = "") -> str:
        return f"{name}: precision {p}, recall {r}{extra}\n"
    c, k, f = result.channels, result.components, result.fault_modes
    return (pr("channels", c.precision, c.recall, f", roles {c.role_accuracy}, secondary {c.secondary_found}")
            + pr("components", k.precision, k.recall, f", parents {k.parent_accuracy}")
            + pr("fault modes", f.precision, f.recall)
            + f"unmatched (resolve by hand): {len(c.unmatched)} channels, {len(k.unmatched)} components, "
              f"{len(f.unmatched)} fault modes\n"
            + f"golden examples stored as values: {len(result.values.examples_stored)}\n"
            + f"disagreements to adjudicate: {len(result.disagreements)}\n")


def disagreement_table(result: GoldenScore) -> str:
    """The disagreements as a Markdown table, for a person to adjudicate (run right, or label right?)."""
    def cell(text: str) -> str:
        return text.replace("|", "\\|")
    rows = [f"| {cell(d.kind)} | {cell(d.item)} | {cell(d.run)} | {cell(d.golden)} | {cell(d.ref)} | |"
            for d in result.disagreements]
    return (f"# {result.machine_id} run {result.run_id}: disagreements with the golden labels\n\n"
            f"Golden file sha256 {result.golden_sha256[:12]}. Verdict: `run` (fix the label) or `label`.\n\n"
            "| kind | item | run says | labels say | run cites | verdict |\n|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")


def understand_score(args: argparse.Namespace, settings: KafulSettings) -> int:
    golden, machine_dir = _golden(args.machine), settings.machine_dir(args.machine)
    manifest = _run(list_manifests(machine_dir), args.run)
    refs = {r.kind: r for r in manifest.phase(PHASE).artifacts}
    result = score(golden, load_artifact(machine_dir, refs["machine_model"], MachineModel),
                   load_artifact(machine_dir, refs["sensor_contract"], SensorContract),
                   load_artifact(machine_dir, refs["understand_report"], UnderstandReport), settings.understand)
    path = machine_dir.joinpath(*SCORES_DIR, f"{manifest.run_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(result))
    table = path.with_suffix(".md")
    write_text_atomic(table, disagreement_table(result))
    sys.stdout.write(_summary(result) + f"wrote {path.relative_to(machine_dir)} and {table.name}\n")
    return EXIT_OK
