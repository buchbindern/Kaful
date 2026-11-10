"""Understand tools (offline; no API calls).

- `kaful understand golden-review <machine>` writes outputs/understand_review/<machine>.html
  (gitignored: it embeds manual pages), the page for filling in
  tests/fixtures/understand/golden/<machine>.toml.
- `kaful understand score <machine> [--run RUN_ID]` scores a run's understand artifacts against
  those labels and saves the score to <machine>/outputs/understand_scores/<run_id>.json, with
  the disagreements as a table to adjudicate (<run_id>.md).
- `kaful understand review <machine> [--run RUN_ID]` and `kaful understand review-apply <machine> --run RUN_ID`:
  the onboarding review for an engineer (tools/onboarding_review), with the run's scope decisions when the
  run has a Scope (M8).
"""

import argparse
import re
import sys
from pathlib import Path

from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.manifest import RunManifest
from kaful.artifacts.scope import Scope
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.config import REPO_ROOT, KafulSettings
from kaful.phases import PhaseName
from kaful.phases.p2_understand.tools.golden_review import write_review
from kaful.artifacts.document_index import DocumentIndex
from kaful.phases.p2_understand.tools import onboarding_review
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
    onboard = understand.add_parser("review", help="write the onboarding review page and record for a run")
    onboard.add_argument("machine")
    onboard.add_argument("--run", help="run id (default: the latest run whose understand phase passed)")
    apply = understand.add_parser("review-apply", help="turn a filled review record into machine.toml declarations")
    apply.add_argument("machine")
    apply.add_argument("--run", required=True)


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


def _cells(line: str) -> list[str]:
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]]


def filled_cells(table: Path) -> dict[tuple[str, ...], tuple[str, str, str]]:
    """The proposal, unsure mark and verdict already written in a table, by its first five cells.

    Reads the older layout (verdict only) too."""
    if not table.exists():
        return {}
    out = {}
    for line in table.read_text(encoding="utf-8").splitlines():
        cells = _cells(line) if line.startswith("| ") else []
        if len(cells) in (6, 8) and cells[0] not in ("kind", "---") and any(cells[5:]):
            rest = cells[5:]
            out[tuple(cells[:5])] = (rest[0], rest[1], rest[2]) if len(rest) == 3 else ("", "", rest[0])
    return out


def disagreement_table(result: GoldenScore, filled: dict[tuple[str, ...], tuple[str, str, str]] | None = None) -> str:
    """The disagreements as a Markdown table, for a person to adjudicate (run right, or label right?).

    Claude pre-fills `proposed` with a one-line reason and marks the ones it is unsure of; Natalie
    reviews those and spot-checks the rest (2026-11-08). Cells already filled are kept."""
    def cell(text: str) -> str:
        return text.replace("|", "\\|")
    rows = []
    for d in result.disagreements:
        key = (d.kind, d.item, d.run, d.golden, d.ref)
        kept = (filled or {}).get(key, ("", "", ""))
        rows.append("| " + " | ".join(cell(c) for c in (*key, *kept)) + " |")
    return (f"# {result.machine_id} run {result.run_id}: disagreements with the golden labels\n\n"
            f"Golden file sha256 {result.golden_sha256[:12]}. `proposed` is Claude's verdict and reason, `unsure` "
            "marks the ones to review; verdict: `run` (fix the label) or `label`.\n\n"
            "| kind | item | run says | labels say | run cites | proposed | unsure | verdict |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


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
    write_text_atomic(table, disagreement_table(result, filled_cells(table)))
    sys.stdout.write(_summary(result) + f"wrote {path.relative_to(machine_dir)} and {table.name}\n")
    return EXIT_OK


def understand_review(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    manifest = _run(list_manifests(machine_dir), args.run)
    refs = {r.kind: r for r in manifest.phase(PHASE).artifacts}
    model = load_artifact(machine_dir, refs["machine_model"], MachineModel)
    index = load_artifact(machine_dir, model.inputs[0], DocumentIndex)
    scoped = [r for r in manifest.phase("scope").artifacts if r.kind == "scope"]
    scope = load_artifact(machine_dir, scoped[0], Scope) if scoped else None
    page, record = onboarding_review.write_review(
        machine_dir, model, load_artifact(machine_dir, refs["sensor_contract"], SensorContract),
        load_artifact(machine_dir, refs["understand_report"], UnderstandReport), index, scope, manifest.run_id)
    sys.stdout.write(f"wrote {page.relative_to(machine_dir)}; review record {record.relative_to(machine_dir)}\n")
    return EXIT_OK


def understand_review_apply(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    record = machine_dir.joinpath(*onboarding_review.REVIEWS_DIR, f"{args.run}.toml")
    if not record.is_file():
        raise ValueError(f"no review record {record.relative_to(machine_dir)}")
    sys.stdout.write("".join(f"{line}\n" for line in onboarding_review.apply_review(machine_dir, record)))
    return EXIT_OK
