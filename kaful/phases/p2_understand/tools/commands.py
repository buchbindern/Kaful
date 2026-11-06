"""`kaful understand golden-review <machine>` (offline; no API calls).

Writes outputs/understand_review/<machine>.html (gitignored: it embeds manual pages), the
page for filling in tests/fixtures/understand/golden/<machine>.toml.
"""

import argparse
import sys

from kaful.config import REPO_ROOT, KafulSettings
from kaful.phases.p2_understand.tools.golden_review import write_review

GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "understand" / "golden"
REVIEW_DIR = REPO_ROOT / "outputs" / "understand_review"
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    understand = commands.add_parser("understand", help="understand tools (offline)").add_subparsers(
        dest="understand_command", required=True)
    review = understand.add_parser("golden-review", help="write the golden-label review page for a machine")
    review.add_argument("machine")


def understand_golden_review(args: argparse.Namespace, settings: KafulSettings) -> int:
    golden = GOLDEN_DIR / f"{args.machine}.toml"
    if not golden.is_file():
        raise ValueError(f"no golden label file {golden.relative_to(REPO_ROOT)}")
    path = write_review(REVIEW_DIR, golden, settings.machine_dir(args.machine), settings.ingest)
    sys.stdout.write(f"wrote {path.relative_to(REPO_ROOT)} (local only: it embeds manual pages)\n"
                     f"save the downloaded file as {golden.relative_to(REPO_ROOT)}\n")
    return EXIT_OK
