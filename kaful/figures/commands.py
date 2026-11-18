"""Figures CLI (offline; no API calls; reads saved artifacts only).

- `kaful figures build [--spec PATH] [--only F2 F3 ...]` draws the spec's figures into the spec's folder
  (default `docs/figures/figures.toml`) and writes `figures_manifest.json` there.
"""

import argparse
import sys
from pathlib import Path

from kaful.config import REPO_ROOT, KafulSettings
from kaful.figures.build import build

DEFAULT_SPEC = Path("docs") / "figures" / "figures.toml"
EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("figures", help="figures from saved artifacts (offline)").add_subparsers(
        dest="figures_command", required=True)
    b = sub.add_parser("build", help="draw the spec's figures (SVG, PNG, CSV) and the figures manifest")
    b.add_argument("--spec", type=Path, default=None, help=f"figure spec (default {DEFAULT_SPEC})")
    b.add_argument("--only", nargs="+", default=[], help="figure ids to rebuild (others keep their entries)")


def figures_build(args: argparse.Namespace, settings: KafulSettings) -> int:
    spec = args.spec if args.spec is not None else REPO_ROOT / DEFAULT_SPEC
    manifest = build(spec, settings, tuple(args.only))
    for r in manifest.figures:
        if not args.only or r.id in args.only:
            sys.stdout.write(f"{r.id}: {', '.join(sorted(r.outputs))} ({len(r.inputs)} inputs)"
                             f"{'; ' + r.selection if r.selection else ''}\n")
    return EXIT_OK
