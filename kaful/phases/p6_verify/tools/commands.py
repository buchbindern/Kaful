"""Verify tools (offline; no API calls).

- `kaful verify legacy-baseline <machine>` runs the verify checks on the machine's legacy twin through the
  sandbox (tools/legacy_baseline) and writes <machine>/outputs/legacy_verification/legacy_baseline.json, the
  pass-rate baseline M15 compares against.
"""

import argparse
import sys

from kaful.config import KafulSettings
from kaful.phases.p6_verify.tools.legacy_baseline import OUT, LegacyBaseline, baseline
from kaful.runs import dump_json, now_utc, write_text_atomic

EXIT_OK = 0


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sub = commands.add_parser("verify", help="verify tools (offline)").add_subparsers(
        dest="verify_command", required=True)
    legacy = sub.add_parser("legacy-baseline", help="run the verify checks on a legacy twin, in the sandbox")
    legacy.add_argument("machine")


def legacy_summary(machine_id: str, result: LegacyBaseline) -> str:
    rows = [f"  {'✓' if c.dynamic_pass else '✗'} {c.component}: "
            + ("; ".join(f"{r.check}: {r.message}" for r in c.results if r.status == "fail" and r.check != "provenance")
               or "dynamic checks pass") for c in result.components]
    composite = "; ".join(f"{r.check}: {r.message}" for r in result.composite if r.status == "fail") or "passes"
    return (f"{machine_id}: dynamic pass rate {result.dynamic_pass_rate:.0%} "
            f"({sum(c.dynamic_pass for c in result.components)}/{len(result.components)}), full pass rate "
            f"{result.full_pass_rate:.0%}; composite {composite}\n" + "\n".join(rows) + "\n")


def verify_legacy_baseline(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    result = baseline(machine_dir, args.machine, settings, now_utc())
    path = machine_dir.joinpath(*OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, dump_json(result))
    sys.stdout.write(legacy_summary(args.machine, result) + f"wrote {path.relative_to(machine_dir)}\n")
    return EXIT_OK
