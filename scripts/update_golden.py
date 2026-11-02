"""
scripts/update_golden.py
------------------------
Refresh the eversys golden fixtures (M2). Each refresh is an approved step
(CLAUDE.md rule 9): never run it to make a failing golden test pass.

    inputs    re-pack the gitignored golden inputs (component code, simulator CSVs)
              from machines/eversys_coffee_machine into tests/fixtures/golden/eversys/inputs/
    expected  run phases 6–7 offline on a scratch copy and write the snapshot to
              tests/fixtures/golden/eversys/expected/

Usage:
    python -B scripts/update_golden.py inputs
    python -B scripts/update_golden.py expected --seed 0
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.harness.golden import (  # noqa: E402
    GOLDEN_DIR, GOLDEN_SEED, build_scratch, pack_inputs, run_golden_subprocess, write_expected)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("what", choices=["inputs", "expected"])
    p.add_argument("--seed", type=int, default=GOLDEN_SEED)
    args = p.parse_args()

    if args.what == "inputs":
        written = pack_inputs()
        print(f"packed {len(written)} inputs → {GOLDEN_DIR.relative_to(ROOT)}/inputs")
        return

    # The worker subprocess blocks the network and pins PYTHONHASHSEED (golden.GOLDEN_ENV).
    with tempfile.TemporaryDirectory() as tmp:
        machines_dir = Path(tmp) / "machines"
        machines_dir.mkdir()
        for line in build_scratch(machines_dir):
            print(f"[golden] {line}")
        outputs = run_golden_subprocess(machines_dir, args.seed)
    write_expected(outputs)
    print(f"wrote {len(outputs)} snapshot files → {GOLDEN_DIR.relative_to(ROOT)}/expected (seed {args.seed})")


if __name__ == "__main__":
    main()
