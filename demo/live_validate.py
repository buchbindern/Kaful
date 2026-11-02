"""
demo/live_validate.py
---------------------
Live demo: re-run phase 6 step C runtime validation on generated components.

Runs the same _validate_component() the pipeline uses, but bypasses the
"already done" cache and writes nothing to disk. No API calls.

Usage (needs the env with progpy, e.g. conda `pipeline311`):
    python demo/live_validate.py                          # all eversys components
    python demo/live_validate.py eversys_coffee_machine ceramic_grinders
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_machine_config
from phases.phase6_codegen.step_c_validate import _validate_component


def main():
    machine_id = sys.argv[1] if len(sys.argv) > 1 else "eversys_coffee_machine"
    only       = sys.argv[2:]
    cfg        = get_machine_config(machine_id)

    # Only the components that made it through triage (specs/ may hold excluded ones)
    modeled = [c["name"] for c in json.loads(cfg["triage_step_c_full_components"].read_text())]
    specs   = {n: json.loads((cfg["codegen_specs_dir"] / f"{n}.json").read_text()) for n in modeled}
    names   = only or modeled

    print(f"\nRuntime validation — {machine_id} ({len(names)} generated components)")
    print("Checks: loads, starts healthy, 1,000 steps without NaN/inf, wears, sane wear rate\n")

    start = time.time()
    for name in names:
        t0 = time.time()
        r  = _validate_component(name, cfg["codegen_code_dir"] / f"{name}.py", specs[name])
        mark = {"pass": "✓ PASS", "warn": "! WARN", "fail": "✗ FAIL"}[r["status"]]
        print(f"  {mark}  {name:<22} {time.time() - t0:5.2f}s")

        d = r["details"]
        for state in (k.removesuffix("_drift") for k in d if k.endswith("_drift")):
            x0, x1 = d["initial_states"][state], d["final_states"][state]
            print(f"            {state:<40} {x0:.3f} → {x1:.3f} after 1,000 steps")
        for msg in r["issues"] + r["warnings"]:
            print(f"            ⚠ {msg}")

    print(f"\nDone in {time.time() - start:.1f}s — nothing written to disk.\n")


if __name__ == "__main__":
    main()
