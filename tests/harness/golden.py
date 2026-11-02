"""
tests/harness/golden.py
-----------------------
Golden run for the reference machine (M2): re-derive eversys's phase 6–7
outputs that involve no LLM call, from frozen LLM checkpoints, and compare
them with a reviewed snapshot.

Scratch copy = tracked eversys outputs + the gitignored inputs packed under
tests/fixtures/golden/eversys/inputs/ (component code, simulator CSVs), minus
the derived outputs listed in DERIVED. The run uses an empty fixture store, so
any LLM call fails the test. machines/ is never written.
"""

import fnmatch
import gzip
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from config import MACHINES_DIR, ROOT_DIR
from tests.harness.fake_anthropic import ReplayAnthropic
from tests.harness.fixtures import FixtureStore
from tests.harness.pipeline import FIXTURES_DIR, offline_pipeline, run_legacy

MACHINE_ID = "eversys_coffee_machine"
GOLDEN_SEED = 0  # seeds Python's and NumPy's global RNGs (phase 7 Monte Carlo is unseeded)
GOLDEN_ENV = {"PYTHONHASHSEED": "0", "ANTHROPIC_API_KEY": "offline-test-key",
              "OPENAI_API_KEY": "offline-test-key", "ANONYMIZED_TELEMETRY": "False"}
GOLDEN_DIR = FIXTURES_DIR / "golden" / "eversys"
INPUTS_DIR = GOLDEN_DIR / "inputs"
EXPECTED_DIR = GOLDEN_DIR / "expected"
TOLERANCES_PATH = GOLDEN_DIR / "tolerances.toml"

# Gitignored files the run needs (packed gzipped into INPUTS_DIR).
INPUT_GLOBS = ("outputs/codegen/components/*.py", "outputs/simulate/*.csv")
# Outputs the golden run re-derives (deleted from the scratch copy first).
DERIVED = ("outputs/codegen/step_c_validation.json", "outputs/codegen/step_d_validation.json",
           "outputs/estimate")
# Copied from the machine dir: everything else that is tracked and needed.
COPY_IGNORE = ("__pycache__", "ledger.jsonl", "estimate", "*.csv", "manual", "rag")
# Compared outputs, relative to the machine dir.
COMPARED_GLOBS = ("outputs/codegen/step_c_validation.json", "outputs/codegen/step_d_validation.json",
                  "outputs/estimate/*/step_*.json")
# Legacy rule-12 violation (absolute path in generated code); fixed in the copy only.
COMPOSITE = "outputs/codegen/composite_model.py"
COMPONENTS = "outputs/codegen/components"


# ── Inputs ────────────────────────────────────────────────────────────────────

def pack_inputs(machine_dir: Path = MACHINES_DIR / MACHINE_ID, dest: Path = INPUTS_DIR) -> list[Path]:
    """Gzip the gitignored inputs into the fixture tree (mtime 0, so bytes are stable)."""
    written = []
    for pattern in INPUT_GLOBS:
        for src in sorted(machine_dir.glob(pattern)):
            out = dest / (src.relative_to(machine_dir).as_posix() + ".gz")
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                       filename="") as gz:
                gz.write(src.read_bytes())
            written.append(out)
    return written


def build_scratch(machines_dir: Path, source: Path = MACHINES_DIR / MACHINE_ID) -> list[str]:
    """Assemble the scratch machine; returns a log of every change made to the copy."""
    dest = machines_dir / MACHINE_ID
    shutil.copytree(source, dest, ignore=_ignore)
    for gz in sorted(INPUTS_DIR.rglob("*.gz")):
        out = dest / gz.relative_to(INPUTS_DIR).as_posix().removesuffix(".gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.decompress(gz.read_bytes()))
    log = []
    for rel in DERIVED:
        path = dest / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        log.append(f"removed derived output {rel}")
    log.append(_repoint_composite(dest))
    return log


def _ignore(directory: str, names: list[str]) -> set[str]:
    """Skip local-only files; codegen/components comes from the packed inputs instead."""
    skipped = set(shutil.ignore_patterns(*COPY_IGNORE)(directory, names))
    if Path(directory).name == "codegen":
        skipped |= {"components"} & set(names)
    return skipped


def _repoint_composite(machine_dir: Path) -> str:
    """Point composite_model.py's hardcoded sys.path entry at the scratch components."""
    path = machine_dir / COMPOSITE
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith("sys.path.insert(0,")]
    if len(hits) != 1:
        raise RuntimeError(f"expected one sys.path.insert line in {COMPOSITE}, found {len(hits)}")
    old = lines[hits[0]].strip()
    lines[hits[0]] = f"sys.path.insert(0, {str(machine_dir / COMPONENTS)!r})\n"
    path.write_text("".join(lines), encoding="utf-8")
    return f"legacy violation (rule 12, fix in M10): {COMPOSITE} had `{old}`; repointed in scratch copy"


# ── Run and collect ───────────────────────────────────────────────────────────

def run_golden(machines_dir: Path, seed: int) -> dict[str, Any]:
    """Run phases 6–7 offline with global RNGs seeded; returns the compared outputs."""
    random.seed(seed)
    np.random.seed(seed)
    client = ReplayAnthropic(FixtureStore(machines_dir / "no-llm-fixtures"))
    with offline_pipeline(machines_dir, client):
        run_legacy(MACHINE_ID, "--from", "phase6")
    if client.calls:
        raise AssertionError(f"golden run made LLM calls: {client.calls}")
    return collect(machines_dir / MACHINE_ID)


def run_golden_subprocess(machines_dir: Path, seed: int = GOLDEN_SEED) -> dict[str, Any]:
    """run_golden in a fresh interpreter with PYTHONHASHSEED pinned.

    ProgPy builds CompositeModel state/input/event order from sets, so phase 7's
    noise assignment depends on the hash seed; no in-process seed can fix that.
    """
    result = machines_dir.parent / "golden_outputs.json"
    env = {**os.environ, **GOLDEN_ENV, "PYTHONPATH": str(ROOT_DIR)}
    proc = subprocess.run([sys.executable, "-B", "-m", "tests.harness.golden",
                           str(machines_dir), str(seed), str(result)],
                          cwd=ROOT_DIR, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"golden subprocess failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
    return json.loads(result.read_text())


def collect(machine_dir: Path) -> dict[str, Any]:
    out = {}
    for pattern in COMPARED_GLOBS:
        for path in sorted(machine_dir.glob(pattern)):
            out[path.relative_to(machine_dir).as_posix()] = json.loads(path.read_text())
    return out


def load_expected(directory: Path = EXPECTED_DIR) -> dict[str, Any]:
    return {gz.relative_to(directory).as_posix().removesuffix(".gz"): json.loads(gzip.decompress(gz.read_bytes()))
            for gz in sorted(directory.rglob("*.json.gz"))}


def write_expected(outputs: dict[str, Any], directory: Path = EXPECTED_DIR) -> None:
    for rel, data in outputs.items():
        out = directory / (rel + ".gz")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=1, sort_keys=True).encode()
        with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
            gz.write(payload)


# ── Compare ───────────────────────────────────────────────────────────────────

def load_tolerances(path: Path = TOLERANCES_PATH) -> list[dict[str, Any]]:
    """Ordered [[rule]] tables: {files, keys, rel, abs, ignore}; first match wins."""
    with open(path, "rb") as f:
        return tomllib.load(f)["rule"]


def compare(expected: dict[str, Any], actual: dict[str, Any],
            rules: list[dict[str, Any]]) -> list[str]:
    """Differences beyond tolerance, as 'file:key.path: detail' strings."""
    diffs = [f"{f}: missing from run" for f in expected.keys() - actual.keys()]
    diffs += [f"{f}: not in snapshot" for f in actual.keys() - expected.keys()]
    for f in sorted(expected.keys() & actual.keys()):
        _walk(f, "", expected[f], actual[f], rules, diffs)
    return diffs


def _rule(file: str, key: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for r in rules:
        if fnmatch.fnmatch(file, r.get("files", "*")) and fnmatch.fnmatch(key, r.get("keys", "*")):
            return r
    return {"rel": 0.0, "abs": 0.0}


def _walk(file: str, key: str, exp: Any, act: Any, rules: list[dict[str, Any]],
          diffs: list[str]) -> None:
    rule = _rule(file, key, rules)
    if rule.get("ignore"):
        return
    if isinstance(exp, dict) and isinstance(act, dict):
        for k in sorted(exp.keys() | act.keys(), key=str):
            if k not in exp or k not in act:
                diffs.append(f"{file}:{key}.{k}: only in {'run' if k in act else 'snapshot'}")
            else:
                _walk(file, f"{key}.{k}", exp[k], act[k], rules, diffs)
    elif isinstance(exp, list) and isinstance(act, list):
        if len(exp) != len(act):
            diffs.append(f"{file}:{key}: length {len(exp)} → {len(act)}")
        for i, (e, a) in enumerate(zip(exp, act)):
            _walk(file, f"{key}[{i}]", e, a, rules, diffs)
    elif _is_number(exp) and _is_number(act):
        if not math.isclose(exp, act, rel_tol=rule.get("rel", 0.0), abs_tol=rule.get("abs", 0.0)):
            diffs.append(f"{file}:{key}: {exp!r} → {act!r}")
    elif exp != act or isinstance(exp, bool) != isinstance(act, bool):  # True == 1 in Python
        diffs.append(f"{file}:{key}: {str(exp)[:80]!r} → {str(act)[:80]!r}")


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


if __name__ == "__main__":  # worker for run_golden_subprocess: <machines_dir> <seed> <out.json>
    from tests.harness.netblock import blocked_network
    with blocked_network():
        outputs = run_golden(Path(sys.argv[1]), int(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(outputs))
