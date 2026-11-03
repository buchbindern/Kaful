"""Hard rule 14: artifact bytes do not depend on PYTHONHASHSEED."""

import os
import subprocess
import sys

from kaful.config import REPO_ROOT

SEEDS = ("0", "1", "12345")


def _artifacts_json(seed: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    env["PYTHONHASHSEED"] = seed
    done = subprocess.run(
        [sys.executable, "-B", "-m", "tests.harness.kaful_factories"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    return done.stdout


def test_artifact_json_is_identical_across_hash_seeds() -> None:
    outputs = {seed: _artifacts_json(seed) for seed in SEEDS}
    assert outputs["0"].count('"schema_version"') == 11  # 10 artifacts + the manifest
    assert len(set(outputs.values())) == 1


def test_factories_really_vary_dict_order_across_seeds() -> None:
    """Guard against a vacuous pass: set iteration order differs between these seeds."""
    script = "print(list({'wear_rate', 'temp_exponent', 'timeout_s', 'cpu_limit_s', 'ingest', 'understand'}))"
    orders = {
        subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed},
                       capture_output=True, text=True, check=True).stdout
        for seed in SEEDS
    }
    assert len(orders) > 1
