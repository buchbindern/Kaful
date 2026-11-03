"""Sandbox runner: results, errors, timeouts and isolation."""

import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from kaful.config import SandboxSettings
from kaful.sandbox.runner import TRUNCATION_MARKER, run_sandboxed

FAST = SandboxSettings(timeout_s=20.0, cpu_limit_s=20, max_output_bytes=1024, hash_seed=7)


def _code(machine_dir: Path, body: str, name: str = "gen.py") -> str:
    path = machine_dir / "outputs" / "runs" / "r" / "build" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body))
    return f"outputs/runs/r/build/{name}"


def test_ok_call_returns_json_value(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        import numpy as np
        def run(payload):
            print("noise on stdout")
            return {"total": float(np.sum(payload["xs"])), "n": len(payload["xs"])}
    """)
    result = run_sandboxed(tmp_path, rel, "run", {"xs": [1.0, 2.5]}, settings=FAST)
    assert result.status == "ok", result.stderr
    assert result.value == {"total": 3.5, "n": 2}
    assert "noise on stdout" in result.stdout
    assert result.code_path == rel and len(result.code_sha256) == 64 and result.returncode == 0


def test_exception_is_reported_with_traceback(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        def run(payload):
            raise ZeroDivisionError("rate blew up")
    """)
    result = run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert (result.status, result.error_type, result.error_message) == ("error", "ZeroDivisionError", "rate blew up")
    assert result.traceback is not None and "ZeroDivisionError" in result.traceback


def test_syntax_error_and_missing_entrypoint_are_errors(tmp_path: Path) -> None:
    bad = _code(tmp_path, "def run(payload) return 1\n", "bad.py")
    assert run_sandboxed(tmp_path, bad, "run", None, settings=FAST).error_type == "SyntaxError"
    ok = _code(tmp_path, "def run(payload):\n    return 1\n", "ok.py")
    assert run_sandboxed(tmp_path, ok, "missing", None, settings=FAST).error_type == "AttributeError"


def test_infinite_loop_times_out_and_is_killed(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        import os, pathlib
        def run(payload):
            pathlib.Path(payload).write_text(str(os.getpid()))
            while True:
                pass
    """)
    pid_file = tmp_path / "pid.txt"
    started = time.monotonic()
    result = run_sandboxed(tmp_path, rel, "run", str(pid_file),
                           settings=FAST.model_copy(update={"timeout_s": 1.0}))
    assert result.status == "timeout"
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_exit_without_result_is_a_crash(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        import sys
        def run(payload):
            sys.exit(3)
    """)
    result = run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert (result.status, result.returncode) == ("crashed", 3)


def test_non_json_return_is_bad_output(tmp_path: Path) -> None:
    rel = _code(tmp_path, "def run(payload):\n    return {1, 2}\n")
    assert run_sandboxed(tmp_path, rel, "run", None, settings=FAST).status == "bad_output"


def test_child_environment_is_scrubbed_and_seeded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    rel = _code(tmp_path, """
        import os, sys
        def run(payload):
            return {"env": sorted(os.environ), "seed": os.environ.get("PYTHONHASHSEED"),
                    "cwd": os.getcwd(), "path": sys.path}
    """)
    result = run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert result.status == "ok" and isinstance(result.value, dict)
    env = result.value["env"]
    assert isinstance(env, list) and "ANTHROPIC_API_KEY" not in env and "PYTHONPATH" not in env
    assert result.value["seed"] == "7"
    cwd = str(result.value["cwd"])
    assert "kaful-sandbox-" in cwd and not cwd.startswith(str(tmp_path))
    assert str(tmp_path) not in str(result.value["path"])


def test_network_is_refused_in_the_child(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        import socket
        def run(payload):
            try:
                socket.create_connection(("example.com", 80), timeout=1)
            except OSError as exc:
                return str(exc)
            return "connected"
    """)
    result = run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert result.value == "network access is disabled in the Kaful sandbox"


def test_relative_writes_stay_in_the_throwaway_dir(tmp_path: Path) -> None:
    rel = _code(tmp_path, """
        def run(payload):
            open("stray.txt", "w").write("x")
            return "done"
    """)
    assert run_sandboxed(tmp_path, rel, "run", None, settings=FAST).status == "ok"
    assert not list(tmp_path.rglob("stray.txt")) and not Path("stray.txt").exists()


def test_output_is_truncated(tmp_path: Path) -> None:
    rel = _code(tmp_path, "def run(payload):\n    print('x' * 5000)\n    return None\n")
    result = run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert result.stdout.endswith(TRUNCATION_MARKER) and len(result.stdout) < 1100


@pytest.mark.parametrize("rel", ["../escape.py", "/etc/passwd", "outputs/missing.py"])
def test_paths_must_stay_inside_the_machine_dir(tmp_path: Path, rel: str) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        run_sandboxed(tmp_path, rel, "run", None, settings=FAST)


def test_symlink_escape_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("def run(payload):\n    return 1\n")
    machine_dir = tmp_path / "machine"
    machine_dir.mkdir()
    (machine_dir / "link.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        run_sandboxed(machine_dir, "link.py", "run", None, settings=FAST)


def test_parent_never_imports_generated_code(tmp_path: Path) -> None:
    rel = _code(tmp_path, "def run(payload):\n    return 1\n")
    before = set(sys.modules)
    run_sandboxed(tmp_path, rel, "run", None, settings=FAST)
    assert {m for m in set(sys.modules) - before if "sandboxed" in m or m == "gen"} == set()
