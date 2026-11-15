"""Run generated code in a subprocess with a timeout (hard rule 7).

Isolation (this guards against buggy generated code, not hostile code; it is not a
security boundary):
- separate interpreter, started with `-B -s -P` and a scrubbed environment, so no
  PYTHONPATH, user site-packages, script or current directory reaches sys.path;
  site-packages (numpy, ProgPy) stay importable;
- the code is loaded inside the child; the parent never imports it, and nothing
  touches sys.path or sys.modules;
- cwd and HOME are a throwaway temporary directory;
- no API keys: the child sees only PATH, HOME, TMPDIR, PYTHONHASHSEED, MPLBACKEND;
- outbound network calls are refused (best effort, inside the child);
- RLIMIT_CPU caps CPU time; RLIMIT_AS (memory) is applied only when configured and
  is unreliable on macOS;
- on timeout the whole process group is killed.
"""

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from kaful.artifacts.base import Identifier, Sha256, StrictModel
from kaful.config import SandboxSettings
from kaful.paths import RelPath, check_rel_path, resolve_in
from kaful.runs import sha256_file
from kaful.sandbox.drivers import DRIVERS, driver_file

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child.py"
RESULT_FILE = "result.json"
TRUNCATION_MARKER = "\n…[truncated]"

SandboxStatus = Literal["ok", "error", "timeout", "crashed", "bad_output"]


class SandboxResult(StrictModel):
    """Outcome of one sandboxed call; saved as data by the caller, never printed."""

    status: SandboxStatus
    value: JsonValue = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None
    duration_s: float
    code_path: RelPath
    code_sha256: Sha256
    entrypoint: Identifier
    target_path: RelPath | None = None  # run_driver: the machine file the driver loaded
    target_sha256: Sha256 | None = None


def _child_env(workdir: Path, settings: SandboxSettings) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONHASHSEED": str(settings.hash_seed),
        "MPLBACKEND": "Agg",
    }


def _limits(settings: SandboxSettings) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (settings.cpu_limit_s, settings.cpu_limit_s))
        if settings.memory_limit_mb is not None:
            limit = settings.memory_limit_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass  # not supported on this platform (macOS)

    return apply


def _truncate(data: bytes, limit: int) -> str:
    text = data[:limit].decode("utf-8", errors="replace")
    return text + TRUNCATION_MARKER if len(data) > limit else text


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _read_result(path: Path) -> dict[str, JsonValue] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def run_sandboxed(
    machine_dir: Path,
    code_path: str,
    entrypoint: str,
    payload: JsonValue,
    *,
    settings: SandboxSettings,
) -> SandboxResult:
    """Call `entrypoint(payload)` from the file at `code_path` (machine-relative) in a
    child interpreter. The payload and the return value must be JSON."""
    rel = check_rel_path(code_path)
    code_file = resolve_in(machine_dir, rel)
    if not code_file.is_file():
        raise FileNotFoundError(f"generated code not found: {rel}")
    base = {"code_path": rel, "code_sha256": sha256_file(code_file), "entrypoint": entrypoint}
    return _spawn(code_file, base, payload, settings)


def run_driver(
    machine_dir: Path,
    driver: str,
    target_path: str,
    entrypoint: str,
    payload: dict[str, JsonValue],
    *,
    settings: SandboxSettings,
) -> SandboxResult:
    """Call `entrypoint(payload)` from a pinned driver (kaful.sandbox.drivers) in a child interpreter; the
    driver loads the machine file `target_path` (machine-relative), given to it as `payload["target_file"]`.
    Only a pinned driver whose content still has its pinned hash runs (DriverNotAllowed otherwise)."""
    code_file = driver_file(driver)
    rel = check_rel_path(target_path)
    target = resolve_in(machine_dir, rel)
    if not target.is_file():
        raise FileNotFoundError(f"generated code not found: {rel}")
    base = {"code_path": DRIVERS[driver].path, "code_sha256": DRIVERS[driver].sha256, "entrypoint": entrypoint,
            "target_path": rel, "target_sha256": sha256_file(target)}
    return _spawn(code_file, base, {**payload, "target_file": str(target)}, settings)


def _spawn(code_file: Path, base: Mapping[str, object], payload: JsonValue, settings: SandboxSettings) -> SandboxResult:
    """Run the child on `code_file`; it refuses to load the file unless it still has `base["code_sha256"]`."""
    with tempfile.TemporaryDirectory(prefix="kaful-sandbox-") as tmp:
        workdir = Path(tmp)
        request = {
            "code_file": str(code_file),
            "code_sha256": base["code_sha256"],
            "entrypoint": base["entrypoint"],
            "payload": payload,
            "result_path": str(workdir / RESULT_FILE),
        }
        started = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, "-B", "-s", "-P", str(CHILD_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=_child_env(workdir, settings),
            start_new_session=True,
            preexec_fn=_limits(settings),
        )
        timed_out = False
        try:
            out, err = proc.communicate(json.dumps(request).encode(), timeout=settings.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            out, err = proc.communicate()
        common = {
            **base,
            "stdout": _truncate(out, settings.max_output_bytes),
            "stderr": _truncate(err, settings.max_output_bytes),
            "returncode": proc.returncode,
            "duration_s": time.monotonic() - started,
        }
        if timed_out:
            return SandboxResult.model_validate({**common, "status": "timeout"})
        return _interpret(_read_result(workdir / RESULT_FILE), proc.returncode, common)


def _interpret(
    result: dict[str, JsonValue] | None, returncode: int, common: dict[str, object]
) -> SandboxResult:
    if result is None:
        status: SandboxStatus = "crashed" if returncode != 0 else "bad_output"
        return SandboxResult.model_validate({**common, "status": status})
    if result.get("ok") is True:
        return SandboxResult.model_validate({**common, "status": "ok", "value": result.get("value")})
    status = "bad_output" if result.get("kind") == "bad_output" else "error"
    return SandboxResult.model_validate({
        **common,
        "status": status,
        "error_type": result.get("type"),
        "error_message": result.get("message"),
        "traceback": result.get("traceback"),
    })
