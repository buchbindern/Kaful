"""Sandbox child bootstrap. Runs in a separate interpreter; stdlib only.

Protocol: the parent writes one JSON request to stdin:
    {"code_file": str, "entrypoint": str, "payload": <json>, "result_path": str}
This script blocks outbound network calls, loads the code file as a module, calls
`entrypoint(payload)`, and writes one JSON object to `result_path`:
    {"ok": true, "value": <json>}
    {"ok": false, "kind": "error" | "bad_output", "type": str, "message": str, "traceback": str}
The result goes to a file, not stdout, so prints in the generated code cannot corrupt it.
It must not import kaful.
"""

import importlib.util
import json
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, NoReturn


def _refuse_network(*args: object, **kwargs: object) -> NoReturn:
    raise OSError("network access is disabled in the Kaful sandbox")


def _block_network() -> None:
    """Best effort: refuse outbound connections and name lookups."""
    for name in ("connect", "connect_ex"):
        setattr(socket.socket, name, _refuse_network)
    for name in ("create_connection", "getaddrinfo"):
        setattr(socket, name, _refuse_network)


def _call(request: dict[str, Any]) -> Any:
    spec = importlib.util.spec_from_file_location("kaful_sandboxed_code", request["code_file"])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {request['code_file']}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, request["entrypoint"])(request["payload"])


def _failure(kind: str, exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "kind": kind,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def main() -> int:
    request = json.load(sys.stdin)
    _block_network()
    try:
        value = _call(request)
    except Exception as exc:  # noqa: BLE001 — any error in generated code is reported
        result = _failure("error", exc)
    else:
        try:
            json.dumps(value)
            result = {"ok": True, "value": value}
        except (TypeError, ValueError) as exc:
            result = _failure("bad_output", exc)
    path = Path(request["result_path"])
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    tmp.replace(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
