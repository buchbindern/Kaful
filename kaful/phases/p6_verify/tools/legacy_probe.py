"""Sandbox driver for the legacy baseline (M11): loads one legacy component file, or the legacy composite, and
steps it the way legacy phase 6 step C did (nominal inputs of 1.0, dt = 1), returning the traces as JSON.

It runs only inside the sandbox child, pinned by path and hash in kaful.sandbox.drivers, so it must not import
kaful (stdlib and numpy only). It checks nothing itself: tools/legacy_baseline scores the traces with the
verify checks. Payload: target_file (set by run_driver), mode ("component" or "composite"), name (the
component's snake_case name), inputs (input names), steps, dt.
"""

import importlib.util
import inspect
import math
from pathlib import Path
from typing import Any


def _load(path: str) -> Any:
    """The probe never touches the import path (rule 7); a legacy composite adds its own components folder."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_class(module: Any, name: str) -> Any:
    camel = "".join(w.capitalize() for w in name.split("_"))
    if hasattr(module, camel):
        return getattr(module, camel)
    for _, obj in sorted(inspect.getmembers(module, inspect.isclass)):
        if hasattr(obj, "next_state") or hasattr(obj, "dx"):
            return obj
    raise ValueError(f"no PrognosticsModel class in {name}.py")


def _number(v: Any) -> float | None:
    """None for a value that is not a finite number (JSON has no NaN)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _step(model: Any, x: Any, u: Any, dt: float) -> Any:
    if hasattr(model, "next_state"):
        return model.next_state(x, u, dt=dt)
    dxdt = model.dx(x, u)
    return {k: x[k] + dxdt.get(k, 0.0) * dt for k in x}


def _record(trace: dict[str, Any], model: Any, x: Any) -> None:
    for k in sorted(x.keys()):
        trace["states"].setdefault(k, []).append(_number(x[k]))
    for k, v in sorted(model.event_state(x).items()):
        trace["event_state"].setdefault(k, []).append(_number(v))
    for k, v in sorted(model.threshold_met(x).items()):
        trace["threshold_met"].setdefault(k, []).append(bool(v))


def _setup(payload: dict[str, Any]) -> tuple[Any, Any]:
    module = _load(payload["target_file"])
    if payload["mode"] == "composite":
        model = module.composite_model
        return model, module.future_loading_eqn(0.0)
    model = _model_class(module, payload["name"])()
    return model, {name: 1.0 for name in payload["inputs"]}


def probe(payload: dict[str, Any]) -> dict[str, Any]:
    trace: dict[str, Any] = {"states": {}, "event_state": {}, "threshold_met": {}, "outputs_t0": {}, "error": None}
    try:
        model, u = _setup(payload)
        x = model.initialize(u=u)
    except Exception as exc:  # noqa: BLE001 - a legacy file that does not load is a result, not a crash
        trace["error"] = {"step": None, "where": "load", "message": f"{type(exc).__name__}: {exc}"}
        return trace
    try:
        trace["outputs_t0"] = {k: _number(v) for k, v in sorted(model.output(x).items())}
    except Exception as exc:  # noqa: BLE001 - legacy step C recorded this and went on stepping
        trace["outputs_t0"] = {"error": f"{type(exc).__name__}: {exc}"}
    step = 0
    try:
        _record(trace, model, x)
        for step in range(1, int(payload["steps"]) + 1):
            x = _step(model, x, u, float(payload["dt"]))
            _record(trace, model, x)
            if any(_number(x[k]) is None for k in x.keys()):
                break
    except Exception as exc:  # noqa: BLE001 - recorded with the step it happened at
        trace["error"] = {"step": step, "where": "run", "message": f"{type(exc).__name__}: {exc}"}
    return trace
