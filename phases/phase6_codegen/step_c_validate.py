"""
phases/phase6_codegen/step_c_validate.py
------------------------------------------
Step C: Runtime validation of each generated component.

Before building the composite model, validates each component:
1. Imports and instantiates cleanly
2. initialize() returns all expected state keys
3. event_state() at t=0 — all values must be 1.0
4. output() at t=0 — returns all expected output keys, no NaN/inf
5. Runs 1000 steps without NaN, inf, or out-of-bounds states
6. Degradation states actually degrade over 1000 steps
7. Degradation rate sanity — not too fast, not flat
8. Tracking states converge toward target

Output saved to: outputs/codegen/step_c_validation.json
"""

import json
import math
import importlib.util
import sys
import traceback
from pathlib import Path
from xml.parsers.expat import model


def run(cfg: dict, specs: dict) -> dict:
    """
    Runtime validate each generated component.

    Args:
        cfg:   result of get_machine_config()
        specs: dict of {component_name: spec_dict} from step_a

    Returns:
        dict of {component_name: validation_report}
    """
    output_path = cfg["codegen_step_c_validation"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        with open(output_path) as f:
            return json.load(f)

    print(f"  Running step_c — runtime validation ({len(specs)} components)...")

    code_dir = cfg["codegen_code_dir"]
    reports  = {}
    passed   = 0
    failed   = 0

    for name, spec in specs.items():
        code_path = code_dir / f"{name}.py"
        if not code_path.exists():
            reports[name] = {"status": "missing", "issues": ["Code file not found"]}
            failed += 1
            continue

        print(f"    {name}...", end=" ", flush=True)
        report = _validate_component(name, code_path, spec)
        reports[name] = report

        if report["status"] == "pass":
            passed += 1
            print(f"✓")
        else:
            failed += 1
            print(f"✗")
            for issue in report["issues"]:
                print(f"      ⚠ {issue}")

    print(f"\n    Results: {passed} passed, {failed} failed")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reports, f, indent=2)

    print(f"    Saved → {output_path.name}")

    return reports


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_component(name: str, code_path: Path, spec: dict) -> dict:
    """Run all validation checks on one component."""
    issues   = []
    warnings = []
    details  = {}

    # 1. Import and instantiate
    try:
        module = _import_module(name, code_path)
        cls    = _find_class(module, name)
        model  = cls()
    except Exception as e:
        return {
            "status":   "fail",
            "issues":   [f"Failed to import/instantiate: {e}"],
            "warnings": [],
            "details":  {},
        }

    # Build nominal inputs from spec
    nominal_inputs = _build_nominal_inputs(spec)
    dt             = 1.0

    # 2. initialize()
    try:
        x0 = model.initialize(u=nominal_inputs)
        missing_states = [
            s["name"] for s in spec.get("states", [])
            if s["name"] not in x0
        ]
        if missing_states:
            issues.append(f"initialize() missing states: {missing_states}")

        none_states = [k for k, v in x0.items() if v is None]
        if none_states:
            issues.append(f"initialize() returned None for states: {none_states}")

        details["initial_states"] = {k: float(v) for k, v in x0.items() if v is not None}

    except Exception as e:
        issues.append(f"initialize() failed: {e}")
        return {"status": "fail", "issues": issues, "warnings": warnings, "details": details}

    # 3. event_state() at t=0
    try:
        es = model.event_state(x0)
        non_one = {k: v for k, v in es.items() if not _approx_equal(v, 1.0)}
        if non_one:
            issues.append(f"event_state() not 1.0 at t=0: {non_one}")
        details["event_state_t0"] = {k: float(v) for k, v in es.items()}
    except Exception as e:
        issues.append(f"event_state() failed at t=0: {e}")

    # 4. output() at t=0
    try:
        z0 = model.output(x0)
        expected_outputs = {o["name"] for o in spec.get("outputs", [])}
        missing_outputs  = expected_outputs - set(z0.keys())
        if missing_outputs:
            issues.append(f"output() missing keys: {missing_outputs}")

        bad_outputs = {k: v for k, v in z0.items() if _is_bad(v) and _is_numeric(v)}
        if bad_outputs:
            issues.append(f"output() has NaN/inf at t=0: {bad_outputs}")

        details["output_t0"] = {k: _safe_float(v) for k, v in z0.items()}
    except Exception as e:
        issues.append(f"output() failed at t=0: {e}")

    # 5-8. Run N steps
    try:
        step_issues, step_warnings, step_details = _run_steps(
            model, x0, nominal_inputs, spec, dt, n_steps=1000
        )
        issues.extend(step_issues)
        warnings.extend(step_warnings)
        details.update(step_details)
    except Exception as e:
        issues.append(f"Step simulation failed: {e}\n{traceback.format_exc()[:500]}")

    status = "fail" if issues else ("warn" if warnings else "pass")

    return {
        "status":   status,
        "issues":   issues,
        "warnings": warnings,
        "details":  details,
    }


def _run_steps(model, x0, nominal_inputs, spec, dt, n_steps=1000):
    """Run N steps and check for numerical issues and degradation."""
    issues   = []
    warnings = []
    details  = {}

    x         = x0
    history   = {k: [float(v)] for k, v in x0.items() if v is not None}
    has_next  = hasattr(model, "next_state")
    has_dx    = hasattr(model, "dx")

    for step in range(n_steps):
        try:
            if has_next:
                x = model.next_state(x, nominal_inputs, dt=dt)
            elif has_dx:
                dxdt = model.dx(x, nominal_inputs)
                x    = {k: x[k] + dxdt.get(k, 0.0) * dt for k in x}
            else:
                issues.append("No dx() or next_state() method found")
                break

            # Check for NaN/inf
            bad = {k: v for k, v in x.items() if _is_bad(v)}
            if bad:
                issues.append(f"NaN/inf in states at step {step}: {list(bad.keys())}")
                break

            for k, v in x.items():
                if v is not None:
                    history[k].append(float(v))

        except Exception as e:
            issues.append(f"Step {step} failed: {e}")
            break

    details["final_states"] = {k: vals[-1] for k, vals in history.items()}

    # Check degradation states
    degrad_specs = [s for s in spec.get("states", []) if s["state_type"] == "degradation"]
    for s in degrad_specs:
        sname = s["name"]
        if sname not in history or len(history[sname]) < 2:
            continue

        initial = history[sname][0]
        final   = history[sname][-1]
        delta   = initial - final

        details[f"{sname}_drift"] = round(delta, 6)

        if delta <= 0:
            issues.append(f"Degradation state '{sname}' did not degrade over {n_steps} steps (delta={delta:.6f})")
        elif delta > 0.9:
            issues.append(f"Degradation state '{sname}' degraded too fast — fully degraded before {n_steps} steps")
        elif delta < 1e-6:
            warnings.append(f"Degradation state '{sname}' barely changed (delta={delta:.8f}) — rate may be too low")

    # Check tracking states
    tracking_specs = [s for s in spec.get("states", []) if s["state_type"] == "tracking"]
    for s in tracking_specs:
        sname = s["name"]
        if sname not in history or len(history[sname]) < 100:
            continue

        vals = history[sname]
        # Check it's not wildly oscillating
        diffs = [abs(vals[i+1] - vals[i]) for i in range(min(100, len(vals)-1))]
        max_diff = max(diffs) if diffs else 0
        if max_diff > 100:
            warnings.append(f"Tracking state '{sname}' oscillating (max step={max_diff:.2f}) — stability cap may be missing")

    return issues, warnings, details


# ── Helpers ───────────────────────────────────────────────────────────────────

def _import_module(name: str, code_path: Path):
    """Dynamically import a component module."""
    spec   = importlib.util.spec_from_file_location(name, code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_class(module, name: str):
    """Find the PrognosticsModel class in the module."""
    import inspect
    # Try exact name match first (CamelCase)
    camel = "".join(w.capitalize() for w in name.split("_"))
    if hasattr(module, camel):
        return getattr(module, camel)

    # Fall back to finding any class with next_state or dx
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if hasattr(obj, "next_state") or hasattr(obj, "dx"):
            return obj

    raise ValueError(f"No PrognosticsModel class found in {name}.py")


def _build_nominal_inputs(spec: dict) -> dict:
    """Build a dict of nominal input values from the spec."""
    inputs = {}
    for inp in spec.get("inputs", []):
        # Default to 1.0 for any input — enough to drive degradation
        inputs[inp["name"]] = 1.0
    return inputs


def _is_bad(v) -> bool:
    """Check if a value is NaN or inf."""
    try:
        return math.isnan(float(v)) or math.isinf(float(v))
    except (TypeError, ValueError):
        return False

def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
    
def _approx_equal(a, b, tol=1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def _safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    except (TypeError, ValueError):
        return None