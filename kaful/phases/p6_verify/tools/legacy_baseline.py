"""The legacy baseline (PLAN M11, for M15): the verify checks run on the legacy twins, through the sandbox.

`kaful verify legacy-baseline <machine>` probes each component the legacy triage kept as a full model
(outputs/triage/step_c_full_components.json) and the legacy composite, each in the sandbox via the pinned driver
legacy_probe (nominal inputs of 1.0, dt = 1, `verify.legacy_steps` steps, as legacy step C ran them), and
scores the traces here. Nothing legacy is imported into this interpreter (rule 7); the code is only read.

Checks per component (the legacy spec gives state types and bounds):
- loads: the file imports, the class is found, initialize() works;
- starts_healthy: every degradation state starts at 1.0; every event_state is 1.0 and no event is met at t=0;
- finite_bounded: every state and output at t=0 is a finite number, stepping never raises, degradation states
  stay in [0, 1], and every state stays within its spec min/max;
- monotone: degradation states never rise and accumulation states never fall (legacy twins have no maintenance);
- event_agreement: threshold_met holds exactly when event_state is 0;
- tracking_converges: over the last `legacy_settle_fraction` of the run, no step moves a tracking state by more
  than `legacy_settle_tolerance` of its span (skipped without tracking states);
- life_consistency: skipped, not assessable (legacy twins have no calibrated life);
- provenance: fails (legacy parameters carry no source; hard rule 3).
Two rates (Natalie 2026-11-13): the dynamic pass rate (every check but life and provenance; the comparable number
for M15) and the full pass rate (every check; 0 for legacy by construction).

Writes outputs/legacy_verification/legacy_baseline.json. Not used during a run (a tool). No API calls.
"""

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from kaful.artifacts.base import Fraction, NonEmptyStr, Sha256, StrictModel
from kaful.artifacts.verification import CheckResult, CheckStatus
from kaful.config import KafulSettings, VerifySettings
from kaful.sandbox.drivers import DRIVERS
from kaful.sandbox.runner import SandboxResult, run_driver

TRIAGE = "outputs/triage/step_c_full_components.json"
SPECS, COMPONENTS, COMPOSITE_FILE = "outputs/codegen/specs", "outputs/codegen/components", "outputs/codegen/composite_model.py"
OUT = ("outputs", "legacy_verification", "legacy_baseline.json")
DYNAMIC = ("event_agreement", "finite_bounded", "loads", "monotone", "starts_healthy", "tracking_converges")
TOL = 1e-6  # legacy step C's own tolerance for "starts at 1.0"
NOT_ASSESSABLE = "not assessable: legacy twins have no calibrated life"
NO_PROVENANCE = "legacy parameters have no source or reference (hard rule 3)"


class LegacyComponent(StrictModel):
    component: NonEmptyStr
    results: tuple[CheckResult, ...]
    sandbox_status: NonEmptyStr
    duration_s: float

    @property
    def dynamic_pass(self) -> bool:
        return all(r.status != "fail" for r in self.results if r.check in DYNAMIC)

    @property
    def full_pass(self) -> bool:
        return all(r.status == "pass" or (r.check == "tracking_converges" and r.status == "skipped")
                   for r in self.results)


class LegacyBaseline(StrictModel):
    machine_id: NonEmptyStr
    created_at: datetime
    probe_sha256: Sha256
    steps: int
    components: tuple[LegacyComponent, ...]
    composite: tuple[CheckResult, ...]
    dynamic_pass_rate: Fraction
    full_pass_rate: Fraction


def _result(check: str, target: str, problems: Sequence[str], measured: dict[str, float] | None = None,
            skipped: str = "") -> CheckResult:
    status: CheckStatus = "skipped" if skipped else "fail" if problems else "pass"
    return CheckResult(check=check, target=target, status=status, measured=measured or {},
                       message=skipped or "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more"
                                                                    if len(problems) > 3 else ""))


def _first_none(values: Sequence[float | None]) -> int | None:
    return next((i for i, v in enumerate(values) if v is None), None)


def _starts_healthy(trace: dict[str, Any], degradation: Sequence[str]) -> list[str]:
    out = [f"{s} starts at {trace['states'][s][0]}, not 1.0" for s in degradation
           if s in trace["states"] and (trace["states"][s][0] is None or abs(trace["states"][s][0] - 1.0) > TOL)]
    out += [f"event_state {e} is {v[0]} at t=0" for e, v in sorted(trace["event_state"].items())
            if v and (v[0] is None or abs(v[0] - 1.0) > TOL)]
    out += [f"event {e} already met at t=0" for e, v in sorted(trace["threshold_met"].items()) if v and v[0]]
    return out


def _finite_bounded(trace: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    out = []
    if trace["error"] is not None:
        out.append(f"step {trace['error']['step']} raised {trace['error']['message']}")
    outputs = trace["outputs_t0"]
    if "error" in outputs:
        out.append(f"output() at t=0 raised {outputs['error']}")
    out += [f"output {k} is not a finite number at t=0" for k, v in outputs.items() if k != "error" and v is None]
    for s in spec.get("states", []):
        values = trace["states"].get(s["name"], [])
        bad = _first_none(values)
        if bad is not None:
            out.append(f"{s['name']} is not finite from step {bad}")
            continue
        low = 0.0 if s.get("state_type") == "degradation" else s.get("min_value")
        high = 1.0 if s.get("state_type") == "degradation" else s.get("max_value")
        if values and low is not None and min(values) < low - TOL * max(1.0, abs(low)):
            out.append(f"{s['name']} falls to {min(values):.6g} < {low:g}")
        if values and high is not None and max(values) > high + TOL * max(1.0, abs(high)):
            out.append(f"{s['name']} rises to {max(values):.6g} > {high:g}")
    return out


def _monotone(trace: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    out = []
    for s in spec.get("states", []):
        values = [v for v in trace["states"].get(s["name"], []) if v is not None]
        sign = {"degradation": -1.0, "accumulation": 1.0}.get(s.get("state_type", ""))
        if sign is None:
            continue
        wrong = next((i for i in range(1, len(values)) if sign * (values[i] - values[i - 1]) < -1e-12), None)
        if wrong is not None:
            out.append(f"{s['name']} ({s['state_type']}) moves the wrong way at step {wrong}")
    return out


def _agreement(trace: dict[str, Any]) -> list[str]:
    out = []
    for e, met in sorted(trace["threshold_met"].items()):
        es = trace["event_state"].get(e, [])
        bad = next((i for i, (m, v) in enumerate(zip(met, es)) if v is not None and ((m and v > TOL) or
                                                                                   (not m and v <= 0.0))), None)
        if bad is not None:
            out.append(f"{e}: threshold_met {met[bad]} while event_state is {es[bad]:.6g} at step {bad}")
    return out


def _converges(trace: dict[str, Any], tracking: Sequence[str], v: VerifySettings) -> list[str]:
    out = []
    for name in tracking:
        values = [x for x in trace["states"].get(name, []) if x is not None]
        if len(values) < 10:
            continue
        span = max(values) - min(values)
        tail = values[int(len(values) * (1.0 - v.legacy_settle_fraction)):]
        moves = max((abs(b - a) for a, b in zip(tail, tail[1:])), default=0.0)
        if span > 0 and moves > v.legacy_settle_tolerance * span:
            out.append(f"{name} still moves {moves:.6g} per step at the end (span {span:.6g})")
    return out


def score_component(name: str, spec: dict[str, Any], trace: dict[str, Any], v: VerifySettings) -> tuple[CheckResult, ...]:
    by_type = {t: [s["name"] for s in spec.get("states", []) if s.get("state_type") == t]
               for t in ("degradation", "tracking")}
    fixed = [_result("life_consistency", name, [], skipped=NOT_ASSESSABLE),
             _result("provenance", name, [f"{len(spec.get('parameters', []))} {NO_PROVENANCE}"])]
    if trace["error"] is not None and trace["error"]["where"] == "load":
        loads = _result("loads", name, [trace["error"]["message"]])
        rest = [_result(c, name, [], skipped="not run: the component did not load") for c in DYNAMIC if c != "loads"]
        return tuple(sorted([loads, *rest, *fixed], key=lambda r: r.check))
    results = [_result("loads", name, []),
               _result("starts_healthy", name, _starts_healthy(trace, by_type["degradation"])),
               _result("finite_bounded", name, _finite_bounded(trace, spec)),
               _result("monotone", name, _monotone(trace, spec)),
               _result("event_agreement", name, _agreement(trace)),
               _result("tracking_converges", name, _converges(trace, by_type["tracking"], v),
                       skipped="" if by_type["tracking"] else "no tracking state"), *fixed]
    return tuple(sorted(results, key=lambda r: r.check))


def score_composite(trace: dict[str, Any]) -> tuple[CheckResult, ...]:
    target = "composite"
    if trace["error"] is not None and trace["error"]["where"] == "load":
        return (_result("loads", target, [trace["error"]["message"]]),)
    nonfinite = [f"{k} is not finite from step {_first_none(v)}" for k, v in sorted(trace["states"].items())
                 if _first_none(v) is not None]
    run_error = [f"step {trace['error']['step']} raised {trace['error']['message']}"] if trace["error"] else []
    return (_result("event_agreement", target, _agreement(trace)),
            _result("finite_bounded", target, run_error + nonfinite), _result("loads", target, []),
            _result("starts_healthy", target, _starts_healthy(trace, ())))


def _probe(machine_dir: Path, target: str, payload: dict[str, JsonValue],
           settings: KafulSettings) -> tuple[SandboxResult, dict[str, Any]]:
    result = run_driver(machine_dir, "legacy_probe", target, "probe", payload, settings=settings.sandbox)
    if result.status == "ok" and isinstance(result.value, dict):
        return result, dict(result.value)
    why = result.error_message or result.status
    return result, {"states": {}, "event_state": {}, "threshold_met": {}, "outputs_t0": {},
                    "error": {"step": None, "where": "load", "message": f"sandbox {result.status}: {why}"}}


def baseline(machine_dir: Path, machine_id: str, settings: KafulSettings, now: datetime) -> LegacyBaseline:
    v = settings.verify
    names = [c["name"] for c in json.loads((machine_dir / TRIAGE).read_text(encoding="utf-8"))]
    components = []
    for name in sorted(names):
        spec = json.loads((machine_dir / SPECS / f"{name}.json").read_text(encoding="utf-8"))
        result, trace = _probe(machine_dir, f"{COMPONENTS}/{name}.py", {
            "mode": "component", "name": name, "inputs": [i["name"] for i in spec.get("inputs", [])],
            "steps": v.legacy_steps, "dt": 1.0}, settings)
        components.append(LegacyComponent(component=name, results=score_component(name, spec, trace, v),
                                          sandbox_status=result.status, duration_s=result.duration_s))
    _, trace = _probe(machine_dir, COMPOSITE_FILE, {"mode": "composite", "name": "composite", "inputs": [],
                                                    "steps": v.legacy_steps, "dt": 1.0}, settings)
    n = max(len(components), 1)
    return LegacyBaseline(
        machine_id=machine_id, created_at=now, probe_sha256=DRIVERS["legacy_probe"].sha256, steps=v.legacy_steps,
        components=tuple(components), composite=score_composite(trace),
        dynamic_pass_rate=sum(c.dynamic_pass for c in components) / n,
        full_pass_rate=sum(c.full_pass for c in components) / n)
