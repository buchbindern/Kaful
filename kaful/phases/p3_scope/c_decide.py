"""Step c: one decision per component, by rule, then the machine.toml [[scope]] overrides (M8, 2026-11-09).

- full_model: G and (O or L): a documented gradual fault mode, and either a channel that observes it or a
  documented life (stated, a life model, or declared) to calibrate it.
- connection_only: an ancestor of a full model (the tree stays connected), a component on a flow path with a
  full model, or G without O or L (flagged for the onboarding review: "documented fault mode, nothing to
  calibrate or observe").
- excluded: the rest.
An override replaces the decision and its reason (decided_by "override"); the signals are kept.
"""

from collections.abc import Mapping, Sequence

from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.scope import Decision, LifeEvidence, ScopeDecision, ScopeSignals
from kaful.config import DeclaredScope
from kaful.phases.p3_scope.a_collect_signals import Collected

FLAG_NOTHING_TO_CALIBRATE = "documented fault mode, nothing to calibrate or observe"
FLAG_UNCOUNTED = "has fault modes that do not count (downgraded or llm_judgement)"


def signals_of(got: Collected, life: Sequence[LifeEvidence]) -> ScopeSignals:
    return ScopeSignals(fault_modes=tuple(sorted(got.fault_modes)), uncounted_fault_modes=tuple(sorted(got.uncounted)),
                        observed_by=tuple(got.observed[k] for k in sorted(got.observed)),
                        driven_by=tuple(got.driven[k] for k in sorted(got.driven)), life=tuple(life),
                        declared=got.declared)


def _life_reason(life: Sequence[LifeEvidence]) -> str:
    by_basis: dict[str, list[str]] = {}
    for e in life:
        by_basis.setdefault(e.basis, []).append(e.ref)
    return "; ".join(f"life {b}: {', '.join(sorted(set(refs))[:3])}" + (" ..." if len(set(refs)) > 3 else "")
                     for b, refs in sorted(by_basis.items()))


def _full_reason(s: ScopeSignals) -> str:
    parts = [f"{len(s.fault_modes)} documented fault mode(s)"]
    if s.observed_by:
        parts.append("observed by " + ", ".join(c.channel for c in s.observed_by))
    if s.life:
        parts.append(_life_reason(s.life))
    return "; ".join(parts)


def _ancestors(model: MachineModel, ids: set[str]) -> dict[str, str]:
    parent = {c.id: c.parent for c in model.components}
    found: dict[str, str] = {}
    for cid in sorted(ids):
        p = parent.get(cid)
        while p is not None and p not in found:
            found[p] = cid
            p = parent.get(p)
    return found


def _neighbours(model: MachineModel, ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for f in sorted(model.flow_paths, key=lambda f: f.id):
        for a, b in ((f.from_component, f.to_component), (f.to_component, f.from_component)):
            if b in ids and a not in ids:
                found.setdefault(a, f"on flow path {f.id} with full model {b}")
    return found


def _rule(cid: str, s: ScopeSignals, full: set[str], up: Mapping[str, str],
          side: Mapping[str, str]) -> tuple[Decision, str, tuple[str, ...]]:
    flags = (FLAG_UNCOUNTED,) if s.uncounted_fault_modes else ()
    if cid in full:
        return "full_model", _full_reason(s), flags
    if s.fault_modes:
        return ("connection_only", f"{len(s.fault_modes)} documented fault mode(s), but no observing channel "
                "and no documented life", tuple(sorted({FLAG_NOTHING_TO_CALIBRATE, *flags})))
    if cid in up:
        return "connection_only", f"parent of full model {up[cid]}", flags
    if cid in side:
        return "connection_only", side[cid], flags
    return "excluded", "no documented fault mode, not a parent of or on a flow path with a full model", flags


def decide(model: MachineModel, collected: Mapping[str, Collected], life: Mapping[str, Sequence[LifeEvidence]],
           overrides: Sequence[DeclaredScope]) -> list[ScopeDecision]:
    signals = {cid: signals_of(collected[cid], life.get(cid, ())) for cid in sorted(collected)}
    full = {cid for cid, s in signals.items() if s.fault_modes and (s.observed_by or s.life)}
    up, side = _ancestors(model, full), _neighbours(model, full)
    forced = {o.component: o for o in overrides}
    decisions = []
    for cid, s in signals.items():
        decision, reason, flags = _rule(cid, s, full, up, side)
        o = forced.get(cid)
        if o is not None:
            decisions.append(ScopeDecision(component=cid, decision=o.decision, reason=o.reason, decided_by="override",
                                           signals=s, flags=tuple(sorted({*flags, f"rule said {decision}: {reason}"}))))
        else:
            decisions.append(ScopeDecision(component=cid, decision=decision, reason=reason, decided_by="rule",
                                           signals=s, flags=flags))
    return decisions
