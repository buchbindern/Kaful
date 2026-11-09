"""Step d: the scope gates (hard rule 2). Each failure is a line in the phase record, and the run stops.

- every MachineModel component has exactly one decision, and no decision names another component;
- a rule's full_model has G and (O or L);
- every machine.toml [[scope]] and [[life]] entry names a component of the MachineModel.
"""

from collections.abc import Sequence

from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.scope import ScopeDecision
from kaful.config import MachineConfig


def check(model: MachineModel, decisions: Sequence[ScopeDecision], machine: MachineConfig) -> tuple[str, ...]:
    ids = {c.id for c in model.components}
    decided = [d.component for d in decisions]
    failures = [f"component {c}: no scope decision" for c in sorted(ids - set(decided))]
    failures += [f"scope decision for unknown component {c}" for c in sorted(set(decided) - ids)]
    failures += [f"component {c}: {decided.count(c)} scope decisions" for c in sorted(set(decided))
                 if decided.count(c) > 1]
    for d in decisions:
        s = d.signals
        if d.decided_by == "rule" and d.decision == "full_model" and not (s.fault_modes and (s.observed_by or s.life)):
            failures.append(f"component {d.component}: full_model by rule without G and (O or L)")
    failures += [f"machine.toml [[scope]] names unknown component {o.component}" for o in machine.scope
                 if o.component not in ids]
    failures += [f"machine.toml [[life]] names unknown component {d.component}" for d in machine.life
                 if d.component not in ids]
    return tuple(failures)
