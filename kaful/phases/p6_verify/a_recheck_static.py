"""Step a: the static checks build gated on (kaful.twin_checks), re-run on the artifacts verify loaded, plus
provenance. A twin from a reused base run is checked too, and M14 finds every check in one report.

Each failing (target, check) is one result, its messages joined; a check with no failure is one `composite`
pass. A static failure is never quarantined: it stops the run.
"""

from collections import defaultdict

from kaful.artifacts.param_set import ParamSet
from kaful.artifacts.twin_definition import TwinDefinition
from kaful.artifacts.verification import CheckResult
from kaful.twin_checks import provenance_problems, static_problems

STATIC_CHECKS = ("bound", "connection_units", "params_exist", "provenance", "units")
COMPOSITE = "composite"


def static_results(twin: TwinDefinition, ps: ParamSet,
                   quarantined: frozenset[str] = frozenset()) -> tuple[CheckResult, ...]:
    """`quarantined`: components step d left out of the twin (the recheck), not reported as unbound."""
    messages: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in (*static_problems(twin, ps, quarantined), *provenance_problems(ps)):
        messages[(p.target, p.check)].append(p.message)
    failed = {check for _, check in messages}
    out = [CheckResult(check=check, target=target, status="fail", message="; ".join(m))
           for (target, check), m in messages.items()]
    out += [CheckResult(check=check, target=COMPOSITE, status="pass") for check in STATIC_CHECKS
            if check not in failed]
    return tuple(sorted(out, key=lambda r: (r.target, r.check)))
