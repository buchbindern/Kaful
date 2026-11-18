"""Which unit a single-unit figure shows: a stated rule, never a hand pick (used by f2_tracking and f3_failure_time).

`median_abs_log_error`: rank the units with at least one scored point error (a true crossing ahead and an
uncensored median) by the median over their issues of |log(median RUL / true RUL)|, the primary estimator's;
show the median-ranked unit (the lower median when even; ties by unit id).
`median_abs_log_error_all_issues`: the same, over units with a scored point error at every issue (a unit whose part
is still running at the last issue; this favours longer-lived units, which the record says).
A unit named in the spec overrides the rule, and that is recorded too.
"""

from dataclasses import dataclass
from statistics import median

from kaful.figures.spec import UnitRule
from kaful.phases.p9_evaluate.f_calibration import Record


@dataclass(frozen=True)
class Choice:
    unit: str
    note: str


def unit_scores(records: list[Record], every_issue: bool = False) -> list[tuple[float, str]]:
    by_unit: dict[str, list[float]] = {}
    incomplete = set()
    for r in records:
        if r.error is not None and r.error.log_ratio is not None:
            by_unit.setdefault(r.unit, []).append(abs(r.error.log_ratio))
        else:
            incomplete.add(r.unit)
    return sorted((float(median(v)), u) for u, v in by_unit.items() if not (every_issue and u in incomplete))


def choose_unit(records: list[Record], explicit: str | None, estimator: str, quantity: str,
                rule: UnitRule = "median_abs_log_error") -> Choice:
    scores = unit_scores(records, every_issue=rule == "median_abs_log_error_all_issues")
    if explicit is not None:
        own = next((s for s, u in scores if u == explicit), None)
        shown = f"{own:.3g}" if own is not None else "none"
        return Choice(explicit, f"{explicit}: named in the spec (median |log error| {shown}, {estimator}, {quantity})")
    if not scores:
        raise ValueError(f"no unit has a scored point error for {estimator} {quantity}")
    i = (len(scores) - 1) // 2
    score, unit = scores[i]
    which = "at every issue" if rule == "median_abs_log_error_all_issues" else ""
    return Choice(unit, f"{unit}: rule {rule}, rank {i + 1} of {len(scores)} units with a scored point error "
                        f"{which}{' ' if which else ''}(median |log(median RUL / true RUL)| {score:.3g}, {estimator}, "
                        f"{quantity})")
