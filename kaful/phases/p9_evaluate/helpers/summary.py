"""The report's summary lines, computed from its sections (phase.py, after a_resolve_lineage to g_sensitivity).

Decided 2026-11-17: when every observed quantity's coverage is consistent with the target but the forecasts are wide,
the summary says "calibrated but not sharp", with the cause (judged lives, sparse observations) and what would sharpen
the forecasts (documented lives, richer channels), naming the parts. Every number is read from the report.
"""

from kaful.artifacts.evaluation import CIRCULAR, Calibration, NotReached, Sensitivity

SHARP_MASS = 0.5  # alpha-lambda's own pass mark (beta): below it on every quantity, the forecasts are not sharp
TOP_PARTS = 3


def _span(values: list[float], fmt: str) -> str:
    lo, hi = format(min(values), fmt), format(max(values), fmt)
    return lo if lo == hi else f"{lo}-{hi}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _calibration_line(k: Calibration) -> tuple[str, bool]:
    observed = [a for a in k.accuracy if a.estimator == k.primary_estimator and a.part_class not in CIRCULAR
                and a.part_class != "not_identifiable"]
    consistent = bool(k.verdicts) and all(v.ci_overlaps_target for v in k.verdicts)
    passed = sum(a.alpha_lambda_pass for a in observed)
    total = sum(a.alpha_lambda_n for a in observed)
    widths = [a.width_relative_median for a in observed if a.width_relative_median is not None]
    ratios = [a.width_ratio_open_loop for a in observed if a.width_ratio_open_loop is not None]
    sharp = total > 0 and passed / total >= SHARP_MASS
    cov = [v.coverage for v in k.verdicts]
    head = ("Calibrated but not sharp" if consistent and not sharp else "Calibrated" if consistent
            else "Not calibrated")
    text = (f"{head}: on {_plural(len(k.verdicts), 'observed quantity').replace('quantitys', 'quantities')} the "
            f"{k.primary_estimator}'s 90% intervals hold the truth {_span(cov, '.0%')} of the time (target "
            f"{k.target[0]:.0%}-{k.target[1]:.0%}{', every interval consistent with it' if consistent else ''}); "
            f"alpha-lambda passes in {passed} of {total} forecasts")
    if widths:
        text += f", and the median 90% interval spans {_span(widths, '.1f')}x the true RUL"
    if ratios:
        text += f" ({_span(ratios, '.2f')}x open-loop's on the same forecasts)"
    open_ = [a for a in observed if a.width_relative_median is None and a.width_censored]
    if open_:
        text += "; the 95% point lies past the horizon in " + ", ".join(
            f"{a.width_censored} of {a.alpha_lambda_n} {a.quantity} forecasts" for a in open_)
    return text + ".", consistent and not sharp


def _cause_and_remedy(s: Sensitivity) -> list[str]:
    n = len(s.components)
    judged = [c for c in s.components if c.life_source == "llm_judgement"]
    observed = [c for c in s.components if c.part_class in ("observed", "not_identifiable")]
    shared = [c for c in observed if c.part_class == "not_identifiable"]
    blind = [c.component for c in s.components if c.part_class != "observed"][:TOP_PARTS]
    lives = [c.component for c in judged][:TOP_PARTS]
    seen = f"{len(observed)} of {n} parts are observed" + (f", {len(shared)} of them on one shared channel and not "
                                                             "identifiable apart" if shared else "")
    out = [f"Cause: wide priors ({len(judged)} of {n} full-model lives are llm_judgement) and sparse observations "
           f"({seen})."]
    remedy = []
    if lives:
        remedy.append(f"documented lives ({', '.join(lives)})")
    if blind:
        remedy.append(f"richer channels ({', '.join(blind)})")
    if remedy:
        out.append("What would sharpen it, most remaining spread first: " + " and ".join(remedy) + ".")
    return out


def summary(calibration: Calibration | None, sensitivity: Sensitivity | None,
            not_reached: list[NotReached]) -> tuple[str, ...]:
    if calibration is None or not calibration.verdicts:
        reasons = "; ".join(f"{n.section}: {n.reason}" for n in sorted(not_reached, key=lambda n: n.section))
        return (f"Not evaluated end to end ({reasons}).",) if reasons else ()
    line, not_sharp = _calibration_line(calibration)
    out = [line]
    if not_sharp and sensitivity is not None:
        out += _cause_and_remedy(sensitivity)
    return tuple(out)
