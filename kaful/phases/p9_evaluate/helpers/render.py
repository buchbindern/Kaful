"""The evaluation report as markdown: rendered by phase.py and tools/commands from what a_resolve_lineage to e_cost
built; no step reads it.

A rendering of the saved artifact, nothing more: every number here is in evaluation_report.json with the file it came
from.
"""

from kaful.artifacts.evaluation import (
    CIRCULAR,
    Bounds,
    Calibration,
    Cost,
    CoverageCell,
    EvaluationReport,
    Grounding,
    Prompts,
    Sensitivity,
    TwinQuality,
)
from kaful.phases import PHASE_ORDER


def _usd(x: float | None) -> str:
    return "-" if x is None else f"${x:.4f}" if x < 0.1 else f"${x:.2f}"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _grounding(g: Grounding) -> list[str]:
    classes = list(g.evidence[0].quotes) if g.evidence else []
    lines = ["## Grounding", "", "Evidence quotes by verification class (code-checked, rule 4):", "",
             "| artifact | " + " | ".join(classes) + " | items with a source | downgraded |",
             "|---|" + "---|" * (len(classes) + 2)]
    for e in g.evidence:
        down = "; ".join(f"{k}: {v}" for k, v in e.downgraded.items()) or "0"
        lines.append(f"| {e.kind} | " + " | ".join(str(e.quotes[c]) for c in classes) + f" | {e.provenanced} | {down} |")
    lines += ["", "Parameters by source: " + ", ".join(f"{k} {v}" for k, v in g.params_by_source.items() if v) + ".", ""]
    if g.life_statements_by_source:
        lines += ["Life and interval statements behind calibrated lives, by source: "
                  + ", ".join(f"{k} {v}" for k, v in g.life_statements_by_source.items())
                  + " (`deployment`: declared in machine.toml, not documented).", ""]
    if g.params_by_source_role:
        lines += ["| source | role | parameters |", "|---|---|---|"]
        lines += [f"| {c.source} | {c.role} | {c.count} |" for c in g.params_by_source_role]
        lines.append("")
    if g.declared_params:
        lines += [f"Declared in machine.toml (`deployment`, not documented): {', '.join(g.declared_params)}.", ""]
    if g.page_coverage:
        c = g.page_coverage
        lines += [f"Ingest index on the M6 quote set ({c.matcher_version}, {c.n_quotes} quotes, `{c.report.path}`): "
                  f"{c.verbatim} verbatim, {c.near} near, {c.unsupported} unsupported; page coverage "
                  f"{_pct(c.page_share)}.", ""]
    return lines + [f"- {n}" for n in g.notes] + ([""] if g.notes else [])


def _prompts(p: Prompts) -> list[str]:
    lines = ["## Prompt quality", "", "| prompt | version | phase | models | calls | lock | lowest score per run |",
             "|---|---|---|---|---|---|---|"]
    for u in p.uses:
        worst = "; ".join(f"run {s.sample}: {min(s.metrics, key=lambda k: (s.metrics[k], k))} "
                          f"{min(s.metrics.values()):.2f}" for s in u.run_scores if s.metrics) or "no saved scores"
        lock = u.lock if u.lock == "locked" else (f"equivalent to the lock v{u.locked_version} (reuse rule)"
                                                  if u.lock == "equivalent_to_lock" else f"{u.lock} (v{u.locked_version})")
        lines.append(f"| {u.prompt_id} | v{u.version} | {u.phase} | {', '.join(u.models)} | {u.calls} | {lock} | "
                     f"{worst} |")
    for m in p.machine_scores:
        lines += ["", f"{m.tool} (`{m.file.path}`){': ' + m.note if m.note else ''}:", ""]
        lines += [f"- {k}: {v:.3g}" for k, v in m.metrics.items()]
    return lines + [""] + [f"- {n}" for n in p.notes] + ([""] if p.notes else [])


def _twin(t: TwinQuality) -> list[str]:
    lines = ["## Twin quality", "", f"Verification (`{t.report.path}`): {t.components} components, passed: {t.passed}.",
             "", "| | dynamic pass rate | full pass rate |", "|---|---|---|",
             f"| this twin | {_pct(t.dynamic_pass_rate)} | {_pct(t.full_pass_rate)} |"]
    if t.legacy:
        lines.append(f"| legacy baseline (M11, {t.legacy.components} components) | "
                     f"{_pct(t.legacy.dynamic_pass_rate)} | {_pct(t.legacy.full_pass_rate)} |")
    lines.append("")
    if t.quarantined:
        lines += [f"Quarantined: {', '.join(t.quarantined)}.", ""]
    if t.failing_checks:
        lines += ["Failing checks (before quarantine): " + ", ".join(f"{k} {v}" for k, v in t.failing_checks.items()),
                  ""]
    return lines


def _ci(b: Bounds | None) -> str:
    return "-" if b is None else f"{b.low:.0%}-{b.high:.0%}"


def _n(x: float | None) -> str:
    return "-" if x is None else f"{x:,.4g}"


def _coverage_rows(cells: list[CoverageCell]) -> list[str]:
    lines = ["| estimator | quantity | part | coverage | covered / pairs (units) | 95% CI, units | 95% CI, pairs | "
             "censored covered / missed | already triggered | by issue fraction |", "|---|---|---|---|---|---|---|---|---|---|"]
    pooled = [c for c in cells if c.issue_fraction is None]
    for c in pooled:
        by = " ".join(f"{x.issue_fraction:g}: {_pct(x.share)}" for x in cells if x.issue_fraction is not None
                      and (x.estimator, x.quantity) == (c.estimator, c.quantity))
        lines.append(f"| {c.estimator} | {c.quantity} | {c.part_class} | {_pct(c.share)} | {c.covered} / {c.pairs} "
                     f"({c.units}) | {_ci(c.ci_units)} | {_ci(c.ci_pairs)} | {c.classes['censored_covered']} / "
                     f"{c.classes['censored_missed']} | {c.classes['already_triggered']} ({c.triggered_agree} agreed) | {by} |")
    return lines


def _sharpness(k: Calibration) -> list[str]:
    """Width of the 90% interval beside its coverage, for every estimator on the judged quantities."""
    cover = {(c.estimator, c.quantity): c for c in k.cells if c.issue_fraction is None}
    rows = [a for a in k.accuracy if a.part_class not in CIRCULAR]
    lines = ["", "### Sharpness: how much the data narrowed the forecasts", "",
             "| quantity | estimator | coverage | median 90% width / true RUL | width vs open-loop, same pairs | "
             "open widths (95% point past the horizon) |", "|---|---|---|---|---|---|"]
    for a in sorted(rows, key=lambda a: (a.quantity, a.estimator)):
        c = cover.get((a.estimator, a.quantity))
        ratio = "-" if a.width_ratio_open_loop is None else f"{a.width_ratio_open_loop:.2f}x"
        lines.append(f"| {a.quantity} | {a.estimator} | {_pct(c.share if c else None)} | "
                     f"{_n(a.width_relative_median)} | {ratio} | {a.width_censored} |")
    return lines


def _calibration(k: Calibration) -> list[str]:
    lo, hi = k.target
    lines = ["## Calibration on synthetic truth", "",
             f"Target: the {k.interval[1] - k.interval[0]:.0%} interval ({k.interval[0]:g}-{k.interval[1]:g} quantiles) "
             f"holds the true next crossing {lo:.0%}-{hi:.0%} of the time, judged on observed quantities of "
             f"{k.primary_estimator}. rho = {k.process_noise_cv:g}, not tuned.", "",
             "| quantity | coverage | 95% CI (units) | verdict | CI overlaps the target |", "|---|---|---|---|---|"]
    lines += [f"| {v.quantity} | {_pct(v.coverage)} | {_ci(v.ci_units)} ({v.units} units) | **{v.verdict}** | "
              f"{v.ci_overlaps_target} |" for v in k.verdicts]
    lines += [""] + [f"_{n}._" for n in k.notes] if k.notes else []
    lines += _sharpness(k)
    judged = [c for c in k.cells if c.part_class not in CIRCULAR]
    circular = [c for c in k.cells if c.part_class in CIRCULAR]
    lines += ["", "### Coverage, observed and label-invariant quantities", ""] + _coverage_rows(judged)
    if circular:
        lines += ["", "### Coverage, parts with no observation of their own (circular by construction)", ""]
        lines += _coverage_rows(circular)
    lines += ["", f"### Accuracy (alpha = {k.alpha:g}, beta = {k.beta:g})", "",
              "| estimator | quantity | part | alpha-lambda (median mass in band) | prognostic horizon, median (share "
              "of span) | units | RMSE median RUL (relative) | median abs. relative error | median abs. log error | "
              "n (median censored) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in k.accuracy:
        lines.append(f"| {a.estimator} | {a.quantity} | {a.part_class} | {_pct(a.alpha_lambda)} "
                     f"({a.alpha_lambda_pass}/{a.alpha_lambda_n}; {_pct(a.alpha_mass_median)}) | {_n(a.horizon_median)} "
                     f"({_pct(a.horizon_share_median)}) | {a.horizon_units} | {_n(a.rmse)} ({_n(a.rmse_relative)}) | "
                     f"{_n(a.abs_relative_error_median)} | {_n(a.abs_log_error_median)} | {a.rmse_n} ({a.rmse_excluded}) |")
    return lines + [""] + [f"- {f}" for f in k.flags] + ([""] if k.flags else [])


def _sensitivity(k: Sensitivity) -> list[str]:
    lines = ["## Sensitivity: what dominates RUL uncertainty, and which data would narrow it", "",
             "Share of var(log RUL) by prior source (first order, mean over components): "
             + ", ".join(f"{src} {_pct(v)}" for src, v in sorted(k.variance_by_source.items(), key=lambda kv: -kv[1]))
             + f" ({k.samples} samples, seed {k.seed}, `{k.param_set.path}`).", "",
             "| component | data on it | life source | sd log RUL (prior) | dominant prior (share) | life: posterior / "
             "prior width | remaining spread | most of the remaining spread | data that would narrow it most |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in k.components:
        top = c.effects[0] if c.effects else None
        post = max((q for q in c.posterior if q.remaining_share is not None), key=lambda q: (q.remaining_share, q.param),
                   default=None)
        lines.append(f"| {c.component} | {c.part_class} | {c.life_source} | {c.sigma_log_rul:.2f} | "
                     f"{f'{top.param} ({top.source}, {top.share:.2f})' if top else '-'} | {_n(c.life_contraction)} | "
                     f"{c.remaining_spread:.2f} | {f'{post.param} ({post.remaining_share:.2f})' if post else '-'} | "
                     f"{'; '.join(c.recommendations) or '-'} |")
    lines.append("")
    lines += [f"- Estimator assumption, not a prior: {a}" for a in k.estimator_assumptions]
    return lines + [f"- {n}" for n in k.notes] + [""]


def _range(low: float, high: float) -> str:
    return _usd(high) if abs(high - low) < 5e-5 else f"{_usd(low)} - {_usd(high)}"


def _cost(c: Cost) -> list[str]:
    m1 = c.m1.by_phase_usd if c.m1 else {}
    lines = ["## Cost", "", "| phase | produced by | calls | billed | unbilled (cache hits, synchronous prices) | "
             "cold run (all batched - none batched) | M1 legacy |", "|---|---|---|---|---|---|---|"]
    for p in c.by_phase:
        lines.append(f"| {p.phase} | {p.produced_by or '-'} | {p.calls} | {_usd(p.billed_usd)} | "
                     f"{_usd(p.unbilled_usd)} | {_range(p.cold_low_usd, p.cold_usd)} | "
                     f"{_usd(m1.get(p.phase)) if c.m1 else '-'} |")
    total_m1 = _usd(c.m1.cold_total_usd) if c.m1 else "no M1 baseline for this machine"
    lines += [f"| **total** | | | {_usd(c.billed_usd)} | | **{_range(c.cold_low_usd, c.cold_usd)}** | {total_m1} |",
              "",
              f"Development spend: {_usd(c.development_billed_usd)} billed over the {c.development_runs} runs with a ledger "
              f"({c.runs_total} runs in all; the others made no API call)."]
    if c.m1:
        by: dict[str, list[str]] = {}
        for step, phase in c.m1.mapping.items():
            by.setdefault(phase, []).append(step)
        lines.append("Legacy steps mapped onto each phase: " + "; ".join(
            f"{phase} {', '.join(steps)}" for phase, steps in sorted(by.items(), key=lambda kv: PHASE_ORDER.index(
                kv[0]) if kv[0] in PHASE_ORDER else len(PHASE_ORDER))) + ".")
        lines.append(f"M1 legacy rerun without ingest (`--from phase2`): {_usd(c.m1.rerun_total_usd)}; source "
                     f"`{c.m1.source.path}`.")
    return lines + [""] + [f"- {n}" for n in c.notes] + ([""] if c.notes else [])


def report_md(r: EvaluationReport) -> str:
    lines = [f"# {r.machine_id}: evaluation of run {r.evaluated_run}", "", f"Report run {r.run_id}; time unit "
             f"{r.time_unit or '-'}; complete: {r.complete}.", ""]
    lines += [f"> {d}" for d in r.disclaimers] + [""]
    if r.summary:
        lines += ["## Summary", ""] + [f"- {x}" for x in r.summary] + [""]
    lines += ["## Lineage", "", "| phase | status | produced by |", "|---|---|---|"]
    lines += [f"| {p.phase} | {p.status} | {p.produced_by or '-'} |" for p in r.lineage] + [""]
    if r.not_reached:
        lines += ["## Not reached", ""] + [f"- **{n.section}**: {n.reason}" for n in r.not_reached] + [""]
    if r.flags:
        lines += ["## Flags", ""] + [f"- {f}" for f in r.flags] + [""]
    if r.grounding:
        lines += _grounding(r.grounding)
    lines += _prompts(r.prompts)
    if r.twin_quality:
        lines += _twin(r.twin_quality)
    if r.calibration:
        lines += _calibration(r.calibration)
    if r.sensitivity:
        lines += _sensitivity(r.sensitivity)
    lines += _cost(r.cost)
    return "\n".join(lines).rstrip() + "\n"
