"""The evaluation report as markdown: rendered by phase.py and tools/commands from what a_resolve_lineage to e_cost
built; no step reads it.

A rendering of the saved artifact, nothing more: every number here is in evaluation_report.json with the file it came
from.
"""

from kaful.artifacts.evaluation import Cost, EvaluationReport, Grounding, Prompts, TwinQuality
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
        lock = u.lock if u.lock == "locked" else f"{u.lock} (v{u.locked_version})"
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


def _cost(c: Cost) -> list[str]:
    m1 = c.m1.by_phase_usd if c.m1 else {}
    lines = ["## Cost", "", "| phase | produced by | calls | billed | unbilled (cache hits) | cold run | M1 legacy |",
             "|---|---|---|---|---|---|---|"]
    for p in c.by_phase:
        lines.append(f"| {p.phase} | {p.produced_by or '-'} | {p.calls} | {_usd(p.billed_usd)} | "
                     f"{_usd(p.unbilled_usd)} | {_usd(p.cold_usd)} | {_usd(m1.get(p.phase)) if c.m1 else '-'} |")
    total_m1 = _usd(c.m1.cold_total_usd) if c.m1 else "no M1 baseline for this machine"
    lines += [f"| **total** | | | {_usd(c.billed_usd)} | | **{_usd(c.cold_usd)}** | {total_m1} |", "",
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
    lines += _cost(r.cost)
    return "\n".join(lines).rstrip() + "\n"
