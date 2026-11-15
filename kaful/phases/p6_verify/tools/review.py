"""The verify section of the onboarding review (M11): quarantined components first, each with the check it failed,
what was measured against the limit, and how to resolve it; then every component's checks.

`kaful understand review <machine> --run RUN_ID` adds it when the run has a VerificationReport. It adds no review
entries: a quarantine is resolved through the entries that already exist (a `param` correction becomes a
machine.toml [[param]], a `scope` correction a [[scope]] override), or by --allow-failures, recorded in the
manifest.

Not used during a run (a tool). No API calls.
"""

import html
from collections.abc import Mapping

from kaful.artifacts.verification import CheckResult, VerificationReport
from kaful.phases.p6_verify.a_recheck_static import COMPOSITE

RESOLVE = {
    "life_consistency": "its rate no longer matches its life prior: check the [[param]] for its life or k",
    "maintenance_resets": "a maintenance rule does not act as declared: check [[maintenance]] and its recovery",
}
RESOLVE_DEFAULT = ("the twin misbehaves for this component: correct its parameters ([[param]]), take it out of "
                   "full models ([[scope]] connection_only), or run with --allow-failures (recorded)")
MARK = {"pass": "✓", "fail": "✗", "warn": "!", "skipped": "–"}


def _numbers(values: Mapping[str, float]) -> str:
    return ", ".join(f"{k} {v:.6g}" for k, v in sorted(values.items()))


def _table(head: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    return ("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")


def _failed_row(r: CheckResult) -> tuple[str, ...]:
    return (f"<b>{html.escape(r.target)}</b>", html.escape(r.check), html.escape(r.message),
            html.escape(_numbers(r.measured)), html.escape(_numbers(r.limits)),
            html.escape(RESOLVE.get(r.check, RESOLVE_DEFAULT)))


def verify_html(report: VerificationReport) -> str:
    held = set(report.quarantined_components)
    failed = [_failed_row(r) for r in report.results if r.status == "fail"]
    per_component = [r for r in report.results if r.target != COMPOSITE]
    checks, targets = sorted({r.check for r in per_component}), sorted({r.target for r in per_component})
    by = {(r.target, r.check): r for r in report.results}
    grid = [(f"<b>{html.escape(t)}</b>" + (" (quarantined)" if t in held else ""),
             *(MARK[by[(t, c)].status] if (t, c) in by else "" for c in checks)) for t in targets]
    status = ("failures allowed (--allow-failures)" if report.allowed_failures
              else "passed" if report.passed and not held else
              f"{len(held)} quarantined" if report.passed else "failed")
    quarantined = "".join(f'<li class="warn"><b>{html.escape(q.component)}</b>: {html.escape(q.reason)}'
                          + (f" (also quarantined: {html.escape(', '.join(q.dependents))})" if q.dependents else "")
                          + "</li>" for q in report.quarantined)
    runs = ", ".join(f"{html.escape(r.name)} ({r.steps:,} steps of {r.dt:.4g})" for r in report.runs)
    return (f"<h2>Verification (verify run {html.escape(report.run_id)}): {html.escape(status)}</h2>"
            f'<p class="meta">Simulations: {runs}. A quarantined component is left out of the twin later phases use; '
            "the rest were checked again without it.</p>"
            f"<h3>Quarantined ({len(report.quarantined)})</h3><ul>{quarantined}</ul>"
            f"<h3>Failed checks ({len(failed)})</h3>"
            + _table(("target", "check", "what happened", "measured", "limits", "to resolve"), failed)
            + "<h3>Checks per component</h3>" + _table(("component", *checks), grid))
