"""The scope section of the onboarding review (M8): every decision with the signals it rests on, flagged first.

`kaful understand review <machine> --run RUN_ID` adds it when the run has a Scope: the page gets a "Scope
decisions" section, and the review record one `scope` entry per component that has a signal or a flag. An
entry the engineer sets to `correct`, with a [items.declare] table (component, decision, reason), becomes a
machine.toml [[scope]] override on review-apply; an `add` of kind `life` becomes a [[life]] declaration
(which counts for L on the next run). The next scope run shows decided_by "override", and the manifest
records each override.

Not used during a run (a tool). No API calls.
"""

import html
from collections.abc import Sequence

from kaful.artifacts.base import Evidence
from kaful.artifacts.scope import Scope, ScopeDecision

ORDER = {"full_model": 0, "connection_only": 1, "excluded": 2}


def reviewed(scope: Scope) -> list[ScopeDecision]:
    """Decisions worth a look, flagged first, then full models, connection only, excluded (ties by id)."""
    keep = [d for d in scope.decisions if d.flags or d.signals.fault_modes or d.signals.life
            or d.decided_by == "override" or d.decision != "excluded"]
    return sorted(keep, key=lambda d: (not d.flags, ORDER[d.decision], d.component))


def _where(e: Evidence) -> str:
    label = f" (label {e.page_label})" if e.page_label and e.page_label != str(e.page) else ""
    return f"{e.doc_id} p{e.page}{label}"


def _list(items: Sequence[str]) -> str:
    return "<br>".join(html.escape(i) for i in items) or '<span class="meta">-</span>'


def _row(d: ScopeDecision) -> tuple[str, ...]:
    s = d.signals
    life = [f"{e.basis}: {e.ref}" + (f" “{e.evidence.quote[:120]}”" if e.evidence else f" ({e.note})")
            for e in s.life]
    return (f"<b>{html.escape(d.component)}</b>", html.escape(f"{d.decision} ({d.decided_by})"),
            str(len(s.fault_modes)),
            _list([f"{c.channel} ({c.confidence:.2f})" for c in s.observed_by]),
            _list([f"{c.channel} ({c.confidence:.2f}, by its {c.named_in})" for c in s.named_by]),
            _list([c.channel for c in s.threshold_events]), _list(life),
            "".join(f'<div class="warn">{html.escape(f)}</div>' for f in d.flags), html.escape(d.reason))


def scope_html(scope: Scope) -> str:
    rows = [_row(d) for d in reviewed(scope)]
    hidden = len(scope.decisions) - len(rows)
    head = ("component", "decision", "G", "O: fault-mode link", "O: channel names it (lower confidence)",
            "O: threshold event", "L: life", "flags", "reason")
    table = ("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
             + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")
    wide = "".join(f"<li>{html.escape(_where(s.evidence))}: "
                   f"“{html.escape(s.evidence.quote)}” <span class=\"meta\">linked: "
                   f"{html.escape(', '.join(s.linked) or 'none')}</span></li>" for s in scope.machine_life)
    return (f"<h2>Scope decisions (scope run {html.escape(scope.run_id)})</h2>"
            '<p class="meta">full_model = a documented fault mode (G) and an observing channel or a documented '
            "life. To change a decision, set its <code>scope</code> entry to <code>correct</code> with an "
            "[items.declare] table (component, decision, reason); to give a component a life, add a "
            "<code>life</code> entry with [items.declare] (component, kind, value, unit). review-apply writes "
            f"them to machine.toml [[scope]] / [[life]]. {hidden} excluded components without a signal are not "
            "listed.</p>" + table + f"<h3>Machine-wide life statements ({len(scope.machine_life)})</h3><ul>"
            + wide + "</ul>")
