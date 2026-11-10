"""The parameterize section of the onboarding review (M9): the life priors with judged ones first, most RUL spread
first, then every other parameter; machine-level flags on top.

`kaful understand review <machine> --run RUN_ID` adds it when the run has a ParamSet; the review record gets one
`param` entry per component's life_nominal (in the same order), then one per other non-point parameter. An
entry set to `correct` with an [items.declare] table (param, unit, low and high or value, reason) becomes a
machine.toml [[param]] on review-apply; the next parameterize run uses it (source deployment, flagged).

Not used during a run (a tool). No API calls.
"""

import html
from collections.abc import Sequence

from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import Parameter, ParamSet
from kaful.phases.p4_parameterize.tools.report import ParamReport



def _prior(x: Parameter) -> str:
    d = x.prior
    if isinstance(d, Delta):
        return f"{d.value:g}"
    return f"median {d.quantile(0.5):,.4g} (5–95% {d.quantile(0.05):,.4g}–{d.quantile(0.95):,.4g})"


def reviewed(ps: ParamSet, report: ParamReport) -> list[str]:
    """param ids worth a look: each component's life (report order: judged first), then other non-point priors."""
    lives = [f"{c.component}.life_nominal" for c in report.components]
    rest = sorted(x.param_id for x in ps.parameters if x.param_id not in lives and not isinstance(x.prior, Delta))
    return [*lives, *rest]


def param_html(ps: ParamSet, report: ParamReport) -> str:
    by_id = {x.param_id: x for x in ps.parameters}
    lives = []
    for c in report.components:
        life = by_id[f"{c.component}.life_nominal"]
        lives.append((f"<b>{html.escape(c.component)}</b>", html.escape(c.family), html.escape(life.source),
                      html.escape(_prior(life)), f"{c.sigma_log_rul:.2f}", html.escape(c.assumption),
                      "".join(f'<div class="warn">{html.escape(f)}</div>' for f in c.flags)))
    others = [(html.escape(x.param_id), html.escape(x.role), html.escape(x.unit), html.escape(_prior(x)),
               html.escape(x.source), html.escape(x.ref)) for x in ps.parameters if not x.param_id.endswith(".life_nominal")]

    def table(head: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
        return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>")
    return (f"<h2>Parameters (parameterize run {html.escape(ps.run_id)}, time unit {html.escape(ps.time_unit)})</h2>"
            + "".join(f'<p class="warn"><b>{html.escape(f)}</b></p>' for f in report.flags)
            + '<p class="meta">Life priors: judged (llm_judgement) first, most RUL spread first. To change one, set '
            "its <code>param</code> entry to <code>correct</code> with an [items.declare] table (param, unit, low "
            "and high as 5th and 95th percentiles, or value; reason); review-apply writes machine.toml [[param]]. "
            "The full sensitivity report is outputs/param_reports/&lt;run&gt;.md.</p>"
            + table(("component", "family", "life source", "life prior", "sd log RUL", "assumption", "flags"), lives)
            + "<h3>Other parameters</h3>"
            + table(("parameter", "role", "unit", "prior", "source", "ref"), others))
