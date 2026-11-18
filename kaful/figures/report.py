"""The demo report: one self-contained HTML page from the built figures and the lineages' artifacts (M16; moved
from M15). Written by `kaful figures build` when the spec has a [report] section, after the figures.

- Figures are embedded as SVG data URIs (isolated documents: matplotlib SVGs share element ids), each checked
  against the sha256 the manifest just recorded, with its spec caption.
- Tables are read from artifacts: provenance per machine (each parameter's source, prior and reference, with a
  manual quote cut to one sentence of at most `quote_max_chars`, because the page is committed), verification
  against the legacy baseline, and scope decisions.
- The forecast section has a unit selector: F3's drawing for every unit, rendered here from the same data.
- No script libraries, no external requests; a few lines of inline JS switch the selector. Output is byte-stable.
"""

import base64
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt

from kaful.artifacts.distributions import Delta
from kaful.artifacts.param_set import Parameter, ParamSet
from kaful.artifacts.scope import Scope
from kaful.artifacts.evaluation import EVALUATION_DISCLAIMERS, EvaluationReport
from kaful.figures import f3_failure_time as f3
from kaful.figures.drawn import Context
from kaful.figures.select import choose_unit
from kaful.figures.sources import LoadedLineage
from kaful.figures.spec import FailureTimeSpec, FigureBase, ReportSpec
from kaful.figures.style import encode, rc
from kaful.figures.truth_view import load_truth, rul_records

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # a sentence ends before a capital letter
ABBREVIATIONS = ("approx.", "e.g.", "i.e.", "no.", "fig.", "max.", "min.", "ca.", "cf.", "vs.", "etc.", "incl.")
ELLIPSIS = "…"
CSS = """
:root{--ink:#1f1f1e;--muted:#5f5e58;--rule:#e2e1dc;--surface:#fcfcfb;--accent:#0072B2}
body{margin:0;background:var(--surface);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.25rem;margin:2.2em 0 .4em;border-top:1px solid var(--rule);padding-top:1em}
.lede{color:var(--muted)}figure{margin:1em 0}figure img{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:.88rem;margin-top:.4em}
.note{background:#f3f2ee;border-left:3px solid var(--accent);padding:.6em .9em;font-size:.9rem;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.84rem;margin:.6em 0}
th,td{text-align:left;padding:.3em .5em;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--muted);font-weight:600}td.q{font-style:italic;color:var(--muted)}
details{margin:.5em 0}summary{cursor:pointer;font-weight:600}
select{font:inherit;padding:.2em .4em}.unit{display:none}.unit.on{display:block}
.wrap{overflow-x:auto}footer{color:var(--muted);font-size:.8rem;margin-top:3em}
"""
JS = """document.getElementById('unit').addEventListener('change',function(e){
document.querySelectorAll('.unit').forEach(function(d){d.classList.toggle('on',d.id==='u-'+e.target.value)})});"""


@dataclass(frozen=True)
class Section:
    title: str
    lede: str
    figures: tuple[str, ...] = ()
    table: str = ""  # a key of TABLES


SECTIONS = (
    Section("1 · The pipeline", "A machine's documents in, a twin and remaining-life forecasts out.", ("F1",)),
    Section("2 · Grounded in the documents", "Every number has a source, and a manual value counts as manual only "
            "when code finds its quote on the cited page.", ("F6",), "provenance"),
    Section("3 · The twin is checked before use", "Static and dynamic checks gate every run; failures stop it.",
            (), "verification"),
    Section("4 · Wear over time", "Each part degrades by a physics-shaped template; documented maintenance resets "
            "it.", ("F8",)),
    Section("5 · Tracking a machine from its sensors", "A particle filter updates each part's health and its own "
            "rate from the readings.", ("F2", "F10")),
    Section("6 · Remaining-life forecasts", "Forecasts at several issue times, against the true failure time.",
            ("F3",), "units"),
    Section("7 · Is the uncertainty honest?", "Calibrated but not sharp.", ("F4", "F7")),
    Section("8 · What if?", "Forecasts under a changed maintenance plan.", ("F9",)),
    Section("9 · Cost", "Every LLM call is logged, cached and priced.", ("F5",)),
    Section("10 · Not everything gets a physics model", "Scope decides which parts get a full model before any "
            "expensive per-part work.", (), "scope"),
)


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def first_sentence(text: str) -> str:
    """Up to the first sentence end that does not follow a known abbreviation."""
    for m in SENTENCE_END.finditer(text):
        last_word = text[: m.start()].rsplit(" ", 1)[-1].lower()
        if last_word not in ABBREVIATIONS:
            return text[: m.start()]
    return text


def short_quote(quote: str, max_chars: int) -> str:
    """At most one sentence, at most max_chars (cut at a word, with an ellipsis)."""
    first = first_sentence(quote.strip())
    if len(first) <= max_chars and first == quote.strip():
        return first
    cut = first if len(first) <= max_chars else first[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + ELLIPSIS


def page_label(doc: str, page: int, label: str | None) -> str:
    return f"{doc} p{page}" + (f" (label {label})" if label else "")


def _num(x: float) -> str:
    return f"{x:.3g}"


def _prior(p: Parameter) -> str:
    if isinstance(p.prior, Delta):
        return _num(p.prior.value)
    q = [p.prior.quantile(x) for x in (0.05, 0.5, 0.95)]
    return f"{_num(q[1])} ({_num(q[0])}–{_num(q[2])})"


def provenance_table(ll: LoadedLineage, max_chars: int) -> str:
    ps = ll.load(ll._ref(("parameterize", "param_set")), ParamSet)
    rows = []
    for p in sorted(ps.parameters, key=lambda p: (p.component, p.name)):
        ev = p.evidence[0] if p.evidence else None
        quote = (f"“{esc(short_quote(ev.quote, max_chars))}” — {esc(page_label(ev.doc_id, ev.page, ev.page_label))}"
                 if ev else "")
        down = f" (from {esc(p.downgraded_from)})" if p.downgraded_from else ""
        rows.append(f"<tr><td>{esc(p.component)}</td><td>{esc(p.name)}</td><td>{esc(p.source)}{down}</td>"
                    f"<td>{esc(_prior(p))} {esc(p.unit)}</td><td>{esc(p.ref)}</td><td class=q>{quote}</td></tr>")
    head = ("<tr><th>part</th><th>parameter</th><th>source</th><th>prior: median (5–95%)</th><th>reference</th>"
            "<th>quote (one sentence at most)</th></tr>")
    return (f"<details><summary>{esc(ll.spec.label)}: {len(ps.parameters)} parameters, "
            f"{len(ps.life_statements)} life statements</summary><div class=wrap><table>{head}{''.join(rows)}"
            f"</table>{_life_table(ps, max_chars)}</div></details>")


def _life_table(ps: ParamSet, max_chars: int) -> str:
    """The stated lives and intervals that calibrated rates rest on."""
    if not ps.life_statements:
        return ""
    rows = []
    for st in sorted(ps.life_statements, key=lambda s: s.id):
        ev = st.evidence[0] if st.evidence else None
        quote = (f"“{esc(short_quote(ev.quote, max_chars))}” — {esc(page_label(ev.doc_id, ev.page, ev.page_label))}"
                 if ev else esc(st.ref))
        rows.append(f"<tr><td>{esc(st.component)}</td><td>{esc(st.kind.replace('_', ' '))}</td>"
                    f"<td>{esc(_num(st.value))} {esc(st.unit)}</td><td>{esc(st.source)}</td>"
                    f"<td>{'yes' if st.usable else 'no'}</td><td class=q>{quote}</td></tr>")
    return ("<p>Life statements (what calibrated rates rest on):</p><table><tr><th>part</th><th>kind</th>"
            "<th>value</th><th>source</th><th>used</th><th>quote or reference</th></tr>"
            f"{''.join(rows)}</table>")


def verification_table(lls: Sequence[LoadedLineage]) -> str:
    rows = []
    for ll in lls:
        tq = ll.report.twin_quality
        if tq is None:
            continue
        legacy = (f"{tq.legacy.dynamic_pass_rate:.0%} of {tq.legacy.components}"
                  if tq.legacy and tq.legacy.dynamic_pass_rate is not None else "–")
        new = f"{tq.dynamic_pass_rate:.0%} of {tq.components}" if tq.dynamic_pass_rate is not None else "–"
        rows.append(f"<tr><td>{esc(ll.spec.label)}</td><td>{esc(new)}</td><td>{esc(legacy)}</td>"
                    f"<td>{'passed' if tq.passed else 'failed'}</td></tr>")
    return ("<table><tr><th>machine</th><th>dynamic checks passed (components)</th><th>legacy prototype</th>"
            f"<th>gate</th></tr>{''.join(rows)}</table>")


def scope_table(ll: LoadedLineage) -> str:
    scope = ll.load(ll._ref(("scope", "scope")), Scope)
    order = {"full_model": 0, "connection_only": 1, "excluded": 2}
    rows = [f"<tr><td>{esc(d.component)}</td><td>{esc(d.decision.replace('_', ' '))}</td><td>{esc(d.reason)}</td>"
            f"</tr>" for d in sorted(scope.decisions, key=lambda d: (order[d.decision], d.component))]
    return (f"<p>{esc(ll.spec.label)}: {sum(d.decision == 'full_model' for d in scope.decisions)} of "
            f"{len(scope.decisions)} components get a full model.</p><details><summary>every decision</summary>"
            f"<div class=wrap><table><tr><th>component</th><th>decision</th><th>reason</th></tr>{''.join(rows)}"
            "</table></div></details>")


def _img(svg: bytes, alt: str) -> str:
    return f'<img alt="{esc(alt)}" src="data:image/svg+xml;base64,{base64.b64encode(svg).decode()}">'


def unit_selector(spec: FailureTimeSpec, ctx: Context) -> str:
    """F3's drawing for every unit, the rule's unit selected first."""
    ll = ctx.lineage(spec.lineage)
    records = rul_records(ll, load_truth(ll), spec.estimators, ctx.settings.evaluate)
    chosen = choose_unit(records.get((spec.estimators[0], spec.quantity), []), spec.unit, spec.estimators[0],
                         spec.quantity, spec.unit_rule).unit
    options, panels = [], []
    for d in f3.extract_units(ctx, ll, records, spec.quantity, spec.estimators):
        with plt.rc_context(rc(ctx.style)):
            svg = encode(f3.render(d, spec.estimators[0], ctx.style, spec.width), ctx.style)["svg"]
        on = " on" if d.unit == chosen else ""
        sel = " selected" if d.unit == chosen else ""
        options.append(f'<option value="{esc(d.unit)}"{sel}>{esc(d.unit)}{" (F3)" if on else ""}</option>')
        panels.append(f'<div class="unit{on}" id="u-{esc(d.unit)}">{_img(svg, f"{spec.id} {d.unit}")}</div>')
    return (f'<p><label>Unit: <select id=unit>{"".join(options)}</select></label> '
            f"<span class=lede>every synthetic unit of this machine; F3 shows the one its rule picks</span></p>"
            f"{''.join(panels)}<script>{JS}</script>")


def build_report(spec: ReportSpec, ctx: Context, figures: dict[str, tuple[FigureBase, bytes]], commit: str) -> str:
    """The page; `figures`: id -> (its spec, its SVG bytes as just built)."""
    lls = [ctx.lineage(n) for n in spec.lineages]
    tables = {"provenance": "".join(provenance_table(ll, spec.quote_max_chars) for ll in lls),
              "verification": verification_table(lls),
              "scope": scope_table(ctx.lineage(spec.scope_lineage))}
    fan = figures[spec.selector_figure][0]
    if not isinstance(fan, FailureTimeSpec):
        raise ValueError(f"report: {spec.selector_figure} is not a failure_time figure")
    tables["units"] = unit_selector(fan, ctx)
    parts = [f"<h1>{esc(spec.title)}</h1><p class=lede>{esc(spec.subtitle)}</p>", _summary(lls)]
    for s in SECTIONS:
        parts.append(f"<h2>{esc(s.title)}</h2><p class=lede>{esc(s.lede)}</p>")
        for fid in s.figures:
            if fid not in figures:
                parts.append(f"<p class=note>{esc(fid)} is not in this figure spec.</p>")
                continue
            f, svg = figures[fid]
            if fid == spec.selector_figure:
                continue  # shown through the selector
            parts.append(f"<figure>{_img(svg, f.id)}<figcaption><b>{esc(f.id)}.</b> {esc(_flat(f.caption))}"
                         "</figcaption></figure>")
        if s.table:
            if s.table == "units":
                parts.append(f"<figcaption><b>{esc(fan.id)}.</b> {esc(_flat(fan.caption))}</figcaption>")
            parts.append(tables[s.table])
    parts.append(f"<footer>Built by <code>kaful figures build</code> from commit {esc(commit[:12])}, from saved "
                 "artifacts only; every figure's inputs (path and sha256) are in figures_manifest.json, and its "
                 "plotted numbers in the CSV beside it.</footer>")
    return ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(spec.title)}</title><style>{CSS}</style></head><body><main>{''.join(parts)}"
            "</main></body></html>\n")


def _summary(lls: Sequence[LoadedLineage]) -> str:
    items = "".join(f"<li><b>{esc(ll.spec.label)}:</b> {esc(_first(ll.report))}</li>" for ll in lls)
    notes = "".join(f"<p>{esc(d)}</p>" for d in EVALUATION_DISCLAIMERS)
    return f"<ul>{items}</ul><div class=note>{notes}</div>"


def _first(report: EvaluationReport) -> str:
    return report.summary[0] if report.summary else "not evaluated end to end"


def _flat(text: str) -> str:
    return " ".join(text.split())
