"""
demo/build_report.py
--------------------
Build a self-contained visual demo report from EXISTING checkpoint files.

No API calls, no pipeline reruns, no third-party packages — charts are inline SVG.
Opens offline in any browser.

Usage:
    python demo/build_report.py                      # eversys_coffee_machine
    python demo/build_report.py <machine_id>

Output: demo/report.html  (gitignored — contains manual excerpts)
"""

import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import get_machine_config  # paths only — no API clients

MACHINE_ID      = sys.argv[1] if len(sys.argv) > 1 else "eversys_coffee_machine"
GROUNDING_FIELD = "extraction_time"   # preferred example for section 2
HEALTH_PROFILE  = "heavy_commercial"  # profile used for the health chart
OUT_PATH        = ROOT / "demo" / "report.html"

ACCENT = "#0f766e"
INK    = "#1f2328"
MUTED  = "#6b7280"
LINE   = "#d0d7de"
PALE   = "#e6f2f1"
BAD    = "#b42318"
WARN   = "#b54708"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path):
    with open(path) as f:
        return json.load(f)


def esc(s) -> str:
    return html.escape(str(s))


def pretty(name: str) -> str:
    """'ceramic_grinders_component.burr_replacement_needed' -> 'Ceramic grinders: burr replacement needed'"""
    if "." in name:
        comp, item = name.split(".", 1)
        comp = comp.removesuffix("_component")
        if item.startswith(comp + "_"):
            item = item[len(comp) + 1:]
        return f"{comp.replace('_', ' ').capitalize()}: {item.replace('_', ' ')}"
    return name.replace("_", " ").capitalize()


def norm(s: str) -> str:
    return re.sub(r"[\s*#_`|]+", " ", s or "").strip().lower()


def nice_ticks(vmax: float, n: int = 5) -> list[float]:
    """Round axis ticks (0, 5000, 10000, ...) covering vmax."""
    raw = vmax / n
    mag = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [step * i for i in range(int(vmax // step) + 2)]


def nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def profiles(cfg) -> list[str]:
    order = {"light": 0, "medium": 1, "moderate": 1, "heavy": 2, "continuous": 2}
    names = [p.name for p in cfg["estimate_dir"].iterdir() if p.is_dir()] if cfg["estimate_dir"].exists() else []
    return sorted(names, key=lambda n: (order.get(n.split("_")[0], 9), n))


# ── Section 1: pipeline map ───────────────────────────────────────────────────

def pipeline_steps(cfg) -> list[tuple[str, list[tuple[str, bool]]]]:
    """(phase label, [(step label, checkpoint present)]) — presence checked on disk."""
    est = cfg["estimate_dir"]
    profs = profiles(cfg)

    def all_profiles(fname):
        return bool(profs) and all((est / p / fname).exists() for p in profs)

    return [
        ("Ingest", [
            ("Index manual", (cfg["rag_dir"] / "chroma.sqlite3").exists()),
        ]),
        ("Schema", [
            ("Retrieve passages", cfg["step_a_chunks"].exists()),
            ("Infer context",     cfg["step_b_context"].exists()),
            ("Extract (×N runs)", cfg["step_c_runs"].exists()),
            ("Normalize",         cfg["step_d_normalized"].exists()),
            ("Merge across runs", cfg["step_e_merged"].exists()),
            ("Finalize",          cfg["final_schema_path"].exists()),
        ]),
        ("Simulate", [
            ("Retrieve",          cfg["sim_step_a_context"].exists()),
            ("Plan",              cfg["sim_step_b_plan"].exists()),
            ("Generate simulator", cfg["sim_step_c_code"].exists()),
            ("Validate output",   cfg["sim_step_d_validation"].exists()),
        ]),
        ("Twin", [
            ("Comprehend",        cfg["twin_step_a_comprehension"].exists()),
            ("Physics per part",  nonempty_dir(cfg["twin_components_dir"])),
            ("Resolve",           cfg["twin_step_c_model"].exists()),
            ("Validate",          cfg["twin_step_d_validation"].exists()),
        ]),
        ("Triage", [
            ("Assign fields",     cfg["triage_step_a_assignments"].exists()),
            ("Triage parts",      cfg["triage_step_b_triage"].exists()),
            ("Filter",            cfg["triage_step_c_full_components"].exists()),
        ]),
        ("Codegen", [
            ("Spec",              nonempty_dir(cfg["codegen_specs_dir"])),
            ("Generate code",     nonempty_dir(cfg["codegen_code_dir"])),
            ("Runtime validate",  cfg["codegen_step_c_validation"].exists()),
            ("Composite model",   cfg["codegen_composite_path"].exists()
                                  and cfg["codegen_step_d_validation"].exists()),
        ]),
        ("Forecast", [
            ("Load data",         all_profiles("step_a_observations.json")),
            ("Ensemble",          all_profiles("step_b_estimated_states.json")),
            ("Remaining life",    all_profiles("step_c_rul.json")),
        ]),
    ]


def section_pipeline(cfg) -> str:
    phases = pipeline_steps(cfg)
    total  = sum(len(s) for _, s in phases)
    done   = sum(ok for _, s in phases for _, ok in s)
    boxes  = []
    for i, (label, steps) in enumerate(phases, 1):
        n_ok  = sum(ok for _, ok in steps)
        full  = n_ok == len(steps)
        items = "".join(
            f'<li class="{"ok" if ok else "missing"}">{"✓" if ok else "✗"} {esc(s)}</li>'
            for s, ok in steps
        )
        boxes.append(f"""
        <div class="phase {'full' if full else 'partial'}">
          <div class="phase-num">{i}</div>
          <div class="phase-name">{esc(label)}</div>
          <div class="phase-count">{n_ok}/{len(steps)} steps</div>
          <ul>{items}</ul>
        </div>""")
    flow = '<div class="arrow">→</div>'.join(boxes)
    return f"""
    <section>
      <h2>1 · The pipeline</h2>
      <div class="flow">{flow}</div>
      <p class="caption">Small steps, each saved and checked. {done} of {total} step checkpoints
      are on disk for this machine, so a rerun picks up where it left off.</p>
    </section>"""


# ── Section 2: grounding ──────────────────────────────────────────────────────

def section_grounding(cfg) -> str:
    schema = load(cfg["final_schema_path"])
    chunks = load(cfg["step_a_chunks"])

    def find_chunk(field):
        ev = norm(field.get("evidence"))
        if not ev:
            return None
        return next((c for c in chunks if ev in norm(c["text"])), None)

    matches = [(f, find_chunk(f)) for f in schema]
    n_verbatim = sum(1 for _, c in matches if c)

    pick = next(((f, c) for f, c in matches if c and f["normalized_name"] == GROUNDING_FIELD), None) \
        or next(((f, c) for f, c in matches if c), None)
    if not pick:
        return """<section><h2>2 · Grounded in the manual</h2>
        <p class="caption">No evidence quote could be matched to a retrieved passage.</p></section>"""

    field, chunk = pick
    evidence = field["evidence"]

    # Show the passage around the quote, with the quote highlighted
    text = chunk["text"]
    words = [re.escape(w) for w in re.findall(r"\w+", evidence)]
    m = re.search(r"\W+".join(words), text, re.I) if words else None
    if m:
        s, e = max(0, m.start() - 350), min(len(text), m.end() + 350)
        passage = (("… " if s > 0 else "") + esc(text[s:m.start()]) +
                   f"<mark>{esc(text[m.start():m.end()])}</mark>" +
                   esc(text[m.end():e]) + (" …" if e < len(text) else ""))
    else:
        passage = esc(text[:800]) + " …"

    pages = field.get("source_pages")
    pages = ", ".join(map(str, pages)) if isinstance(pages, list) else pages
    chunk_pages = f"{chunk.get('start_page')}–{chunk.get('end_page')}" if chunk.get("start_page") else "n/a"

    return f"""
    <section>
      <h2>2 · Grounded in the manual</h2>
      <div class="split">
        <div class="card">
          <div class="label">Extracted data field</div>
          <div class="big">{esc(field['normalized_name'])}</div>
          <table class="kv">
            <tr><td>Unit</td><td>{esc(field.get('unit') or '—')}</td></tr>
            <tr><td>Typical range</td><td>{esc(field.get('typical_range') or '—')}</td></tr>
            <tr><td>Cited pages</td><td>{esc(pages or '—')}</td></tr>
            <tr><td>Evidence quote</td><td>“{esc(evidence)}”</td></tr>
          </table>
        </div>
        <div class="card passage">
          <div class="label">Retrieved manual passage · pages {esc(chunk_pages)}</div>
          <div class="passage-text">{passage}</div>
        </div>
      </div>
      <p class="caption">Every fact traces to the manual. For {n_verbatim} of {len(schema)} schema fields
      the evidence quote appears word-for-word in the passage it was retrieved from;
      the rest are paraphrased or combined from several lines.</p>
    </section>"""


# ── Section 3: validation ─────────────────────────────────────────────────────

CHECKS = [
    ("Loads &amp; runs",        ["Failed to import", "initialize()", "output()"]),
    ("Starts healthy",          ["event_state()"]),
    ("1,000 steps, no NaN/inf", ["NaN/inf in states", "Step ", "No dx()"]),
    ("Actually wears",          ["did not degrade"]),
    ("Sane wear rate",          ["degraded too fast", "barely changed"]),
]


def section_validation(cfg) -> str:
    reports = load(cfg["codegen_step_c_validation"])
    checked = cfg["codegen_step_c_validation"].stat().st_mtime
    head = "".join(f"<th>{c}</th>" for c, _ in CHECKS)
    rows, n_pass, edited = [], 0, []
    for name, r in reports.items():
        code = cfg["codegen_code_dir"] / f"{name}.py"
        stale = code.exists() and code.stat().st_mtime > checked
        if stale:
            edited.append(name)
        issues   = r.get("issues", [])
        warnings = r.get("warnings", [])
        fatal    = any("Failed to import" in i for i in issues)
        cells = []
        for _, keys in CHECKS:
            hit_issue = fatal or any(any(k in i for k in keys) for i in issues)
            hit_warn  = any(any(k in w for k in keys) for w in warnings)
            if hit_issue:
                cells.append('<td class="fail">✗</td>')
            elif hit_warn:
                cells.append('<td class="warn">!</td>')
            else:
                cells.append('<td class="pass">✓</td>')
        n_pass += r["status"] == "pass"
        note = "<span class='stale'>code changed after this check</span>" if stale else ""
        rows.append(f"<tr><td class='name'>{esc(pretty(name))}{note}</td>{''.join(cells)}</tr>")

    return f"""
    <section>
      <h2>3 · Generated code is tested before use</h2>
      <table class="grid">
        <thead><tr><th>Generated component</th>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="caption">Each generated component is executed and checked before the full machine
      model is assembled. {n_pass} of {len(reports)} passed every check in the saved run.
      {f"{len(edited)} component files were edited after that run — re-check them live." if edited else ""}</p>
    </section>"""


# ── Section 4: health over time ───────────────────────────────────────────────

def health_series(cfg, profile: str):
    """Health-like states: bounded in [0, 1] and starting near 1.0."""
    data = load(cfg["estimate_dir"] / profile / "step_b_estimated_states.json")
    est, unc = data["estimated_states"], data["state_uncertainty"]
    keys = [k for k in est[0] if k != "t"]
    out = []
    for k in keys:
        vals = [row[k] for row in est]
        if 0.95 <= vals[0] <= 1.05 and all(-0.05 <= v <= 1.05 for v in vals):
            sd = [row[k] for row in unc]
            out.append((k, [row["t"] for row in est], vals, sd))
    return out


def svg_health(series) -> str:
    W, H, L, R, T, B = 1200, 470, 70, 440, 20, 70
    pw, ph = W - L - R, H - T - B
    ticks = nice_ticks(max(max(s[1]) for s in series))
    tmax = ticks[-1]
    X = lambda t: L + pw * t / tmax
    Y = lambda v: T + ph * (1 - max(0.0, min(1.0, v)))
    shades = ["#0f766e", "#5b8f8a", "#8a9aa8", "#4b5563", "#a3b8b5", "#6b7280", "#94a3b8"]

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'<text x="{L-10}" y="{Y(v)+5}" text-anchor="end" class="tick">{v:.0%}</text>')
    for t in ticks:
        parts.append(f'<text x="{X(t)}" y="{T+ph+26}" text-anchor="middle" class="tick">{t:,.0f}</text>')
    parts.append(f'<text x="{L+pw/2}" y="{H-6}" text-anchor="middle" class="axis">simulated time steps</text>')

    labels = []
    for i, (k, ts, vals, sd) in enumerate(series):
        c = shades[i % len(shades)]
        up = [f"{X(t):.1f},{Y(v + 1.645*s):.1f}" for t, v, s in zip(ts, vals, sd)]
        lo = [f"{X(t):.1f},{Y(v - 1.645*s):.1f}" for t, v, s in zip(ts, vals, sd)]
        parts.append(f'<polygon points="{" ".join(up + lo[::-1])}" fill="{c}" opacity="0.15"/>')
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(ts, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3"/>')
        labels.append((Y(vals[-1]), c, f"{pretty(k)} · {vals[-1]:.0%}"))

    # End-of-line labels, nudged apart so they don't overlap
    labels.sort()
    last = -1e9
    for y, c, text in labels:
        y = max(y, last + 22)
        last = y
        parts.append(f'<text x="{L+pw+12}" y="{y+5}" class="lbl" fill="{c}">{esc(text)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def section_health(cfg) -> str:
    profs = profiles(cfg)
    profile = HEALTH_PROFILE if HEALTH_PROFILE in profs else (profs[-1] if profs else None)
    series = health_series(cfg, profile) if profile else []
    if not series:
        return ""
    return f"""
    <section>
      <h2>4 · Wear accumulating over time</h2>
      {svg_health(series)}
      <p class="caption">The generated model's health states over the <b>{esc(pretty(profile))}</b>
      run (100% = new). Line = ensemble average; shaded = 90% band across 100 runs
      started from slightly different initial conditions.</p>
    </section>"""


# ── Section 5: remaining-life forecast ────────────────────────────────────────

def section_rul(cfg) -> str:
    profs = profiles(cfg)
    if not profs:
        return ""
    rul = {p: load(cfg["estimate_dir"] / p / "step_c_rul.json").get("predictions", {}) for p in profs}
    # How long each profile ran before the forecast starts (the profiles' only real difference)
    age = {p: load(cfg["estimate_dir"] / p / "step_b_estimated_states.json")["estimated_states"][-1]["t"]
           for p in profs}

    # Chart: events with a finite forecast in every profile (skip near-immediate ones)
    events = [e for e in rul[profs[0]]
              if all((rul[p].get(e) or {}).get("rul_mean") not in (None, 0) for p in profs)
              and min(rul[p][e]["rul_mean"] for p in profs) > 100]
    events.sort(key=lambda e: rul[profs[-1]][e]["rul_mean"])

    W, L, R, rowh, gap = 1200, 400, 40, 18, 16
    grp = rowh * len(profs) + gap
    H = 40 + grp * len(events) + 70
    ticks = nice_ticks(max(rul[p][e].get("rul_max", rul[p][e]["rul_mean"]) for p in profs for e in events))
    vmax = ticks[-1]
    X = lambda v: L + (W - L - R) * v / vmax
    shades = {0: "#b6d4d1", 1: "#5b8f8a", 2: ACCENT}

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img">']
    for v in ticks:
        parts.append(f'<line x1="{X(v)}" x2="{X(v)}" y1="30" y2="{H-70}" stroke="{LINE}"/>')
        parts.append(f'<text x="{X(v)}" y="{H-44}" text-anchor="middle" class="tick">{v:,.0f}</text>')
    parts.append(f'<text x="{L+(W-L-R)/2}" y="{H-6}" text-anchor="middle" class="axis">'
                 f'forecast time steps until the event</text>')
    for gi, e in enumerate(events):
        y0 = 36 + gi * grp
        parts.append(f'<text x="{L-14}" y="{y0 + rowh*len(profs)/2 + 5}" text-anchor="end" class="lbl">{esc(pretty(e))}</text>')
        for pi, p in enumerate(profs):
            s = rul[p][e]
            y = y0 + pi * rowh
            c = shades.get(pi, ACCENT)
            parts.append(f'<rect x="{L}" y="{y}" width="{X(s["rul_mean"])-L:.1f}" height="{rowh-4}" fill="{c}"/>')
            if s.get("rul_max", 0) > s.get("rul_min", 0):
                ym = y + (rowh - 4) / 2
                parts.append(f'<line x1="{X(s["rul_min"]):.1f}" x2="{X(s["rul_max"]):.1f}" y1="{ym}" y2="{ym}" '
                             f'stroke="{INK}" stroke-width="2"/>')
    parts.append("</svg>")

    legend = "".join(f'<span class="key"><i style="background:{shades.get(i, ACCENT)}"></i>'
                     f'{esc(pretty(p))} · after {age[p]:,} steps</span>'
                     for i, p in enumerate(profs))

    # Full table so nothing is hidden by the chart's selection
    def cell(s):
        if not s or "error" in s:
            return '<td class="fail">error</td>'
        if s.get("already_triggered"):
            return '<td class="warn">already triggered</td>'
        if s.get("rul_mean") is None:
            return '<td class="muted">beyond horizon</td>'
        rng = f" <span class='muted'>({s['rul_min']:,.0f}–{s['rul_max']:,.0f})</span>" \
              if s.get("rul_max", 0) > s.get("rul_min", 0) else ""
        return f"<td>{s['rul_mean']:,.0f}{rng}</td>"

    all_events = list(rul[profs[0]].keys())
    rows = "".join(f"<tr><td class='name'>{esc(pretty(e))}</td>{''.join(cell(rul[p].get(e)) for p in profs)}</tr>"
                   for e in all_events)
    head = "".join(f"<th>{esc(pretty(p))}</th>" for p in profs)

    return f"""
    <section>
      <h2>5 · Remaining-life forecast</h2>
      <div class="legend">{legend}<span class="key"><b class="whisker"></b>min–max across Monte Carlo samples</span></div>
      {"".join(parts)}
      <p class="caption">Forward Monte Carlo forecast (ProgPy): steps remaining until each event, starting
      from the end of each simulated run. All three use the same model and the same per-step load,
      so a machine that has already run longer has less life left — the bars shift by roughly the
      difference in run length. Shown: events with a forecast in every run.
      Not yet validated against real failure data.</p>
      <details><summary>All {len(all_events)} events</summary>
        <table class="grid small"><thead><tr><th>Event</th>{head}</tr></thead><tbody>{rows}</tbody></table>
      </details>
    </section>"""


# ── Section 6: triage ─────────────────────────────────────────────────────────

def section_triage(cfg) -> str:
    triage = load(cfg["triage_step_b_triage"])["triaged_components"]
    modeled = {c["name"] for c in load(cfg["triage_step_c_full_components"])}
    groups = {"Fully modeled": [], "Simple state": [], "Excluded": []}
    for c in triage:
        if c["name"] in modeled:
            groups["Fully modeled"].append((c, c.get("reasoning", "")))
        elif c["decision"] == "simple_state":
            groups["Simple state"].append((c, c.get("reasoning", "")))
        else:
            why = c.get("reasoning", "")
            if c["decision"] == "full_component":
                why = "Manually excluded in machine config (queries.py). " + why
            groups["Excluded"].append((c, why))

    cols = []
    for title, items in groups.items():
        lis = "".join(f"<li><b>{esc(pretty(c['name']))}</b><span>{esc(why.split('. ')[0].rstrip('.'))}.</span></li>"
                      for c, why in items)
        cols.append(f'<div class="card tri"><div class="label">{title}</div>'
                    f'<div class="big">{len(items)}</div><ul>{lis}</ul></div>')
    return f"""
    <section>
      <h2>6 · Not everything gets a physics model</h2>
      <div class="tri-row">{''.join(cols)}</div>
      <p class="caption">Each component is triaged: full degradation model, simple state, or left out —
      with the model's reason recorded.</p>
    </section>"""


# ── Page ──────────────────────────────────────────────────────────────────────

CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #fff; color: {INK};
       font: 20px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
main {{ max-width: 1240px; margin: 0 auto; padding: 32px 24px 80px; }}
header h1 {{ font-size: 44px; margin: 0 0 4px; letter-spacing: -0.5px; }}
header p {{ color: {MUTED}; margin: 0; font-size: 22px; }}
section {{ margin-top: 64px; padding-top: 32px; border-top: 1px solid {LINE}; }}
h2 {{ font-size: 32px; margin: 0 0 24px; }}
.caption {{ font-size: 22px; color: {INK}; margin: 20px 0 0; max-width: 980px; }}
.muted {{ color: {MUTED}; }}
.flow {{ display: flex; align-items: stretch; gap: 6px; }}
.arrow {{ align-self: center; color: {MUTED}; font-size: 22px; }}
.phase {{ flex: 1; border: 2px solid {LINE}; border-radius: 10px; padding: 12px 10px; min-width: 0; }}
.phase.full {{ border-color: {ACCENT}; background: {PALE}; }}
.phase-num {{ font-size: 14px; color: {MUTED}; }}
.phase-name {{ font-weight: 700; font-size: 20px; }}
.phase-count {{ font-size: 15px; color: {ACCENT}; font-weight: 600; margin-bottom: 6px; }}
.phase ul {{ list-style: none; margin: 0; padding: 0; font-size: 13px; line-height: 1.4; }}
.phase li.ok {{ color: {INK}; }}
.phase li.missing {{ color: {BAD}; }}
.split {{ display: grid; grid-template-columns: 1fr 1.3fr; gap: 20px; }}
.card {{ border: 1px solid {LINE}; border-radius: 10px; padding: 20px; }}
.label {{ font-size: 15px; text-transform: uppercase; letter-spacing: .06em; color: {MUTED}; margin-bottom: 6px; }}
.big {{ font-size: 30px; font-weight: 700; color: {ACCENT}; margin-bottom: 12px; }}
.kv td {{ padding: 4px 12px 4px 0; vertical-align: top; }}
.kv td:first-child {{ color: {MUTED}; white-space: nowrap; }}
.passage-text {{ font: 16px/1.55 ui-monospace, Menlo, monospace; white-space: pre-wrap; max-height: 360px; overflow: auto; }}
mark {{ background: #fde68a; padding: 1px 2px; }}
table.grid {{ border-collapse: collapse; width: 100%; }}
.grid th, .grid td {{ padding: 10px 14px; border-bottom: 1px solid {LINE}; text-align: center; }}
.grid th {{ font-size: 16px; color: {MUTED}; font-weight: 600; }}
.grid td.name, .grid th:first-child {{ text-align: left; }}
.grid td.pass {{ color: {ACCENT}; font-weight: 700; font-size: 24px; }}
.grid td.fail {{ color: {BAD}; font-weight: 700; }}
.grid td.warn {{ color: {WARN}; font-weight: 700; }}
.stale {{ display: block; font-size: 14px; color: {WARN}; }}
.grid.small td, .grid.small th {{ font-size: 15px; padding: 6px 10px; }}
svg.chart {{ width: 100%; height: auto; display: block; }}
svg .tick {{ font-size: 15px; fill: {MUTED}; }}
svg .axis {{ font-size: 16px; fill: {MUTED}; }}
svg .lbl {{ font-size: 16px; fill: {INK}; font-weight: 600; }}
.legend {{ display: flex; gap: 24px; font-size: 17px; margin-bottom: 8px; flex-wrap: wrap; }}
.key i {{ display: inline-block; width: 14px; height: 14px; margin-right: 6px; vertical-align: -1px; }}
.whisker {{ display: inline-block; width: 22px; border-top: 2px solid {INK}; margin-right: 6px; vertical-align: 4px; }}
details {{ margin-top: 20px; font-size: 17px; }}
summary {{ cursor: pointer; color: {MUTED}; }}
.tri-row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
.tri ul {{ margin: 0; padding-left: 18px; font-size: 16px; }}
.tri li {{ margin-bottom: 8px; }}
.tri li span {{ display: block; color: {MUTED}; font-size: 14px; }}
"""


# ── Twin tab: interactive component view ──────────────────────────────────────

# Schematic positions (x, y) on a 1010×570 canvas; box size 180×84
TWIN_LAYOUT = {
    "bean_hoppers":     (40, 40),   "external":        (540, 40),  "touch_screen":    (790, 40),
    "ceramic_grinders": (40, 180),  "brewing_chamber": (290, 180), "milk_system":     (540, 180),
    "powder_unit":      (790, 180), "coffee_boiler":   (290, 320), "steam_boiler":    (540, 320),
    "cleaning_system":  (790, 320), "grounds_drawer":  (40, 460),  "water_pump":      (415, 460),
    "drip_tray":        (790, 460),
}
BOX_W, BOX_H = 180, 84


def _clip(cx, cy, tx, ty):
    """Point where the segment from a box centre toward (tx, ty) leaves the box."""
    dx, dy = tx - cx, ty - cy
    s = min((BOX_W / 2 + 6) / abs(dx) if dx else 1e9, (BOX_H / 2 + 6) / abs(dy) if dy else 1e9)
    return cx + dx * s, cy + dy * s


def twin_data(cfg) -> dict:
    """Everything the Twin tab needs, read from saved checkpoints only."""
    triage     = {c["name"]: c for c in load(cfg["triage_step_b_triage"])["triaged_components"]}
    modeled    = [c["name"] for c in load(cfg["triage_step_c_full_components"])]
    validation = load(cfg["codegen_step_c_validation"])
    model      = load(cfg["twin_step_c_model"])

    meta = {}
    for name, c in triage.items():
        kind = "full" if name in modeled else ("simple" if c["decision"] == "simple_state" else "excluded")
        spec_path = cfg["codegen_specs_dir"] / f"{name}.json"
        spec = load(spec_path) if kind == "full" and spec_path.exists() else {}
        meta[name] = {
            "label":  pretty(name),
            "kind":   kind,
            "reason": c.get("reasoning", ""),
            "valid":  validation.get(name, {}).get("status"),
            "units":  {s["name"]: s.get("unit") for s in spec.get("states", [])},
        }

    edges = []
    for fp in model.get("flow_paths", []):
        for sig in fp.get("boundary_signals", []):
            a, b = sig.get("from"), sig.get("to")
            if a in TWIN_LAYOUT and b in TWIN_LAYOUT:
                edges.append({"from": a, "to": b, "signal": sig.get("signal_name"),
                              "substance": fp.get("substance"), "evidence": fp.get("evidence", "")})

    runs = {}
    for p in profiles(cfg):
        b = load(cfg["estimate_dir"] / p / "step_b_estimated_states.json")
        r = load(cfg["estimate_dir"] / p / "step_c_rul.json").get("predictions", {})
        comps = {n: {"events": {}, "states": {}, "rul": {}} for n in modeled}
        for row_key, bucket in (("estimated_events", "events"), ("estimated_states", "states")):
            keys = [k for k in b[row_key][0] if k != "t"]
            for k in keys:
                comp, item = k.split(".", 1)
                comp = comp.removesuffix("_component")
                if comp in comps:
                    comps[comp][bucket][item] = [None if row.get(k) is None else round(row[k], 5)
                                                 for row in b[row_key]]
        for k, s in r.items():
            comp, item = k.split(".", 1)
            comp = comp.removesuffix("_component")
            if comp in comps:
                comps[comp]["rul"][item] = s
        runs[p] = {"label": pretty(p), "times": [row["t"] for row in b["estimated_states"]], "comps": comps}

    return {"meta": meta, "edges": edges, "runs": runs}


def section_twin(cfg) -> str:
    data = twin_data(cfg)
    parts = [f'<svg viewBox="0 0 1010 570" class="chart twin-svg" role="img">',
             f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker></defs>']

    for e in data["edges"]:
        (ax, ay), (bx, by) = TWIN_LAYOUT[e["from"]], TWIN_LAYOUT[e["to"]]
        acx, acy, bcx, bcy = ax + BOX_W / 2, ay + BOX_H / 2, bx + BOX_W / 2, by + BOX_H / 2
        x1, y1 = _clip(acx, acy, bcx, bcy)
        x2, y2 = _clip(bcx, bcy, acx, acy)
        parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{MUTED}" '
                     f'stroke-width="2" marker-end="url(#arr)"><title>{esc(e["substance"])}: {esc(e["signal"])}'
                     f' — manual: “{esc(e["evidence"])}”</title></line>')

    for name, (x, y) in TWIN_LAYOUT.items():
        if name == "external":
            parts.append(f'<g><rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="42" fill="#fff" '
                         f'stroke="{INK}" stroke-width="2"/><text x="{x+BOX_W/2}" y="{y+50}" text-anchor="middle" '
                         f'class="twin-name">☕ Dispense</text></g>')
            continue
        m = data["meta"].get(name)
        if not m:
            continue
        cls = {"full": "comp full", "simple": "comp simple", "excluded": "comp excluded"}[m["kind"]]
        sub = {"full": "", "simple": "simple state", "excluded": "excluded"}[m["kind"]]
        parts.append(f'<g class="{cls}" data-name="{name}"><rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10"/>'
                     f'<text x="{x+BOX_W/2}" y="{y+36}" text-anchor="middle" class="twin-name">{esc(m["label"])}</text>'
                     f'<text x="{x+BOX_W/2}" y="{y+62}" text-anchor="middle" class="twin-sub">{sub}</text></g>')
    parts.append("</svg>")

    runs = list(data["runs"])
    opts = "".join(f'<option value="{p}"{" selected" if p == runs[-1] else ""}>{esc(pretty(p))}</option>' for p in runs)
    payload = json.dumps(data).replace("</", "<\\/")

    return f"""
    <section class="twin">
      <h2>The machine twin</h2>
      <div class="twin-controls">
        <label>Run <select id="tw-run">{opts}</select></label>
        <label class="grow">Steps run <input id="tw-t" type="range" min="0" value="0"> <b id="tw-tval"></b></label>
        <span class="key"><i class="sw healthy"></i>healthy</span>
        <span class="key"><i class="sw degrading"></i>nearing alarm</span>
        <span class="key"><i class="sw past"></i>alarm reached</span>
      </div>
      <div class="twin-grid">
        <div>{''.join(parts)}</div>
        <aside id="tw-panel" class="card"></aside>
      </div>
      <p class="caption">Each block is a component the pipeline modeled from the manual; arrows are the flows it
      extracted (hover for the manual quote). Colour = how close the component's nearest alarm is, from the
      saved ensemble run (1 = far, 0 = reached). Click a component for detail.</p>
      <script id="tw-data" type="application/json">{payload}</script>
    </section>"""


TWIN_JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('tw-data').textContent);
  const runSel = document.getElementById('tw-run'), slider = document.getElementById('tw-t');
  const tval = document.getElementById('tw-tval'), panel = document.getElementById('tw-panel');
  let selected = 'milk_system';
  const pretty = s => s.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  const fmt = v => v == null ? '—' : (Math.abs(v) >= 100 ? v.toFixed(1) : (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3));
  const status = v => v <= 0 ? 'past' : (v < 0.5 ? 'degrading' : 'healthy');

  function nearest(comp, i) {
    const evs = Object.entries(comp.events);
    if (!evs.length) return null;
    return Math.min(...evs.map(([, s]) => s[i] == null ? 0 : s[i]));
  }

  function rulText(s, endT) {
    if (!s || s.error) return 'error';
    if (s.already_triggered) return 'already reached';
    if (s.rul_mean == null) return 'beyond horizon';
    const rng = s.rul_max > s.rul_min ? ` (${Math.round(s.rul_min).toLocaleString()}–${Math.round(s.rul_max).toLocaleString()})` : '';
    const n = Math.round(s.rul_mean);
    return `${n.toLocaleString()} ${n === 1 ? 'step' : 'steps'}${rng}`;
  }

  function render() {
    const run = D.runs[runSel.value], i = +slider.value, t = run.times[i];
    tval.textContent = t.toLocaleString();
    document.querySelectorAll('g.comp').forEach(g => {
      const name = g.dataset.name, comp = run.comps[name];
      g.classList.remove('healthy', 'degrading', 'past', 'sel');
      if (comp) g.classList.add(status(nearest(comp, i)));
      if (name === selected) g.classList.add('sel');
    });

    const m = D.meta[selected], comp = run.comps[selected];
    if (!comp) {
      panel.innerHTML = `<div class="label">${m.kind === 'simple' ? 'Simple state' : 'Excluded'}</div>
        <div class="big">${m.label}</div><p class="muted">${m.reason}</p>
        <p>Not given a physics model, so there is no wear curve or forecast.</p>`;
      return;
    }
    // Spec states only — skip coupling values ProgPy adds for connected ports
    const states = Object.entries(comp.states).filter(([k]) => k in m.units).map(([k, s]) =>
      `<tr><td>${pretty(k)}</td><td>${fmt(s[0])} → <b>${fmt(s[i])}</b></td></tr>`).join('');
    const events = Object.entries(comp.events).map(([k, s]) => {
      const v = s[i] == null ? 0 : s[i], st = status(v);
      return `<tr><td>${pretty(k)}</td><td><span class="bar"><span class="${st}" style="width:${Math.max(0, Math.min(1, v)) * 100}%"></span></span>
        ${s[i] == null ? 'n/a' : v.toFixed(2)}</td><td>${rulText(comp.rul[k])}</td></tr>`;
    }).join('');
    const endT = run.times[run.times.length - 1].toLocaleString();
    panel.innerHTML = `<div class="label">Fully modeled · validation: ${m.valid || 'n/a'}</div>
      <div class="big">${m.label}</div>
      <h4>Health states <span class="muted">(start → step ${t.toLocaleString()})</span></h4>
      <table class="kv">${states}</table>
      <h4>Alarms <span class="muted">(1 = far · 0 = reached)</span></h4>
      <table class="kv ev"><tr><th></th><th>now</th><th>remaining after step ${endT}</th></tr>${events}</table>`;
  }

  function setRun() {
    slider.max = D.runs[runSel.value].times.length - 1;
    slider.value = slider.max;
    render();
  }
  document.querySelectorAll('g.comp').forEach(g => g.addEventListener('click', () => { selected = g.dataset.name; render(); }));
  runSel.addEventListener('change', setRun);
  slider.addEventListener('input', render);
  document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x === b));
    document.querySelectorAll('.pane').forEach(p => p.hidden = p.id !== b.dataset.pane);
  }));
  setRun();
})();
"""

TWIN_CSS = f"""
.tabs {{ display: flex; gap: 8px; margin-top: 28px; }}
.tab {{ font: inherit; font-size: 18px; padding: 8px 20px; border: 1px solid {LINE}; border-radius: 999px;
       background: #fff; color: {MUTED}; cursor: pointer; }}
.tab.on {{ background: {ACCENT}; border-color: {ACCENT}; color: #fff; }}
.twin-controls {{ display: flex; gap: 24px; align-items: center; flex-wrap: wrap; font-size: 17px; margin-bottom: 12px; }}
.twin-controls .grow {{ flex: 1; display: flex; gap: 10px; align-items: center; min-width: 260px; }}
.twin-controls input[type=range] {{ flex: 1; accent-color: {ACCENT}; }}
.twin-controls select {{ font: inherit; font-size: 16px; padding: 4px 8px; }}
.sw {{ border-radius: 3px; border: 2px solid; }}
.sw.healthy {{ background: {PALE}; border-color: {ACCENT}; }}
.sw.degrading {{ background: #fef3c7; border-color: {WARN}; }}
.sw.past {{ background: #fee4e2; border-color: {BAD}; }}
.twin-grid {{ display: grid; grid-template-columns: 1.55fr 1fr; gap: 20px; align-items: start; }}
.twin-svg .twin-name {{ font-size: 21px; font-weight: 700; fill: {INK}; pointer-events: none; }}
.twin-svg .twin-sub {{ font-size: 16px; fill: {MUTED}; pointer-events: none; }}
.twin-svg g.comp rect {{ fill: #fff; stroke: {LINE}; stroke-width: 2; }}
.twin-svg g.comp {{ cursor: pointer; }}
.twin-svg g.simple rect {{ stroke-dasharray: 6 4; }}
.twin-svg g.excluded rect {{ fill: #f6f8fa; stroke-dasharray: 2 4; }}
.twin-svg g.healthy rect {{ fill: {PALE}; stroke: {ACCENT}; }}
.twin-svg g.degrading rect {{ fill: #fef3c7; stroke: {WARN}; }}
.twin-svg g.past rect {{ fill: #fee4e2; stroke: {BAD}; }}
.twin-svg g.sel rect {{ stroke-width: 5; }}
#tw-panel {{ font-size: 16px; min-height: 420px; }}
#tw-panel h4 {{ margin: 18px 0 6px; font-size: 17px; }}
#tw-panel .kv td, #tw-panel .kv th {{ font-size: 15px; padding: 4px 10px 4px 0; text-align: left; }}
#tw-panel .kv th {{ color: {MUTED}; font-weight: 600; }}
.bar {{ display: inline-block; width: 70px; height: 9px; background: #eef1f4; border-radius: 5px;
       vertical-align: middle; margin-right: 6px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; }}
.bar .healthy {{ background: {ACCENT}; }} .bar .degrading {{ background: {WARN}; }} .bar .past {{ background: {BAD}; }}
"""


def main():
    cfg = get_machine_config(MACHINE_ID)
    sections = [
        section_pipeline(cfg),
        section_grounding(cfg),
        section_validation(cfg),
        section_health(cfg),
        section_rul(cfg),
        section_triage(cfg),
    ]
    title = pretty(MACHINE_ID)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kaful Demo Report</title><style>{CSS}{TWIN_CSS}</style></head>
<body><main>
<header><h1>Kaful · {esc(title)}</h1>
<p>From a machine manual to a model of how it wears out, with a remaining-life forecast.
Built from saved pipeline checkpoints.</p>
<nav class="tabs"><button class="tab on" data-pane="pane-twin">Twin</button>
<button class="tab" data-pane="pane-report">How it was built</button></nav></header>
<div class="pane" id="pane-twin">{section_twin(cfg)}</div>
<div class="pane" id="pane-report" hidden>{''.join(sections)}</div>
</main><script>{TWIN_JS}</script></body></html>"""
    OUT_PATH.write_text(page)
    print(f"✓ Wrote {OUT_PATH.relative_to(ROOT)} ({len(page)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
