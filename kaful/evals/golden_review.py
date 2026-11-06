"""Golden review page: each golden case's source page beside its expected values.

The reviewer marks every expected value correct, wrong (with a correction) or unsure,
and lists values the case is missing. The page exports a `golden_review.toml` that is
saved next to the eval set and applied to the case files by hand. The HTML embeds page
images from the manuals, so it is written under the gitignored outputs/ directory and
never published.
"""

import base64
import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from kaful.evals.models import EvalCase, EvalSet
from kaful.paths import resolve_in

REVIEW_DPI = 130
INPUT_EXCERPT_CHARS = 3000
REVIEW_FILE = "golden_review.toml"


@dataclass(frozen=True)
class ExpectedItem:
    """One expected value of one golden spec, as the reviewer sees it."""

    golden: int
    item: int
    metric: str
    compare: str
    field: str
    text: str


def expected_items(case: EvalCase) -> list[ExpectedItem]:
    """Every expected value of a case, one per list item (or key) so each can be judged alone."""
    items = []
    for g, spec in enumerate(case.golden):
        values: list[Any]
        if isinstance(spec.expected, dict):
            values = [f"{k} → {v}" for k, v in sorted(spec.expected.items())]
        elif isinstance(spec.expected, list):
            values = list(spec.expected)
        else:
            values = [spec.expected]
        for i, value in enumerate(values):
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            items.append(ExpectedItem(g, i, spec.metric, spec.compare, spec.field, text))
    return items


def _page_image(evalset: EvalSet, case: EvalCase) -> str | None:
    if case.page is None or not evalset.page_available(case):
        return None
    path = resolve_in(evalset.machine_dir(case.page.machine), case.page.path)
    with fitz.open(path) as doc:
        pixmap = doc[case.page.page - 1].get_pixmap(dpi=REVIEW_DPI, alpha=False)
        return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def _source(evalset: EvalSet, case: EvalCase) -> str:
    image = _page_image(evalset, case)
    if image is not None:
        assert case.page is not None
        return f'<img alt="{html.escape(case.page.path)} page {case.page.page}" src="data:image/png;base64,{image}">'
    if case.page is not None:
        return "<p class='warn'>Source page not on disk.</p>"
    text = "\n\n".join(f"[{k}]\n{v}" for k, v in evalset.inputs(case).items())
    return f"<pre>{html.escape(text[:INPUT_EXCERPT_CHARS])}</pre>"


def _card(evalset: EvalSet, case: EvalCase) -> str:
    rows = []
    for it in expected_items(case):
        key = f"{case.id}/{it.golden}/{it.item}"
        buttons = "".join(f"<button data-k='{key}' data-v='{v}'>{v}</button>" for v in ("correct", "wrong", "unsure"))
        rows.append(f"""<li><code class="value">{html.escape(it.text)}</code>
<span class="meta">{html.escape(it.metric)} · {html.escape(it.compare)} · {html.escape(it.field)}</span>
<div class="buttons">{buttons}</div><input class="fix" data-k="{key}" placeholder="correction (if wrong)"></li>""")
    where = (f"{case.page.machine} · {case.page.path} · page {case.page.page}" if case.page else "text inputs")
    return f"""<section class="card" id="{case.id}"><header><b>{html.escape(case.id)}</b> · {html.escape(where)}</header>
<p class="desc">{html.escape(case.description)}</p>
<div class="body"><div class="src">{_source(evalset, case)}</div><div class="side"><ol>{''.join(rows)}</ol>
<textarea class="missing" data-c="{case.id}" placeholder="values on the page the case should also expect (one per line)"></textarea>
</div></div></section>"""


STYLE = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e3e1da; --accent:#2f5d8a; --on:#ffffff;
  --bad:#a33a2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#171716; --fg:#ecebe6; --muted:#a3a29c; --card:#21211f; --line:#3a3936; --accent:#7fb0e0; --on:#10202e;
  --bad:#e08a7a; } }
:root[data-theme="dark"] { --bg:#171716; --fg:#ecebe6; --muted:#a3a29c; --card:#21211f; --line:#3a3936;
  --accent:#7fb0e0; --on:#10202e; --bad:#e08a7a; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto;
  max-width:1500px; padding:0 16px 80px; }
.bar { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0;
  display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:2; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:14px 0; padding:12px; }
.card.done { border-color:var(--accent); }
.body { display:flex; gap:14px; flex-wrap:wrap; } .src { flex:1.3; min-width:280px; }
.src img { width:100%; border:1px solid var(--line); } .side { flex:1; min-width:280px; }
ol { padding-left:20px; } li { margin-bottom:10px; } .value { font-size:13px; color:var(--fg); }
.meta, .desc { color:var(--muted); font-size:12px; } .meta { display:block; }
.buttons { display:flex; gap:6px; margin:4px 0; }
button { border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:6px; padding:4px 9px;
  cursor:pointer; } button.on { background:var(--accent); color:var(--on); border-color:var(--accent); }
button.on[data-v="wrong"] { background:var(--bad); border-color:var(--bad); }
input, textarea { width:100%; box-sizing:border-box; padding:5px; background:transparent; color:var(--fg);
  border:1px solid var(--line); border-radius:6px; font:inherit; } textarea { min-height:60px; }
pre { white-space:pre-wrap; font-size:12px; color:var(--muted); } .warn { color:var(--bad); }
"""

SCRIPT = """
const KEY = "kaful-golden-review-" + META.prompt_id + "-v" + META.version;
let state = {verdicts: {}, fixes: {}, missing: {}};
try { state = Object.assign(state, JSON.parse(localStorage.getItem(KEY) || "{}")); } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); }
function render() {
  document.querySelectorAll("button[data-k]").forEach(b =>
    b.classList.toggle("on", state.verdicts[b.dataset.k] === b.dataset.v));
  document.querySelectorAll(".fix").forEach(n => { if (document.activeElement !== n) n.value = state.fixes[n.dataset.k] || ""; });
  document.querySelectorAll(".missing").forEach(n => { if (document.activeElement !== n) n.value = state.missing[n.dataset.c] || ""; });
  let done = 0;
  CASES.forEach(c => { const ok = c.items.every(it => state.verdicts[c.id + "/" + it.golden + "/" + it.item]);
    document.getElementById(c.id).classList.toggle("done", ok); });
  const all = CASES.flatMap(c => c.items.map(it => c.id + "/" + it.golden + "/" + it.item));
  all.forEach(k => { if (state.verdicts[k]) done += 1; });
  document.getElementById("count").textContent = done + " / " + all.length + " values judged";
}
document.addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (!b) return;
  state.verdicts[b.dataset.k] = b.dataset.v; save(); });
document.addEventListener("input", e => {
  if (e.target.classList.contains("fix")) state.fixes[e.target.dataset.k] = e.target.value;
  else if (e.target.classList.contains("missing")) state.missing[e.target.dataset.c] = e.target.value;
  else return; save(); });
const q = s => '"' + String(s).replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"').replace(/\\n/g, "\\\\n") + '"';
function toml() {
  const out = ["# Golden review for " + META.prompt_id + " v" + META.version + ".",
    "# verdict: correct | wrong | unsure; correction applies when wrong.",
    "schema_version = 1", "prompt_id = " + q(META.prompt_id), "version = " + META.version, ""];
  CASES.forEach(c => {
    c.items.forEach(it => { const k = c.id + "/" + it.golden + "/" + it.item;
      out.push("[[values]]", "case_id = " + q(c.id), "golden = " + it.golden, "item = " + it.item,
        "expected = " + q(it.text), "verdict = " + q(state.verdicts[k] || ""));
      if (state.fixes[k]) out.push("correction = " + q(state.fixes[k])); out.push(""); });
    if (state.missing[c.id]) out.push("[[missing]]", "case_id = " + q(c.id), "values = " + q(state.missing[c.id]), "");
  });
  return out.join("\\n");
}
document.getElementById("export").onclick = () => {
  const blob = new Blob([toml()], {type: "text/plain"}); const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "golden_review.toml"; a.click(); };
render();
"""


def review_html(evalset: EvalSet, cases: Sequence[EvalCase]) -> str:
    prompt = evalset.prompt
    meta = {"prompt_id": prompt.id, "version": prompt.version}
    data = [{"id": c.id, "items": [{"golden": it.golden, "item": it.item, "text": it.text} for it in expected_items(c)]}
            for c in cases]
    cards = "\n".join(_card(evalset, c) for c in cases)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Golden review</title>
<style>{STYLE}</style></head><body>
<div class="bar"><b>Golden review · {html.escape(prompt.id)} v{prompt.version}</b>
<span id="count"></span><button id="export">Download {REVIEW_FILE}</button></div>
<p>For each expected value, check it against the page: <b>correct</b>, <b>wrong</b> (type the correct
value) or <b>unsure</b>. A table row is written as its cells joined by <code>|</code>; the check passes when
those cells appear together in one extracted row. Add anything the case should also expect. Progress is kept
in this browser. When done, download the file and save it as
<code>{html.escape(str(Path('kaful/prompts', *prompt.id.split('.'), 'evals', REVIEW_FILE)))}</code>.</p>
{cards}
<script>const META = {json.dumps(meta)}; const CASES = {json.dumps(data)};{SCRIPT}</script></body></html>
"""


def write_review(out_dir: Path, evalset: EvalSet, tag: str = "golden") -> Path:
    """Write the review page for the cases carrying `tag`; return its path."""
    cases = [c for c in evalset.cases if tag in c.tags]
    if not cases:
        raise ValueError(f"{evalset.prompt.id}: no cases tagged {tag!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{evalset.prompt.id}__v{evalset.prompt.version}.html"
    path.write_text(review_html(evalset, cases), encoding="utf-8")
    return path
