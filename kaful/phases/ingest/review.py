"""Classifier review page: a stratified page sample to hand-label, as one local HTML file.

For each machine and each predicted class, up to `review_per_class` pages are taken in
`sample_key` order (a hash of document sha + page), so the sample is deterministic and
does not depend on page order. The page shows each page's image, predicted class,
signals, raw-text excerpt and ruled-table markdown, and exports the labels as TOML in
the `PageLabels` format. The HTML embeds page images from the manuals, so it is
written under the gitignored outputs/ directory and never published.
"""

import base64
import html
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import NEEDS_LLM, PAGE_CLASSES, PageText
from kaful.paths import resolve_in
from kaful.phases.ingest.classifier_eval import LabelledPage, PageLabels, labels_toml, sample_key
from kaful.phases.ingest.classify import CLASSIFIER_VERSION
from kaful.phases.ingest.read import ReadDocument

REVIEW_DPI = 100
EXCERPT_CHARS = 1200


@dataclass(frozen=True)
class MachineDocs:
    machine_id: str
    machine_dir: Path
    documents: tuple[ReadDocument, ...]


@dataclass(frozen=True)
class Sampled:
    machine: MachineDocs
    document: ReadDocument
    page: PageText
    entry: LabelledPage


def sample_pages(machines: Sequence[MachineDocs], per_class: int) -> list[Sampled]:
    """Up to `per_class` pages per (machine, predicted class), in deterministic order."""
    picked = []
    for machine in sorted(machines, key=lambda m: m.machine_id):
        for page_class in PAGE_CLASSES:
            stratum = sorted(((d, p) for d in machine.documents for p in d.pages if p.page_class == page_class),
                             key=lambda dp: sample_key(dp[0].record.sha256, dp[1].page))
            chosen = stratum[:per_class]
            for document, page in chosen:
                entry = LabelledPage(machine_id=machine.machine_id, path=document.record.path,
                                     doc_sha256=document.record.sha256, page=page.page, predicted=page_class,
                                     stratum_size=len(stratum), stratum_sampled=len(chosen))
                picked.append(Sampled(machine, document, page, entry))
    return sorted(picked, key=lambda s: s.entry.key)


def sample_labels(sampled: Sequence[Sampled], per_class: int) -> PageLabels:
    return PageLabels(sampled_with=CLASSIFIER_VERSION, per_class=per_class, pages=tuple(s.entry for s in sampled))


def _image_b64(sampled: Sampled) -> str:
    with fitz.open(resolve_in(sampled.machine.machine_dir, sampled.document.record.path)) as doc:
        pixmap = doc[sampled.page.page - 1].get_pixmap(dpi=REVIEW_DPI, alpha=False)
        return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def _card(index: int, sampled: Sampled) -> str:
    page, entry = sampled.page, sampled.entry
    signals = ", ".join(f"{k}={v}" for k, v in page.signals.model_dump().items())
    tables = "".join(f"<pre class='md'>{html.escape(layer.text)}</pre>" for layer in page.layers)
    buttons = "".join(f"<button data-i='{index}' data-c='{c}'>{c}</button>" for c in PAGE_CLASSES)
    route = "LLM" if page.needs_llm else ("raw + table md" if page.layers else "raw only")
    return f"""<section class="card" id="p{index}">
<header><b>{html.escape(entry.machine_id)}</b> · {html.escape(entry.path)} · page {page.page}
(label {html.escape(page.label or '–')}) · predicted <span class="pred">{entry.predicted}</span> → {route}
· stratum {entry.stratum_sampled}/{entry.stratum_size}</header>
<div class="body"><img loading="lazy" alt="page {page.page}" src="data:image/png;base64,{_image_b64(sampled)}">
<div class="side"><div class="buttons">{buttons}</div>
<input class="note" data-i="{index}" placeholder="note (optional)">
<details><summary>signals</summary><code>{html.escape(signals)}</code></details>
<details><summary>raw text excerpt</summary><pre>{html.escape(page.text[:EXCERPT_CHARS])}</pre></details>
{f'<details open><summary>ruled-table markdown</summary>{tables}</details>' if tables else ''}
</div></div></section>"""


STYLE = """
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#ffffff; --line:#e3e1da; --accent:#2f5d8a; --on:#ffffff; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#171716; --fg:#ecebe6; --muted:#a3a29c; --card:#21211f; --line:#3a3936; --accent:#7fb0e0; --on:#10202e; } }
:root[data-theme="dark"] { --bg:#171716; --fg:#ecebe6; --muted:#a3a29c; --card:#21211f; --line:#3a3936;
  --accent:#7fb0e0; --on:#10202e; }
body { background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto;
  max-width:1400px; padding:0 16px 80px; }
.bar { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 0;
  display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:2; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:14px 0; padding:12px; }
.card.done { border-color:var(--accent); }
.body { display:flex; gap:14px; flex-wrap:wrap; } img { max-width:min(640px,100%); border:1px solid var(--line); }
.side { flex:1; min-width:260px; } .buttons { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
button { border:1px solid var(--line); background:transparent; color:var(--fg); border-radius:6px; padding:5px 9px;
  cursor:pointer; } button.on { background:var(--accent); color:var(--on); border-color:var(--accent); }
.note { width:100%; box-sizing:border-box; padding:5px; background:transparent; color:var(--fg);
  border:1px solid var(--line); border-radius:6px; }
pre, code { white-space:pre-wrap; font-size:12px; color:var(--muted); } .pred { color:var(--accent); font-weight:600; }
"""

SCRIPT = """
const KEY = "kaful-classifier-labels-" + META.sampled_with;
let state = {}; try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render(); }
function render() {
  document.querySelectorAll("button[data-c]").forEach(b => {
    const s = state[b.dataset.i] || {}; b.classList.toggle("on", s.label === b.dataset.c); });
  document.querySelectorAll(".note").forEach(n => { const s = state[n.dataset.i] || {};
    if (document.activeElement !== n) n.value = s.note || ""; });
  let done = 0; PAGES.forEach((_, i) => { const ok = !!(state[i] && state[i].label);
    document.getElementById("p" + i).classList.toggle("done", ok); done += ok; });
  document.getElementById("count").textContent = done + " / " + PAGES.length + " labelled";
}
document.addEventListener("click", e => { const b = e.target.closest("button[data-c]"); if (!b) return;
  state[b.dataset.i] = Object.assign({}, state[b.dataset.i], {label: b.dataset.c}); save(); });
document.addEventListener("input", e => { if (!e.target.classList.contains("note")) return;
  state[e.target.dataset.i] = Object.assign({}, state[e.target.dataset.i], {note: e.target.value}); save(); });
const q = s => '"' + String(s).replace(/\\\\/g, "\\\\\\\\").replace(/"/g, '\\\\"') + '"';
function toml() {
  const out = ["# Classifier labels: set `label` to the class each page should get (see the review page).",
    "schema_version = 1", "sampled_with = " + q(META.sampled_with), "per_class = " + META.per_class, ""];
  PAGES.forEach((p, i) => { const s = state[i] || {};
    out.push("[[pages]]", "machine_id = " + q(p.machine_id), "path = " + q(p.path), "doc_sha256 = " + q(p.doc_sha256),
      "page = " + p.page, "predicted = " + q(p.predicted), "stratum_size = " + p.stratum_size,
      "stratum_sampled = " + p.stratum_sampled);
    if (s.label) out.push("label = " + q(s.label)); if (s.note) out.push("note = " + q(s.note)); out.push(""); });
  return out.join("\\n");
}
document.getElementById("export").onclick = () => {
  const blob = new Blob([toml()], {type: "text/plain"}); const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "page_class_labels.toml"; a.click(); };
render();
"""


def review_html(sampled: Sequence[Sampled], labels: PageLabels) -> str:
    meta = {"sampled_with": labels.sampled_with, "per_class": labels.per_class}
    pages = [p.model_dump(mode="json") for p in labels.pages]
    llm = ", ".join(sorted(NEEDS_LLM))
    cards = "\n".join(_card(i, s) for i, s in enumerate(sampled))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Classifier review</title>
<style>{STYLE}</style></head><body>
<div class="bar"><b>Page classifier review · {html.escape(labels.sampled_with)}</b>
<span id="count"></span><button id="export">Download page_class_labels.toml</button></div>
<p>Pick the class each page <i>should</i> get, judged by what it needs: <b>{llm}</b> go to the LLM;
<b>table_ruled</b> means the ruled-table markdown shown is good enough (no LLM); <b>text</b> means raw text
loses nothing that matters (e.g. a decorative photo); <b>blank</b> means no content. Progress is kept in this
browser. When done, download the file and save it as <code>tests/fixtures/ingest/page_class_labels.toml</code>.</p>
{cards}
<script>const META = {json.dumps(meta)}; const PAGES = {json.dumps(pages)};{SCRIPT}</script></body></html>
"""


def write_review(out_dir: Path, sampled: Sequence[Sampled], labels: PageLabels) -> tuple[Path, Path]:
    """Write the review page and an unlabelled labels file; return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / f"{labels.sampled_with}.html"
    page.write_text(review_html(sampled, labels), encoding="utf-8")
    blank = out_dir / "page_class_labels.unlabelled.toml"
    blank.write_text(labels_toml(labels), encoding="utf-8")
    return page, blank
