"""Review page for the understand golden labels (tests/fixtures/understand/golden/<machine>.toml).

One card per row: what the row is, Claude's suggestion and reason, buttons for the answer,
and text fields where a row needs one (parent, page, note). Rows that cite a page show
that page (physical page and printed label) as an image. The page starts from the answers already in the file, and progress is kept in the browser
for that version of the file (so a file changed since cannot be overwritten by older answers);
"Download" writes the same file back with the answers filled in: the original text is
embedded and only the answer fields are replaced, so comments and suggestions are kept.
`status` becomes "reviewed" once every row is answered.

Not used during a run (a tool). The page embeds manual pages, so it goes to outputs/
(gitignored).
"""

import base64
import hashlib
import html
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.base import VALUE_ROLES
from kaful.artifacts.sensor_contract import DUAL_ROLES
from kaful.config import IngestSettings, load_machine_config
from kaful.evals.golden_review import STYLE
from kaful.paths import resolve_in
from kaful.phases.p1_ingest.a_check_documents import check_documents
from kaful.phases.p1_ingest.helpers.page_signals import printed_labels

REVIEW_DPI = 110
KEY_CHARS = 16  # of the file's sha256: browser progress belongs to one version of the file
ROLES = ("observation", "operating_input", "usage_clock", "event")


@dataclass(frozen=True)
class Section:
    title: tuple[str, ...]          # fields joined for the card title
    info: tuple[str, ...]           # fields shown as facts
    choices: dict[str, tuple[str, ...]]  # answer field -> its buttons
    texts: tuple[str, ...]          # answer fields typed in
    required: str                   # the answer that marks a row done
    help: str


SECTIONS: dict[str, Section] = {
    "legacy_fields": Section(
        ("name",), ("legacy_category", "legacy_unit", "legacy_range", "legacy_pages", "legacy_evidence"),
        {"verdict": ("channel", "not_channel", "unsure"), "role": ROLES, "secondary_role": DUAL_ROLES}, ("note",),
        "verdict", "Is this legacy schema field a channel the machine reports as data? If yes, which role (and a "
        "secondary role if it plays both operating_input and observation)?"),
    "candidate_channels": Section(
        ("name",), ("unit", "quote"),
        {"verdict": ("channel", "not_channel", "unsure"), "role": ROLES, "secondary_role": DUAL_ROLES},
        ("note",), "verdict", "Channels the legacy schema missed. Is each one a channel, and with which role?"),
    "components": Section(
        ("id",), ("legacy_category", "legacy_evidence"),
        {"verdict": ("top_level", "child", "not_a_component")}, ("parent", "note"), "verdict",
        "Is each a main (top-level) component, a part of one (child: give its parent id), or not a component?"),
    "fault_modes": Section(
        ("component", "what_degrades"), ("driven_by", "legacy_rate", "legacy_evidence"),
        {"verdict": ("yes", "no")}, ("doc_id", "page", "note"), "verdict",
        "Is this a real fault mode of that component? Give the document and physical page where it is stated. "
        "To merge two rows, answer no and write merge_with:<other row> in the note."),
    "candidate_fault_modes": Section(
        ("component", "what_degrades"), ("driven_by", "reset_by", "quote", "other_pages"),
        {"verdict": ("yes", "no")}, ("note",), "verdict",
        "Fault modes you named in your review, with the page that states each. Is each a real fault mode?"),
    "values": Section(
        ("value",), ("quote",), {"role": VALUE_ROLES}, ("note",), "role",
        "What is each value: a spec, a variant (one of several configurations), a default, a setting_range, "
        "or an example (a screenshot or worked example)?"),
}


def _labels(machine_dir: Path, settings: IngestSettings) -> tuple[dict[str, Path], dict[tuple[str, int], str]]:
    """Each document's path, and each page's printed label, by (doc_id, physical page)."""
    paths, labels = {}, {}
    for record in check_documents(machine_dir, load_machine_config(machine_dir)).records:
        paths[record.doc_id] = resolve_in(machine_dir, record.path)
        with fitz.open(paths[record.doc_id]) as doc:
            for i, label in enumerate(printed_labels(doc, settings)):
                if label.label:
                    labels[(record.doc_id, i + 1)] = label.label
    return paths, labels


def _image(path: Path, page: int) -> str:
    with fitz.open(path) as doc:
        return base64.b64encode(doc[page - 1].get_pixmap(dpi=REVIEW_DPI, alpha=False).tobytes("png")).decode("ascii")


def _card(section: str, index: int, row: dict[str, Any], spec: Section, where: str, image: str | None) -> str:
    key = f"{section}/{index}"
    title = " — ".join(str(row.get(f, "")) for f in spec.title)
    facts = "".join(f"<div class='meta'><b>{f}</b>: {html.escape(str(row[f]))}</div>"
                    for f in spec.info if str(row.get(f, "")).strip())
    suggestion = ", ".join(f"{k.removeprefix('suggested_')}: {row[k]}" for k in sorted(row)
                           if k.startswith("suggested_") and row[k])
    buttons = "".join(
        f"<div class='buttons'><span class='meta'>{field}</span>"
        + "".join(f"<button data-k='{key}' data-f='{field}' data-v='{v}'>{v}</button>" for v in options) + "</div>"
        for field, options in spec.choices.items())
    texts = "".join(f"<input class='fix' data-k='{key}' data-f='{f}' placeholder='{f}'>" for f in spec.texts)
    src = f"<div class='src'><img loading='lazy' alt='{html.escape(where)}' src='data:image/png;base64,{image}'></div>" \
        if image else ""
    return f"""<section class="card" id="{key}"><header><b>{html.escape(title)}</b>
<span class="meta">{html.escape(where)}</span></header><div class="body">{src}<div class="side">{facts}
<p class="desc">Claude suggests {html.escape(suggestion or '—')}: {html.escape(str(row.get('why', '')))}</p>
{buttons}{texts}</div></div></section>"""


SCRIPT = """
const KEY = "kaful-understand-golden-" + META.machine + "-" + META.key;
let stored = {}; try { stored = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { stored = {}; }
const state = JSON.parse(JSON.stringify(META.given));  // the file's answers, then this browser's newer ones
for (const [k, fields] of Object.entries(stored)) state[k] = Object.assign(state[k] || {}, fields);
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
function paint() {
  document.querySelectorAll("button[data-k]").forEach(b => {
    const s = state[b.dataset.k] || {}; b.classList.toggle("on", s[b.dataset.f] === b.dataset.v); });
  document.querySelectorAll("input[data-k]").forEach(i => {
    const s = state[i.dataset.k] || {}; if (document.activeElement !== i) i.value = s[i.dataset.f] || ""; });
  let done = 0, total = 0;
  for (const [section, rows] of Object.entries(META.rows)) rows.forEach((_, i) => {
    total += 1; const s = state[section + "/" + i] || {}; const ok = !!s[META.required[section]];
    if (ok) done += 1; document.getElementById(section + "/" + i).classList.toggle("done", ok); });
  document.getElementById("count").textContent = done + " / " + total + " rows answered";
}
document.addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (!b) return;
  const s = state[b.dataset.k] = state[b.dataset.k] || {};
  s[b.dataset.f] = s[b.dataset.f] === b.dataset.v ? "" : b.dataset.v; save(); paint(); });
document.addEventListener("input", e => { const i = e.target.closest("input[data-k]"); if (!i) return;
  (state[i.dataset.k] = state[i.dataset.k] || {})[i.dataset.f] = i.value; save(); paint(); });
function exported() {
  const answers = {}; for (const [section, rows] of Object.entries(META.rows)) answers[section] = META.answers[section];
  let section = null, index = -1, complete = true;
  for (const [s, rows] of Object.entries(META.rows)) rows.forEach((_, i) => {
    if (!(state[s + "/" + i] || {})[META.required[s]]) complete = false; });
  return META.text.split("\\n").map(line => {
    const head = line.match(/^\\[\\[([a-z_]+)\\]\\]$/);
    if (head) { if (head[1] !== section) { section = head[1]; index = -1; } index += 1; return line; }
    if (/^status = /.test(line)) return 'status = "' + (complete ? "reviewed" : "draft") + '"';
    const field = line.match(/^([a-z_]+) = /);
    if (!field || !section || !(answers[section] || []).includes(field[1])) return line;
    const v = (state[section + "/" + index] || {})[field[1]] || "";
    return field[1] + " = " + JSON.stringify(v);
  }).join("\\n");
}
document.getElementById("export").addEventListener("click", () => {
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([exported()], {type: "text/plain"}));
  a.download = META.machine + ".toml"; a.click(); });
paint();
"""


def review_html(golden_path: Path, machine_dir: Path, settings: IngestSettings) -> str:
    text = golden_path.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    paths, labels = _labels(machine_dir, settings)
    cards, rows = [], {}
    for section, spec in SECTIONS.items():
        rows[section] = data.get(section, [])
        if not rows[section]:
            continue
        cards.append(f"<h2>{section.replace('_', ' ')}</h2><p class='desc'>{html.escape(spec.help)}</p>")
        for i, row in enumerate(rows[section]):
            doc, page = row.get("doc_id"), row.get("page")
            page = int(page) if isinstance(page, str) and page.isdigit() else page
            where, image = "", None
            if isinstance(doc, str) and isinstance(page, int) and doc in paths:
                label = labels.get((doc, page))
                where = f"{doc} p{page}" + (f" (label {label})" if label else "")
                image = _image(paths[doc], page)
            cards.append(_card(section, i, row, spec, where, image))
    answers = {k: [*s.choices, *s.texts] for k, s in SECTIONS.items()}
    given = {f"{k}/{i}": {f: str(row[f]) for f in answers[k] if str(row.get(f, ""))}
             for k, v in rows.items() for i, row in enumerate(v)}
    meta = {"machine": data["machine_id"], "text": text,
            "key": hashlib.sha256(text.encode("utf-8")).hexdigest()[:KEY_CHARS], "rows": {k: [None] * len(v) for k, v in rows.items() if v},
            "required": {k: s.required for k, s in SECTIONS.items()}, "answers": answers,
            "given": {k: v for k, v in given.items() if v}}
    target = Path("tests", "fixtures", "understand", "golden", golden_path.name)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Golden labels · {html.escape(data['machine_id'])}</title><style>{STYLE}h2 {{ margin-top:28px; }}</style></head>
<body><div class="bar"><b>Understand golden labels · {html.escape(data['machine_id'])}</b>
<span id="count"></span><button id="export">Download {html.escape(golden_path.name)}</button></div>
<p>Answer each row with the buttons; click an answer again to clear it. Claude's suggestion is a starting point,
not a default. Progress is kept in this browser. When done, download the file and save it over
<code>{html.escape(str(target))}</code> (comments and suggestions are kept; only your answers change).</p>
{''.join(cards)}
<script>const META = {json.dumps(meta)};{SCRIPT}</script></body></html>
"""


def write_review(out_dir: Path, golden_path: Path, machine_dir: Path, settings: IngestSettings) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{golden_path.stem}.html"
    path.write_text(review_html(golden_path, machine_dir, settings), encoding="utf-8")
    return path
