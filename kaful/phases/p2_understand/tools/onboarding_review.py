"""Onboarding review of an understand run, for an engineer (M7 exit item, Natalie 2026-11-08).

- `kaful understand review <machine> [--run RUN_ID]` writes outputs/understand_reviews/<run_id>.html, one
  read-only page: failed checks and dropped items first, then every item from lowest confidence up
  (downgraded first), then the component tree, the channels and the fault modes. Every quote links to
  its page (document and physical page, with the printed label). It also writes <run_id>.toml, the
  review record: one entry per item with a blank `action`. An existing record is kept.
- The engineer sets `action` per entry: `confirm`, `correct` (with a `declare` table holding the
  corrected item) or `add` (a new entry with `declare`), plus `reviewer` and `reviewed_on`.
- `kaful understand review-apply <machine> --run RUN_ID` turns component and channel corrections and
  additions into machine.toml [[components]] / [[instrumentation]] declarations (source `deployment`;
  a declaration replaces the extraction of the same id on the next run). Confirmations and fault-mode
  entries are recorded only: confidence is computed by code (hard rule 15), and fault modes have no
  declaration format yet (M9).
- When the run has a Scope (M8), the page adds its decisions (p3_scope/tools/review) and the record a `scope`
  entry per component worth a look; a `scope` correction becomes a machine.toml [[scope]] override and a
  `life` addition a [[life]] declaration.

Not used during a run (a tool). No API calls.
"""

import html
import json
import re
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import Provenanced, StrictModel
from kaful.artifacts.document_index import DocumentIndex
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import ItemCheck, ItemKind, UnderstandReport
from kaful.artifacts.scope import Scope
from kaful.config import (
    MACHINE_CONFIG_FILE,
    DeclaredChannel,
    DeclaredComponent,
    DeclaredLife,
    DeclaredParam,
    DeclaredScope,
    MachineConfig,
    load_machine_config,
)
from kaful.evals.golden_review import STYLE
from kaful.phases.p2_understand.d_build_artifacts import UNCONFIRMED_PART
from kaful.artifacts.param_set import ParamSet
from kaful.phases.p3_scope.tools import review as scope_review
from kaful.phases.p4_parameterize.tools import review as param_review
from kaful.phases.p4_parameterize.tools.report import ParamReport
from kaful.runs import write_text_atomic

REVIEWS_DIR = ("outputs", "understand_reviews")
ReviewAction = Literal["", "confirm", "correct", "add"]
ReviewKind = ItemKind | Literal["scope", "life", "param"]
DECLARED: dict[str, tuple[str, str, type[StrictModel]]] = {  # kind -> machine.toml table, its key, model
    "component": ("components", "id", DeclaredComponent),
    "channel": ("instrumentation", "name", DeclaredChannel),
    "scope": ("scope", "component", DeclaredScope),
    "life": ("life", "component", DeclaredLife),
    "param": ("param", "param", DeclaredParam),
}


class ReviewEntry(StrictModel):
    kind: ReviewKind
    id: str = Field(min_length=1)
    action: ReviewAction = ""
    note: str = ""
    declare: dict[str, Any] | None = Field(default=None, description="The corrected or added item (correct/add).")

    @model_validator(mode="after")
    def _declare(self) -> "ReviewEntry":
        if self.action in ("correct", "add") and self.kind in DECLARED:
            if self.declare is None:
                raise ValueError(f"{self.kind} {self.id}: '{self.action}' needs a declare table")
            table, key, model = DECLARED[self.kind]
            model.model_validate(self.declare)
            if self.declare.get(key) != self.id:
                raise ValueError(f"{self.kind} {self.id}: declare.{key} must be {self.id!r}")
        return self


class ReviewRecord(StrictModel):
    schema_version: int = 1
    machine_id: str
    run_id: str
    reviewer: str = ""
    reviewed_on: str = ""
    items: tuple[ReviewEntry, ...] = ()


def _labels(index: DocumentIndex) -> tuple[dict[str, str], dict[tuple[str, int], str]]:
    return ({d.doc_id: d.path for d in index.documents},
            {(p.doc_id, p.page): p.label for p in index.pages if p.label})


def _links(item: Provenanced, paths: dict[str, str], labels: dict[tuple[str, int], str]) -> str:
    out = []
    for e in item.evidence:
        label = labels.get((e.doc_id, e.page))
        where = f"{e.doc_id} p{e.page}" + (f" (label {label})" if label and label != str(e.page) else "")
        href = f"../../{paths[e.doc_id]}#page={e.page}" if e.doc_id in paths else ""
        quote = html.escape(e.quote[:160])
        out.append(f'<a href="{html.escape(href)}">{html.escape(where)}</a> <span class="meta">{e.verification}: '
                   f'“{quote}”</span>')
    return "<br>".join(out) or '<span class="meta">no quote (' + html.escape(item.source) + ")</span>"


def review_order(report: UnderstandReport) -> list[ItemCheck]:
    """Downgraded items first, then by rising confidence (ties by kind and id)."""
    return sorted(report.items, key=lambda i: (i.downgraded_from is None, i.confidence, i.kind, i.id))


def _rows(cells: Sequence[Sequence[str]]) -> str:
    return "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return ("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>" + _rows(rows)
            + "</table>")


def _tree(model: MachineModel, link: dict[str, str]) -> str:
    children: dict[str | None, list[str]] = {}
    for c in model.components:
        children.setdefault(c.parent, []).append(c.id)
    names = {c.id: c for c in model.components}

    def branch(parent: str | None) -> str:
        kids = sorted(children.get(parent, []))
        return "<ul>" + "".join(f"<li><b>{html.escape(k)}</b> {html.escape(names[k].name)} "
                                f'<span class="meta">{names[k].source}, confidence {names[k].confidence:.2f}</span>'
                                f"<br>{link[k]}{branch(k)}</li>" for k in kids) + "</ul>" if kids else ""
    return branch(None)


def _parts(item: ItemCheck, report: UnderstandReport) -> str:
    """'found in k of n parts' for an item of a run whose context came in parts (M7c), else blank."""
    prompt = "understand.fault_modes" if item.kind == "fault_mode" or item.found_by == "fault_modes" \
        else "understand.sensor_channels" if item.kind == "channel" else "understand.components"
    total = len(report.contexts.get(prompt, ()))
    return f"{item.parts_found} of {total}" if item.parts_found is not None and total > 1 else ""


def review_html(model: MachineModel, contract: SensorContract, report: UnderstandReport, index: DocumentIndex,
                scope: Scope | None = None, run_id: str | None = None,
                params: tuple[ParamSet, ParamReport] | None = None) -> str:
    paths, labels = _labels(index)
    items: dict[tuple[str, str], Provenanced] = {("component", c.id): c for c in model.components}
    items |= {("channel", c.name): c for c in contract.channels} | {("fault_mode", f.id): f for f in model.fault_modes}
    items |= {("flow_path", f.id): f for f in model.flow_paths}
    link = {k: _links(v, paths, labels) for k, v in items.items()}
    unconfirmed = [d for d in report.dropped if UNCONFIRMED_PART in d]
    failed = [*report.unit_problems, *(d for d in report.dropped if UNCONFIRMED_PART not in d)]
    first = [(html.escape(i.kind) + (" (carrier, from fault modes)" if i.found_by == "fault_modes" else ""),
              f"<b>{html.escape(i.id)}</b>", i.source + (f" (was {i.downgraded_from})"
              if i.downgraded_from else ""), f"{i.confidence:.2f}", f"{i.agreement:.2f}", _parts(i, report),
              link.get((i.kind, i.id), "")) for i in review_order(report)]
    channels = [(f"<b>{html.escape(c.name)}</b>", c.role, c.dtype, html.escape(c.unit or c.unit_status or ""),
                 html.escape(c.description), link[("channel", c.name)]) for c in contract.channels]
    modes = [(f"<b>{html.escape(f.id)}</b>", html.escape(f.component), html.escape(f.mechanism),
              html.escape(f.effect), link[("fault_mode", f.id)]) for f in model.fault_modes]
    body = (f"<h1>Onboarding review · {html.escape(model.machine_id)}</h1>"
            f'<p class="meta">run {html.escape(run_id or model.run_id)}. Record each decision in the review file '
            f"outputs/understand_reviews/{html.escape(run_id or model.run_id)}.toml (confirm, correct or add), then run "
            "<code>kaful understand review-apply</code>.</p>"
            f"<h2>Failed checks and dropped items ({len(failed)})</h2><ul>"
            + "".join(f'<li class="warn">{html.escape(f)}</li>' for f in failed) + "</ul>"
            + f"<h2>Parts only reference documents name ({len(unconfirmed)})</h2>"
            + '<p class="meta">Fault-mode carriers no machine manual confirms. If the machine has one, add it to '
            "the review file; review-apply declares it in machine.toml [[components]].</p><ul>"
            + "".join(f'<li class="warn">{html.escape(u)}</li>' for u in unconfirmed) + "</ul>"
            + "<h2>Review first: downgraded, then lowest confidence</h2>"
            + _table(("kind", "id", "source", "confidence", "agreement", "parts", "evidence"), first)
            + "<h2>Component tree</h2>" + _tree(model, {k[1]: v for k, v in link.items() if k[0] == "component"})
            + f"<h2>Channels (usage clock: {html.escape(contract.usage_clock or 'none')})</h2>"
            + _table(("name", "role", "dtype", "unit", "description", "evidence"), channels)
            + "<h2>Fault modes</h2>" + _table(("id", "component", "mechanism", "effect", "evidence"), modes)
            + (scope_review.scope_html(scope) if scope is not None else "")
            + (param_review.param_html(*params) if params is not None else ""))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>Onboarding review</title><style>{STYLE}'
            "table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); "
            "padding:6px; text-align:left; vertical-align:top; } a { color:var(--accent); } "
            "ul ul { border-left:1px solid var(--line); }</style></head><body>" + body + "</body></html>")


def _value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def _fields(d: dict[str, Any]) -> str:
    return "".join(f"{k} = {_value(v)}\n" for k, v in d.items() if v is not None)


def record_toml(record: ReviewRecord) -> str:
    """The review record as TOML (scalars, lists and the flat `declare` tables only)."""
    head = {k: v for k, v in record.model_dump().items() if k != "items"}
    out = ["# Onboarding review record. action: confirm | correct | add (blank: not reviewed). correct and add of a\n"
           "# component or channel need a [items.declare] table with the item as machine.toml declares it.\n",
           _fields(head)]
    for e in record.items:
        out.append("\n[[items]]\n" + _fields({k: v for k, v in e.model_dump().items() if k != "declare"}))
        if e.declare is not None:
            out.append("[items.declare]\n" + _fields(e.declare))
    return "".join(out)


def write_review(machine_dir: Path, model: MachineModel, contract: SensorContract, report: UnderstandReport,
                 index: DocumentIndex, scope: Scope | None = None, run_id: str | None = None,
                 params: tuple[ParamSet, ParamReport] | None = None) -> tuple[Path, Path]:
    """The page and record, named for `run_id` (the reviewed run; default the understand run)."""
    run = run_id or model.run_id
    out = machine_dir.joinpath(*REVIEWS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    page, record = out / f"{run}.html", out / f"{run}.toml"
    write_text_atomic(page, review_html(model, contract, report, index, scope, run, params))
    if not record.exists():
        entries = [ReviewEntry(kind=i.kind, id=i.id) for i in review_order(report)]
        entries += [ReviewEntry(kind="scope", id=d.component) for d in scope_review.reviewed(scope)] if scope else []
        entries += [ReviewEntry(kind="param", id=pid) for pid in param_review.reviewed(*params)] if params else []
        write_text_atomic(record, record_toml(ReviewRecord(machine_id=model.machine_id, run_id=run,
                                                           items=tuple(entries))))
    return page, record


def insert_declaration(text: str, table: str, key: str, item_id: str, block: str) -> str:
    """machine.toml text with `block` (a [[table]] entry) placed so the table stays sorted by `key`."""
    found = [(m.start(), m.group(1)) for m in re.finditer(rf'^\[\[{table}\]\]\n{key} = "([^"]+)"', text, re.M)]
    if any(i == item_id for _, i in found):
        raise ValueError(f"{table} {item_id}: already declared in {MACHINE_CONFIG_FILE}; edit it there")
    later = [pos for pos, i in found if i > item_id]
    if later:
        return text[:later[0]] + block + "\n" + text[later[0]:]
    if found:
        nxt = re.compile(r"^\[", re.M).search(text, found[-1][0] + 1)
        if nxt is not None:
            return text[:nxt.start()] + block + "\n" + text[nxt.start():]
    return text.rstrip("\n") + "\n\n" + block


def apply_review(machine_dir: Path, record_path: Path) -> list[str]:
    """Write the record's component and channel corrections and additions to machine.toml; what was done, by line."""
    with record_path.open("rb") as handle:
        record = ReviewRecord.model_validate(tomllib.load(handle))
    config_path = machine_dir / MACHINE_CONFIG_FILE
    text, done = config_path.read_text(encoding="utf-8"), []
    stamp = f"{record.reviewer or 'unnamed reviewer'}, {record.reviewed_on or date.today().isoformat()}"
    for e in sorted(record.items, key=lambda e: (e.kind, e.id)):
        if not e.action:
            continue
        if e.action == "confirm" or e.kind not in DECLARED or e.declare is None:
            done.append(f"{e.kind} {e.id}: {e.action} recorded"
                        + ("" if e.action == "confirm" else " (no declaration format for this kind)"))
            continue
        table, key, _ = DECLARED[e.kind]
        declare = {key: e.declare[key], **e.declare, **({"note": e.declare.get("note") or f"onboarding review ({stamp})"}
                                   if e.kind in ("component", "scope", "life", "param") else {})}
        block = (f"# onboarding review of run {record.run_id} ({stamp}): {e.action}"
                 + (f"; {e.note}" if e.note else "") + f"\n[[{table}]]\n" + _fields(declare))
        text = insert_declaration(text, table, key, e.id, block)
        done.append(f"{e.kind} {e.id}: {e.action} -> machine.toml [[{table}]]")
    MachineConfig.model_validate(tomllib.loads(text))  # the file must still load before it is written
    write_text_atomic(config_path, text)
    load_machine_config(machine_dir)
    return done
