"""Score an understand run against the golden labels (tests/fixtures/understand/golden/<machine>.toml).

Deterministic, no API calls. Output names are matched to golden rows by normalized name or by
a row's optional `aliases`; an output item that matches no row is listed as `unmatched` and
counts against precision until a person resolves it (add an alias, or a row), because an
unreviewed extra is not evidence of a correct one. `unsure` rows are not scored.

- channels: precision, recall, role and secondary-role accuracy (deployment channels skipped).
- components: precision, recall, parent accuracy.
- fault modes: an output mode matches a golden row of the same component when it cites the
  row's page or names most of the row's `what_degrades` words; precision counts matches to
  `yes` rows (and `candidate_fault_modes`), and matches to `no` rows (effects, advice) are listed.
- values: golden values found among stored ranges and periods, their roles, and examples stored.
- grounding: items by source and by best evidence class (from the UnderstandReport).

Not used during a run (a tool). The score is saved data: outputs/understand_scores/<run_id>.json.
"""

import hashlib
import re
import tomllib
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from kaful.artifacts.base import Evidence, StrictModel
from kaful.artifacts.machine_model import FaultMode, MachineModel
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.config import UnderstandSettings
from kaful.phases.p2_understand.c_verify_quotes import EVIDENCE_RANK

Row = dict[str, Any]


class Matches(StrictModel):
    """Precision and recall of one kind of item, with what did not match."""

    precision: float | None
    recall: float | None
    matched: tuple[str, ...] = Field(description="'output -> golden row', sorted.")
    missed: tuple[str, ...] = Field(description="Golden positives no output item matched, sorted.")
    wrong: tuple[str, ...] = Field(description="Output items that match a golden negative, sorted.")
    unmatched: tuple[str, ...] = Field(description="Output items that match no golden row (resolve by hand), sorted.")


class ChannelScore(Matches):
    role_accuracy: float | None
    role_errors: tuple[str, ...]
    secondary_found: float | None = Field(description="Share of golden secondary roles the run gave.")


class ComponentScore(Matches):
    parent_accuracy: float | None
    parent_errors: tuple[str, ...]


class ValueScore(StrictModel):
    golden: int
    found: tuple[str, ...] = Field(description="'value: golden role -> stored role', sorted.")
    examples_stored: tuple[str, ...] = Field(description="Golden examples stored as a range or period (must be none).")


class Grounding(StrictModel):
    by_source: dict[str, dict[str, int]] = Field(description="Item kind -> source -> count.")
    by_evidence: dict[str, dict[str, int]] = Field(description="Item kind -> best evidence class -> count.")


class GoldenScore(StrictModel):
    machine_id: str
    run_id: str
    golden_sha256: str
    channels: ChannelScore
    components: ComponentScore
    fault_modes: Matches
    values: ValueScore
    grounding: Grounding


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _lookup(rows: Iterable[Row], key: str) -> dict[str, Row]:
    out: dict[str, Row] = {}
    for row in rows:
        for name in (row[key], *row.get("aliases", ())):
            out[norm(name)] = row
    return out


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 6) if d else None


def _matches(found: dict[str, Row], positives: Sequence[Row], key: str, wrong: list[str],
             unmatched: list[str]) -> Matches:
    hits = sorted(f"{out} -> {row[key]}" for out, row in found.items())
    hit_rows = {row[key] for row in found.values()}
    missed = sorted(row[key] for row in positives if row[key] not in hit_rows)
    total = len(found) + len(wrong) + len(unmatched)
    return Matches(precision=_ratio(len(found), total), recall=_ratio(len(positives) - len(missed), len(positives)),
                   matched=tuple(hits), missed=tuple(missed), wrong=tuple(sorted(wrong)),
                   unmatched=tuple(sorted(unmatched)))


def _sort(names: Sequence[tuple[str, Row | None]], verdict: str, key: str) -> tuple[dict[str, Row], list[str], list[str]]:
    """Output names -> the positive row each matches; output names matching a negative; output names matching none."""
    found: dict[str, Row] = {}
    wrong: list[str] = []
    unmatched: list[str] = []
    for out, row in names:
        if row is None:
            unmatched.append(out)
        elif row["verdict"] in verdict.split("|"):
            found[out] = row
        elif row["verdict"] != "unsure":
            wrong.append(f"{out} -> {row[key]} ({row['verdict']})")
    return found, wrong, unmatched


def score_channels(golden: Row, contract: SensorContract) -> ChannelScore:
    rows = golden["legacy_fields"] + golden["candidate_channels"]
    lookup = _lookup(rows, "name")
    outputs = [c for c in contract.channels if c.source != "deployment"]
    found, wrong, unmatched = _sort([(c.name, lookup.get(norm(c.name))) for c in outputs], "channel", "name")
    base = _matches(found, [r for r in rows if r["verdict"] == "channel"], "name", wrong, unmatched)
    by_name = {c.name: c for c in outputs}
    errors = sorted(f"{out}: {by_name[out].role}, golden {row['role']}" for out, row in found.items()
                    if row["role"] and by_name[out].role != row["role"])
    graded = [out for out, row in found.items() if row["role"]]
    dual = [(out, row) for out, row in found.items() if row.get("secondary_role")]
    given = [out for out, row in dual if row["secondary_role"] in {r.role for r in by_name[out].secondary_roles}]
    secondary_rows = [r for r in rows if r.get("secondary_role") and r["verdict"] == "channel"]
    return ChannelScore(**base.model_dump(), role_accuracy=_ratio(len(graded) - len(errors), len(graded)),
                        role_errors=tuple(errors), secondary_found=_ratio(len(given), len(secondary_rows)))


def score_components(golden: Row, model: MachineModel) -> ComponentScore:
    rows = golden["components"]
    lookup = _lookup(rows, "id")
    found, wrong, unmatched = _sort([(c.id, lookup.get(norm(c.id))) for c in model.components],
                                    "top_level|child", "id")
    base = _matches(found, [r for r in rows if r["verdict"] in ("top_level", "child")], "id", wrong, unmatched)
    parents = {c.id: c.parent for c in model.components}
    errors = []
    for out, row in sorted(found.items()):
        parent = parents[out]
        got = lookup[norm(parent)]["id"] if parent and norm(parent) in lookup else parent
        expected = row["parent"] if row["verdict"] == "child" else None
        if got != expected:
            errors.append(f"{out}: parent {got}, golden {expected}")
    return ComponentScore(**base.model_dump(), parent_accuracy=_ratio(len(found) - len(errors), len(found)),
                          parent_errors=tuple(errors))


def _words(text: str, settings: UnderstandSettings) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= settings.golden_min_word_chars}


def _fits(mode: FaultMode, row: Row, settings: UnderstandSettings) -> tuple[bool, bool]:
    """(cites the row's page, names enough of its what_degrades words)."""
    page = str(row.get("page", ""))
    cites = bool(page) and any(e.doc_id == row.get("doc_id") and str(e.page) == page for e in mode.evidence)
    want = _words(row["what_degrades"], settings)
    have = _words(f"{mode.name} {mode.mechanism} {mode.effect}", settings)
    return cites, bool(want) and len(want & have) / len(want) >= settings.golden_text_overlap


def _best(mode: FaultMode, component: str, rows: Sequence[Row], settings: UnderstandSettings) -> Row | None:
    """The row this mode fits best: a cited page first, then the words; ties go to the first row."""
    fits = [(_fits(mode, r, settings), i) for i, r in enumerate(rows) if r["component"] == component]
    fits = [(f, i) for f, i in fits if any(f)]
    return rows[min(fits, key=lambda x: (not x[0][0], not x[0][1], x[1]))[1]] if fits else None


def _label(row: Row) -> str:
    return f"{row['component']}: {row['what_degrades']}"


def score_fault_modes(golden: Row, model: MachineModel, settings: UnderstandSettings) -> Matches:
    components = _lookup(golden["components"], "id")
    yes = [r for r in golden["fault_modes"] + golden.get("candidate_fault_modes", []) if r["verdict"] == "yes"]
    no = [r for r in golden["fault_modes"] if r["verdict"] == "no"]
    found: dict[str, Row] = {}
    wrong: list[str] = []
    unmatched: list[str] = []
    for mode in model.fault_modes:
        owner = components.get(norm(mode.component))
        component = owner["id"] if owner else mode.component
        hit, miss = _best(mode, component, yes, settings), _best(mode, component, no, settings)
        if hit is not None:
            found[mode.id] = {**hit, "label": _label(hit)}
        elif miss is not None:
            wrong.append(f"{mode.id} -> {_label(miss)} (no)")
        else:
            unmatched.append(mode.id)
    return _matches(found, [{**r, "label": _label(r)} for r in yes], "label", wrong, unmatched)


def _stored(contract: SensorContract) -> list[Evidence]:
    return [e for c in contract.channels for e in (c.range_evidence, c.sample_period_evidence) if e is not None]


def score_values(golden: Row, contract: SensorContract) -> ValueScore:
    stored = _stored(contract)
    found, examples = [], []
    for row in golden.get("values", []):
        quote = norm(row["quote"])
        for e in stored:
            if e.doc_id == row["doc_id"] and e.page == row["page"] and (quote in norm(e.quote) or norm(e.quote) in quote):
                found.append(f"{row['value']}: {row['role']} -> {e.value_role}")
                if row["role"] == "example":
                    examples.append(f"{row['value']} (p{row['page']})")
    return ValueScore(golden=len(golden.get("values", [])), found=tuple(sorted(found)),
                      examples_stored=tuple(sorted(examples)))


def grounding(report: UnderstandReport) -> Grounding:
    sources: dict[str, Counter[str]] = {}
    classes: dict[str, Counter[str]] = {}
    for item in report.items:
        sources.setdefault(item.kind, Counter())[item.source] += 1
        best = min(item.evidence, key=EVIDENCE_RANK.index) if item.evidence else "none"
        classes.setdefault(item.kind, Counter())[best] += 1
    return Grounding(by_source={k: dict(sorted(v.items())) for k, v in sorted(sources.items())},
                     by_evidence={k: dict(sorted(v.items())) for k, v in sorted(classes.items())})


def score(golden_path: Path, model: MachineModel, contract: SensorContract, report: UnderstandReport,
          settings: UnderstandSettings) -> GoldenScore:
    text = golden_path.read_bytes()
    golden = tomllib.loads(text.decode("utf-8"))
    if golden["machine_id"] != model.machine_id:
        raise ValueError(f"golden labels are for {golden['machine_id']!r}, the run for {model.machine_id!r}")
    return GoldenScore(machine_id=model.machine_id, run_id=model.run_id, golden_sha256=hashlib.sha256(text).hexdigest(),
                       channels=score_channels(golden, contract), components=score_components(golden, model),
                       fault_modes=score_fault_modes(golden, model, settings), values=score_values(golden, contract),
                       grounding=grounding(report))
