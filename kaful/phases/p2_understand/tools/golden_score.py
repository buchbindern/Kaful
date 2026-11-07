"""Score an understand run against the golden labels (tests/fixtures/understand/golden/<machine>.toml).

Deterministic, no API calls. Output names are matched to golden rows by normalized name or by
a row's optional `aliases`; an output item that matches no row is listed as `unmatched` and
counts against precision until a person resolves it (add an alias, or a row), because an
unreviewed extra is not evidence of a correct one. `unsure` rows are not scored.

- channels: precision, recall, role and secondary-role accuracy (deployment channels skipped).
- components: precision, recall, parent accuracy.
- fault modes: an output mode matches a golden row of the same component when it cites the
  row's page or names most of the row's `what_degrades` words (words of the component's own
  name do not count, and a trailing "s" is ignored); precision counts matches to `yes` rows
  (and `candidate_fault_modes`), and matches to `no` rows (effects, advice) are listed.
- one golden row is credited once: further output items matching it are `duplicates` (a merge
  failure) and count against precision.
- values: golden values found among stored ranges and periods, their roles, and examples stored.
- grounding: items by source and by best evidence class (from the UnderstandReport).
- disagreements: every place the run and the labels differ, with the run's citation, for a person
  to adjudicate (the labels can be wrong too).

Not used during a run (a tool). The score is saved data: outputs/understand_scores/<run_id>.json.
"""

import hashlib
import re
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
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
UNSCORED = ("unsure", "")  # verdicts that are not scored (no answer yet counts as unsure)


class Disagreement(StrictModel):
    """One place the run and the golden labels differ."""

    kind: str = Field(description="channel | component | fault_mode | value.")
    item: str
    run: str = Field(description="What the run says.")
    golden: str = Field(description="What the labels say.")
    ref: str = Field(default="", description="The run item's citation, or the golden row's page.")


class Matches(StrictModel):
    """Precision and recall of one kind of item, with what did not match."""

    precision: float | None
    recall: float | None
    matched: tuple[str, ...] = Field(description="'output -> golden row', one per golden row, sorted.")
    duplicates: tuple[str, ...] = Field(default=(), description="Further outputs matching a credited row, sorted.")
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
    disagreements: tuple[Disagreement, ...] = Field(description="Sorted by kind, then item.")


def pages_of(value: object) -> list[int]:
    """A golden row's physical pages: none (""), one (53 or "53"), or a range ("60-61")."""
    text = str(value).strip()
    if not text or text == "0":
        return []
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


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


@dataclass
class Sorted:
    """Output items by what they match: credited (one per golden row), duplicates, negatives, nothing."""

    credited: dict[str, Row] = field(default_factory=dict)
    duplicates: dict[str, Row] = field(default_factory=dict)
    wrong: dict[str, Row] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)

    def matches(self, positives: Sequence[Row], key: str) -> Matches:
        hit_rows = {row[key] for row in self.credited.values()}
        total = len(self.credited) + len(self.duplicates) + len(self.wrong) + len(self.unmatched)
        return Matches(precision=_ratio(len(self.credited), total),
                       recall=_ratio(len(hit_rows), len(positives)),
                       matched=tuple(sorted(f"{o} -> {r[key]}" for o, r in self.credited.items())),
                       duplicates=tuple(sorted(f"{o} -> {r[key]}" for o, r in self.duplicates.items())),
                       missed=tuple(sorted(r[key] for r in positives if r[key] not in hit_rows)),
                       wrong=tuple(sorted(f"{o} -> {r[key]} ({r['verdict']})" for o, r in self.wrong.items())),
                       unmatched=tuple(sorted(self.unmatched)))


def _sort(names: Sequence[tuple[str, Row | None]], positive: Sequence[str], key: str) -> Sorted:
    """Sort output names (credited in name order, so the first name of a row gets the credit)."""
    out = Sorted()
    for name, row in sorted(names, key=lambda nr: nr[0]):
        if row is None:
            out.unmatched.append(name)
        elif row["verdict"] in positive:
            target = out.duplicates if any(r[key] == row[key] for r in out.credited.values()) else out.credited
            target[name] = row
        elif row["verdict"] not in UNSCORED:
            out.wrong[name] = row
    return out


def _common(kind: str, s: Sorted, positives: Sequence[Row], key: str, refs: dict[str, str],
            what: dict[str, str], golden_says: Callable[[Row], str],
            negative_says: Callable[[Row], str] = lambda r: str(r["verdict"])) -> list[Disagreement]:
    """Negatives the run produced, outputs with no row, duplicates, and positives the run missed."""
    hit = {r[key] for r in s.credited.values()}
    out = [Disagreement(kind=kind, item=o, run=what[o], golden=negative_says(r), ref=refs[o])
           for o, r in s.wrong.items()]
    out += [Disagreement(kind=kind, item=o, run=what[o], golden="no row", ref=refs[o]) for o in s.unmatched]
    out += [Disagreement(kind=kind, item=o, run=f"another name for {r[key]}", golden="one item", ref=refs[o])
            for o, r in s.duplicates.items()]
    out += [Disagreement(kind=kind, item=r[key], run="not extracted", golden=golden_says(r), ref=_golden_ref(r))
            for r in positives if r[key] not in hit]
    return out


def _golden_ref(row: Row) -> str:
    page = str(row.get("page", ""))
    return f"{row['doc_id']} p{page}" if row.get("doc_id") and pages_of(page) else ""


def score_channels(golden: Row, contract: SensorContract) -> tuple[ChannelScore, list[Disagreement]]:
    rows = golden["legacy_fields"] + golden["candidate_channels"]
    lookup = _lookup(rows, "name")
    by_name = {c.name: c for c in contract.channels if c.source != "deployment"}
    s = _sort([(n, lookup.get(norm(n))) for n in by_name], ("channel",), "name")
    positives = [r for r in rows if r["verdict"] == "channel"]
    refs, what = {n: c.ref for n, c in by_name.items()}, {n: f"a channel ({c.role})" for n, c in by_name.items()}
    found = _common("channel", s, positives, "name", refs, what,
                    lambda r: f"channel ({r['role']})" if r["role"] else "channel")
    graded = [(o, r) for o, r in s.credited.items() if r["role"]]
    errors = [(o, r) for o, r in graded if by_name[o].role != r["role"]]
    found += [Disagreement(kind="channel", item=o, run=f"role {by_name[o].role}", golden=f"role {r['role']}",
                           ref=refs[o]) for o, r in errors]
    dual = [(o, r) for o, r in s.credited.items() if r.get("secondary_role")]
    lacking = [(o, r) for o, r in dual if r["secondary_role"] not in {x.role for x in by_name[o].secondary_roles}]
    found += [Disagreement(kind="channel", item=o, run="no such secondary role",
                           golden=f"secondary {r['secondary_role']}", ref=refs[o]) for o, r in lacking]
    secondary_rows = [r for r in positives if r.get("secondary_role")]
    score = ChannelScore(**s.matches(positives, "name").model_dump(),
                         role_accuracy=_ratio(len(graded) - len(errors), len(graded)),
                         role_errors=tuple(sorted(f"{o}: {by_name[o].role}, golden {r['role']}" for o, r in errors)),
                         secondary_found=_ratio(len(dual) - len(lacking), len(secondary_rows)))
    return score, found


def score_components(golden: Row, model: MachineModel) -> tuple[ComponentScore, list[Disagreement]]:
    rows = golden["components"]
    lookup = _lookup(rows, "id")
    by_id = {c.id: c for c in model.components}
    s = _sort([(i, lookup.get(norm(i))) for i in by_id], ("top_level", "child"), "id")
    positives = [r for r in rows if r["verdict"] in ("top_level", "child")]
    refs, what = {i: c.ref for i, c in by_id.items()}, {i: "a component" for i in by_id}
    found = _common("component", s, positives, "id", refs, what, lambda r: str(r["verdict"]))
    errors = []
    for out, row in sorted(s.credited.items()):
        parent = by_id[out].parent
        got = lookup[norm(parent)]["id"] if parent and norm(parent) in lookup else parent
        expected = row["parent"] if row["verdict"] == "child" else None
        if got != expected:
            errors.append(f"{out}: parent {got}, golden {expected}")
            found.append(Disagreement(kind="component", item=out, run=f"parent {got}", golden=f"parent {expected}",
                                      ref=refs[out]))
    score = ComponentScore(**s.matches(positives, "id").model_dump(), parent_errors=tuple(errors),
                           parent_accuracy=_ratio(len(s.credited) - len(errors), len(s.credited)))
    return score, found


def _words(text: str, settings: UnderstandSettings) -> set[str]:
    return {w.removesuffix("s") for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= settings.golden_min_word_chars}


def _fits(mode: FaultMode, row: Row, own: set[str], settings: UnderstandSettings) -> tuple[bool, bool]:
    """(cites the row's page, names enough of its what_degrades words other than the component's own)."""
    pages = pages_of(row.get("page", ""))
    cites = any(e.doc_id == row.get("doc_id") and e.page in pages for e in mode.evidence)
    want = _words(row["what_degrades"], settings) - own
    have = _words(f"{mode.name} {mode.mechanism} {mode.effect}", settings)
    return cites, bool(want) and len(want & have) / len(want) >= settings.golden_text_overlap


def _best(mode: FaultMode, component: str, rows: Sequence[Row], own: set[str],
          settings: UnderstandSettings) -> Row | None:
    """The row this mode fits best: a cited page first, then the words; ties go to the first row."""
    fits = [(_fits(mode, r, own, settings), i) for i, r in enumerate(rows) if r["component"] == component]
    fits = [(f, i) for f, i in fits if any(f)]
    return rows[min(fits, key=lambda x: (not x[0][0], not x[0][1], x[1]))[1]] if fits else None


def _label(row: Row) -> str:
    return f"{row['component']}: {row['what_degrades']}"


def score_fault_modes(golden: Row, model: MachineModel,
                      settings: UnderstandSettings) -> tuple[Matches, list[Disagreement]]:
    components = _lookup(golden["components"], "id")
    yes = [{**r, "label": _label(r)} for r in golden["fault_modes"] + golden.get("candidate_fault_modes", [])
           if r["verdict"] == "yes"]
    no = [{**r, "label": _label(r)} for r in golden["fault_modes"] if r["verdict"] == "no"]
    names: list[tuple[str, Row | None]] = []
    for mode in model.fault_modes:
        owner = components.get(norm(mode.component))
        component = owner["id"] if owner else mode.component
        own = set().union(*(_words(n, settings) for n in (component, *(owner or {}).get("aliases", ()))))
        hit = _best(mode, component, yes, own, settings)
        names.append((mode.id, hit if hit is not None else _best(mode, component, no, own, settings)))
    s = _sort(names, ("yes",), "label")
    by_id = {m.id: m for m in model.fault_modes}
    refs, what = {i: m.ref for i, m in by_id.items()}, {i: "a fault mode" for i in by_id}
    found = _common("fault_mode", s, yes, "label", refs, what, lambda r: "yes",
                    lambda r: f"{r['verdict']} ({r['label']})")
    return s.matches(yes, "label"), found


def _stored(contract: SensorContract) -> list[tuple[str, Evidence]]:
    return [(c.ref, e) for c in contract.channels for e in (c.range_evidence, c.sample_period_evidence)
            if e is not None]


def score_values(golden: Row, contract: SensorContract) -> tuple[ValueScore, list[Disagreement]]:
    found, examples, disagree = [], [], []
    for row in golden.get("values", []):
        quote = norm(row["quote"])
        for ref, e in _stored(contract):
            if e.doc_id == row["doc_id"] and e.page == row["page"] and (quote in norm(e.quote) or norm(e.quote) in quote):
                found.append(f"{row['value']}: {row['role']} -> {e.value_role}")
                if row["role"] == "example":
                    examples.append(f"{row['value']} (p{row['page']})")
                    disagree.append(Disagreement(kind="value", item=row["value"], run=f"stored as a range ({e.value_role})",
                                                 golden="example", ref=ref))
                elif e.value_role != row["role"]:
                    disagree.append(Disagreement(kind="value", item=row["value"], run=f"role {e.value_role}",
                                                 golden=f"role {row['role']}", ref=ref))
    score = ValueScore(golden=len(golden.get("values", [])), found=tuple(sorted(found)),
                       examples_stored=tuple(sorted(examples)))
    return score, disagree


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
    channels, a = score_channels(golden, contract)
    components, b = score_components(golden, model)
    fault_modes, c = score_fault_modes(golden, model, settings)
    values, d = score_values(golden, contract)
    disagreements = tuple(sorted(a + b + c + d, key=lambda x: (x.kind, x.item, x.run)))
    return GoldenScore(machine_id=model.machine_id, run_id=model.run_id, golden_sha256=hashlib.sha256(text).hexdigest(),
                       channels=channels, components=components, fault_modes=fault_modes, values=values,
                       grounding=grounding(report), disagreements=disagreements)
