"""Classifier evaluation against hand labels.

Labels live in tests/fixtures/ingest/page_class_labels.toml (page references only, no manual
text). Each entry names a page by machine, document path, document sha256 and page
number, the class the classifier predicted when the page was sampled, and the class a
person gave it (`label`, empty until labelled).

The gate metric is needs-LLM recall on the labelled pages: a needs-LLM page the
classifier sends to raw text only loses its table or figure content. Precision is
reported because it drives cost. The sample is stratified by predicted class, so a
prevalence-weighted recall (each page weighted by its stratum's size / sample size)
is reported alongside, ungated.
"""

import hashlib
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from kaful.artifacts.base import Fraction, Identifier, NonEmptyStr, Sha256, StrictModel, sorted_unique
from kaful.artifacts.document_index import NEEDS_LLM, PageClass
from kaful.paths import RelPath

LabelKey = tuple[str, str, int]


class LabelledPage(StrictModel):
    machine_id: Identifier
    path: RelPath
    doc_sha256: Sha256
    page: int = Field(ge=1)
    predicted: PageClass = Field(description="Classifier output when the page was sampled.")
    stratum_size: int = Field(ge=1, description="Pages with this predicted class in this machine.")
    stratum_sampled: int = Field(ge=1)
    label: PageClass | None = None
    note: str = ""

    @property
    def key(self) -> LabelKey:
        return self.machine_id, self.path, self.page


class PageLabels(StrictModel):
    schema_version: Literal[1] = 1
    sampled_with: NonEmptyStr = Field(description="Classifier version the sample was drawn with.")
    per_class: int = Field(ge=1)
    pages: Annotated[tuple[LabelledPage, ...], sorted_unique(lambda p: p.key, "pages")]


def sample_key(doc_sha256: str, page: int) -> str:
    """Deterministic sampling order within a stratum."""
    return hashlib.sha256(f"{doc_sha256}:{page}".encode()).hexdigest()


def read_labels(path: Path) -> PageLabels:
    with path.open("rb") as handle:
        return PageLabels.model_validate(tomllib.load(handle))


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def labels_toml(labels: PageLabels) -> str:
    """TOML text for a labels file (the review page writes the same format)."""
    lines = ["# Classifier labels: set `label` to the class each page should get (see the review page).",
             "schema_version = 1", f"sampled_with = {_toml_str(labels.sampled_with)}",
             f"per_class = {labels.per_class}", ""]
    for p in labels.pages:
        lines += ["[[pages]]", f"machine_id = {_toml_str(p.machine_id)}", f"path = {_toml_str(p.path)}",
                  f"doc_sha256 = {_toml_str(p.doc_sha256)}", f"page = {p.page}",
                  f"predicted = {_toml_str(p.predicted)}", f"stratum_size = {p.stratum_size}",
                  f"stratum_sampled = {p.stratum_sampled}"]
        lines += [f"label = {_toml_str(p.label)}"] if p.label else []
        lines += [f"note = {_toml_str(p.note)}"] if p.note else []
        lines.append("")
    return "\n".join(lines)


# ── Metrics ──────────────────────────────────────────────────────────────────

class ConfusionCell(StrictModel):
    label: PageClass
    predicted: PageClass
    count: int = Field(ge=1)


class MissedPage(StrictModel):
    machine_id: Identifier
    path: RelPath
    page: int
    label: PageClass
    predicted: PageClass


class ClassifierEvalResult(StrictModel):
    schema_version: Literal[1] = 1
    kind: Literal["classifier_eval"] = "classifier_eval"
    classifier_version: NonEmptyStr
    thresholds: dict[str, float]
    n_labelled: int = Field(ge=0)
    n_unlabelled: int = Field(ge=0)
    confusion: Annotated[tuple[ConfusionCell, ...], sorted_unique(lambda c: (c.label, c.predicted), "cells")]
    needs_llm_tp: int
    needs_llm_fn: int
    needs_llm_fp: int
    needs_llm_tn: int
    needs_llm_recall: Fraction | None
    needs_llm_precision: Fraction | None
    weighted_recall: Fraction | None
    min_recall: Fraction
    passed: bool
    missed: Annotated[tuple[MissedPage, ...], sorted_unique(lambda m: (m.machine_id, m.path, m.page), "missed")]
    git_commit: NonEmptyStr
    created_at: datetime


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def evaluate_labels(labels: PageLabels, predictions: dict[LabelKey, PageClass], *, classifier_version: str,
                    thresholds: dict[str, float], min_recall: float, git_commit: str,
                    created_at: datetime) -> ClassifierEvalResult:
    """Score current predictions against the labelled pages (unlabelled ones are counted only)."""
    done = [p for p in labels.pages if p.label is not None]
    cells: Counter[tuple[PageClass, PageClass]] = Counter()
    counts = Counter[str]()
    weighted_tp = weighted_pos = 0.0
    missed = []
    for page in done:
        assert page.label is not None
        predicted = predictions[page.key]
        cells[(page.label, predicted)] += 1
        truth, guess = page.label in NEEDS_LLM, predicted in NEEDS_LLM
        counts[("t" if truth == guess else "f") + ("p" if guess else "n")] += 1
        if truth:
            weight = page.stratum_size / page.stratum_sampled
            weighted_pos += weight
            weighted_tp += weight if guess else 0.0
            if not guess:
                missed.append(MissedPage(machine_id=page.machine_id, path=page.path, page=page.page,
                                         label=page.label, predicted=predicted))
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    return ClassifierEvalResult(
        classifier_version=classifier_version, thresholds=dict(sorted(thresholds.items())), n_labelled=len(done),
        n_unlabelled=len(labels.pages) - len(done),
        confusion=tuple(ConfusionCell(label=t, predicted=g, count=n) for (t, g), n in sorted(cells.items())),
        needs_llm_tp=counts["tp"], needs_llm_fn=counts["fn"], needs_llm_fp=counts["fp"], needs_llm_tn=counts["tn"],
        needs_llm_recall=recall, needs_llm_precision=_ratio(counts["tp"], counts["tp"] + counts["fp"]),
        weighted_recall=_ratio(weighted_tp, weighted_pos), min_recall=min_recall,
        passed=recall is not None and recall >= min_recall and not (len(labels.pages) - len(done)),
        missed=tuple(sorted(missed, key=lambda m: (m.machine_id, m.path, m.page))),
        git_commit=git_commit, created_at=created_at,
    )
