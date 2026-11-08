"""Eval sets (input) and eval results (output).

A prompt's eval set lives in its `evals/` directory:

    evals/evalset.toml     deterministic checks, plus the judge prompt (if any)
    evals/cases/*.toml     one case each: inputs, and golden expectations for key fields
    evals/sources/...      long input texts the cases refer to

A case can also take an input (raw text) or an image (PNG) from one page of a machine
document (`from_page`). Manuals are not committed, so such a case needs the document on
disk with the recorded sha256; otherwise loading its inputs raises SourceUnavailable.

Results go to evals/results/<prompt_id>/v<version>__<model>.json (committed; written
only by `kaful eval run`), so prompt versions and models can be compared.
"""

import hashlib
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import fitz  # type: ignore[import-untyped]
from pydantic import Field, model_validator

from kaful.artifacts.base import Identifier, NonEmptyStr, Sha256, StrictModel, by_attr, sorted_unique
from kaful.config import KafulSettings
from kaful.paths import RelPath, resolve_in
from kaful.phases.p1_ingest.helpers.page_signals import raw_text
from kaful.phases.p1_ingest.helpers.render_pages import page_png
from kaful.prompts.registry import Prompt, Threshold

EVALS_SUBDIR = "evals"
EVALSET_FILE = "evalset.toml"
CASES_SUBDIR = "cases"

CheckKind = Literal["quote_in_source", "in_range", "count_between", "unit_parses", "numbers_in_source",
                    "listed_in_source"]
Compare = Literal["exact", "normalized_str", "rel_tol", "set_f1", "keyed", "violation", "contains_all", "within",
                  "excludes_all", "not_in", "at_most_once"]
QuoteClass = Literal["verbatim", "near", "unsupported"]


class CheckSpec(StrictModel):
    """A deterministic check. `field` is a path like 'channels[].evidence' or 'tables[source=x].rows[]'."""

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: CheckKind
    field: NonEmptyStr
    source: str | None = Field(default=None, description="quote_in_source: the input holding the source text.")
    judge: bool = Field(default=False, description="quote_in_source: send unsupported quotes to the judge.")
    min: float | None = None
    max: float | None = None
    cases: tuple[NonEmptyStr, ...] | None = Field(default=None, description=(
        "Only these case ids (a check written for one machine's text, e.g. tiny_pump's flow path count); "
        "None: every case."))

    def applies_to(self, case_id: str) -> bool:
        return self.cases is None or case_id in self.cases

    @model_validator(mode="after")
    def _check_args(self) -> "CheckSpec":
        if self.kind in ("quote_in_source", "numbers_in_source", "listed_in_source") and not self.source:
            raise ValueError(f"{self.name}: {self.kind} needs `source`")
        if self.kind in ("in_range", "count_between") and self.min is None and self.max is None:
            raise ValueError(f"{self.name}: {self.kind} needs min and/or max")
        return self


class JudgeSpec(StrictModel):
    prompt: NonEmptyStr


class EvalSetSpec(StrictModel):
    checks: Annotated[tuple[CheckSpec, ...], by_attr("name")] = ()
    judge: JudgeSpec | None = None


class InputFile(StrictModel):
    file: RelPath


class SourcePage(StrictModel):
    """One page of a machine document (machines/<machine>/<path>), pinned by sha256."""

    machine: Identifier
    path: RelPath
    sha256: Sha256
    page: int = Field(ge=1)


class PageInput(StrictModel):
    """An input taken from the case's `page`: its raw text layer or its rendered PNG."""

    from_page: Literal["raw_text", "png"]


class SourceUnavailable(FileNotFoundError):
    """A case's source document is not on disk (manuals are not committed)."""


class GoldenSpec(StrictModel):
    """One hand-labelled expectation, scored in [0, 1].

    exact / normalized_str / rel_tol: `field` resolves to one value.
    set_f1: `field` resolves to a list; F1 against the expected list.
    keyed: `field` is a list of objects; `expected` maps item[`key`] -> item[`value`];
        score = share of expected keys whose value matches (strings normalised,
        numbers within `rel_tol`).
    violation: 1.0 when the value equals `expected` (a forbidden answer), else 0.0;
        use with a `max` threshold.
    contains_all: `field` resolves to values whose text (list items and object fields joined
        with ' | ', e.g. a table row's cells) is searched; score = share of the `expected` strings found in the joined
        text as whole words, compared case- and punctuation-insensitively. Decimal commas and points
        are kept apart, so '14,0' does not match '14.0'.
    within: `field` resolves to a list; score = share of its values that are in `expected`
        (precision; 1.0 for an empty list). Catches over-extraction.
    excludes_all: like contains_all, but score = share of the `expected` (forbidden) words
        that do NOT occur.
    at_most_once: score = share of the `expected` phrases that at most one value contains (every word
        of the phrase, case- and punctuation-insensitively). Catches one quantity reported twice.
    not_in: score = share of the values `field` resolves to that are NOT in `expected`
        (1.0 when nothing resolves). E.g. an example value's role must not be a spec role.
    """

    metric: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.]*$")]
    field: NonEmptyStr
    compare: Compare
    expected: Any
    key: str | None = None
    value: str | None = None
    rel_tol: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _keyed_args(self) -> "GoldenSpec":
        if self.compare == "keyed" and not (self.key and self.value and isinstance(self.expected, dict)):
            raise ValueError(f"{self.metric}: keyed needs key, value and an expected table")
        return self


class EvalCase(StrictModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]*$")]
    description: NonEmptyStr
    page: SourcePage | None = None
    inputs: dict[str, str | InputFile | PageInput]
    images: dict[str, PageInput] = {}
    golden: tuple[GoldenSpec, ...] = ()
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _page_inputs(self) -> "EvalCase":
        uses_page = [v for v in [*self.inputs.values(), *self.images.values()] if isinstance(v, PageInput)]
        if uses_page and self.page is None:
            raise ValueError(f"{self.id}: from_page inputs need a `page`")
        if any(v.from_page != "png" for v in self.images.values()):
            raise ValueError(f"{self.id}: images must use from_page = 'png'")
        return self


class EvalSet:
    """A prompt's eval set, loaded from disk with input files read."""

    def __init__(self, prompt: Prompt, settings: KafulSettings | None = None, tag: str | None = None) -> None:
        self._settings = settings
        self.prompt = prompt
        self.directory = prompt.directory / EVALS_SUBDIR
        with (self.directory / EVALSET_FILE).open("rb") as handle:
            self.spec = EvalSetSpec.model_validate(tomllib.load(handle))
        cases = []
        for path in sorted((self.directory / CASES_SUBDIR).glob("*.toml")):
            with path.open("rb") as handle:
                cases.append(EvalCase.model_validate(tomllib.load(handle)))
        self.tag = tag
        if tag is not None:
            cases = [c for c in cases if tag in c.tags]
        if not cases:
            raise ValueError(f"{prompt.id}: eval set has no cases" + (f" tagged {tag!r}" if tag else ""))
        self.cases = tuple(sorted(cases, key=lambda c: c.id))

    def inputs(self, case: EvalCase) -> dict[str, str]:
        def text(value: str | InputFile | PageInput) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, InputFile):
                return resolve_in(self.directory, value.file).read_text(encoding="utf-8")
            return self._page_raw_text(case)

        return {name: text(value) for name, value in sorted(case.inputs.items())}

    def images(self, case: EvalCase) -> dict[str, bytes]:
        return {name: self._page_png(case) for name in sorted(case.images)}

    def page_available(self, case: EvalCase) -> bool:
        try:
            self._page_file(case)
        except SourceUnavailable:
            return False
        return True

    def machine_dir(self, machine: str) -> Path:
        return (self._settings or KafulSettings()).machine_dir(machine)

    def _page_file(self, case: EvalCase) -> Path:
        assert case.page is not None
        path = resolve_in(self.machine_dir(case.page.machine), case.page.path)
        if not path.is_file():
            raise SourceUnavailable(f"{case.id}: {case.page.machine}/{case.page.path} is not on disk")
        if hashlib.sha256(path.read_bytes()).hexdigest() != case.page.sha256:
            raise ValueError(f"{case.id}: {case.page.machine}/{case.page.path} differs from the recorded sha256")
        return path

    def _page_raw_text(self, case: EvalCase) -> str:
        assert case.page is not None
        with fitz.open(self._page_file(case)) as doc:
            return raw_text(doc[case.page.page - 1])

    def _page_png(self, case: EvalCase) -> bytes:
        assert case.page is not None
        settings = self._settings or KafulSettings()
        with fitz.open(self._page_file(case)) as doc:
            return page_png(doc[case.page.page - 1], settings.ingest)


# ── Results ───────────────────────────────────────────────────────────────────

class CheckOutcome(StrictModel):
    check: NonEmptyStr
    status: Literal["pass", "fail", "skip"]
    details: tuple[str, ...] = ()


class MetricScore(StrictModel):
    name: NonEmptyStr
    value: float = Field(ge=0.0, le=1.0)


class CaseResult(StrictModel):
    case_id: NonEmptyStr
    attempts: int | None = Field(default=None, description="Validation attempts; None if the call failed.")
    error: str | None = None
    scores: Annotated[tuple[MetricScore, ...], by_attr("name")]
    checks: Annotated[tuple[CheckOutcome, ...], by_attr("check")] = ()

    def score(self, name: str) -> float | None:
        return next((s.value for s in self.scores if s.name == name), None)


class MetricSummary(StrictModel):
    name: NonEmptyStr
    value: float = Field(description="Mean over the cases that produced this metric.")
    n_cases: int = Field(ge=1)


class ThresholdOutcome(StrictModel):
    metric: NonEmptyStr
    threshold: Threshold
    value: float | None
    passed: bool


class JudgeRef(StrictModel):
    prompt_id: NonEmptyStr
    version: int
    content_sha256: Sha256
    model: NonEmptyStr
    effort: str | None = None


class EvalResult(StrictModel):
    """Scores of one prompt version on one model over its eval set."""

    schema_version: Literal[1] = 1
    kind: Literal["eval_result"] = "eval_result"
    prompt_id: NonEmptyStr
    version: int
    content_sha256: Sha256
    model: NonEmptyStr
    effort: str | None
    judge: JudgeRef | None
    source: Literal["offline", "live"]
    cases: Annotated[tuple[CaseResult, ...], by_attr("case_id")]
    metrics: Annotated[tuple[MetricSummary, ...], by_attr("name")]
    thresholds: Annotated[tuple[ThresholdOutcome, ...], sorted_unique(lambda t: t.metric, "metric")]
    passed: bool
    billed_cost_usd: float = Field(ge=0.0)
    unbilled_cost_usd: float = Field(ge=0.0, description="Cost of the recorded responses this run replayed.")
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    git_commit: NonEmptyStr
    created_at: datetime
    case_tag: str | None = Field(default=None, description="Only cases with this tag were run; never lockable.")
    sample: int = Field(default=0, ge=0, description="Repeat index: run k uses sample k of every request.")
    routing: str | None = Field(default=None, description="Routed run: '<model> -> <escalation model> when <rule>'.")

    def metric(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


def results_path(results_root: Path, prompt_id: str, version: int, model: str, case_tag: str | None = None,
                 sample: int = 0) -> Path:
    suffix = (f"__tag-{case_tag}" if case_tag else "") + (f"__run-{sample}" if sample else "")
    return results_root / prompt_id / f"v{version}__{model}{suffix}.json"
