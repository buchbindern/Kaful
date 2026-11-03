"""Eval sets (input) and eval results (output).

A prompt's eval set lives in its `evals/` directory:

    evals/evalset.toml     deterministic checks, plus the judge prompt (if any)
    evals/cases/*.toml     one case each: inputs, and golden expectations for key fields
    evals/sources/...      long input texts the cases refer to

Results go to evals/results/<prompt_id>/v<version>__<model>.json (committed; written
only by `kaful eval run`), so prompt versions and models can be compared.
"""

import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import NonEmptyStr, Sha256, StrictModel, by_attr, sorted_unique
from kaful.paths import RelPath, resolve_in
from kaful.prompts.registry import Prompt, Threshold

EVALS_SUBDIR = "evals"
EVALSET_FILE = "evalset.toml"
CASES_SUBDIR = "cases"

CheckKind = Literal["quote_in_source", "in_range", "count_between", "unit_parses"]
Compare = Literal["exact", "normalized_str", "rel_tol", "set_f1", "keyed", "violation"]
QuoteClass = Literal["verbatim", "near", "unsupported"]


class CheckSpec(StrictModel):
    """A deterministic check. `field` is a path like 'channels[].evidence'."""

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: CheckKind
    field: NonEmptyStr
    source: str | None = Field(default=None, description="quote_in_source: the input holding the source text.")
    judge: bool = Field(default=False, description="quote_in_source: send unsupported quotes to the judge.")
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_args(self) -> "CheckSpec":
        if self.kind == "quote_in_source" and not self.source:
            raise ValueError(f"{self.name}: quote_in_source needs `source`")
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


class GoldenSpec(StrictModel):
    """One hand-labelled expectation, scored in [0, 1].

    exact / normalized_str / rel_tol: `field` resolves to one value.
    set_f1: `field` resolves to a list; F1 against the expected list.
    keyed: `field` is a list of objects; `expected` maps item[`key`] -> item[`value`];
        score = share of expected keys whose value matches (strings normalised,
        numbers within `rel_tol`).
    violation: 1.0 when the value equals `expected` (a forbidden answer), else 0.0;
        use with a `max` threshold.
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
    inputs: dict[str, str | InputFile]
    golden: tuple[GoldenSpec, ...] = ()
    tags: tuple[str, ...] = ()


class EvalSet:
    """A prompt's eval set, loaded from disk with input files read."""

    def __init__(self, prompt: Prompt) -> None:
        self.directory = prompt.directory / EVALS_SUBDIR
        with (self.directory / EVALSET_FILE).open("rb") as handle:
            self.spec = EvalSetSpec.model_validate(tomllib.load(handle))
        cases = []
        for path in sorted((self.directory / CASES_SUBDIR).glob("*.toml")):
            with path.open("rb") as handle:
                cases.append(EvalCase.model_validate(tomllib.load(handle)))
        if not cases:
            raise ValueError(f"{prompt.id}: eval set has no cases")
        self.cases = tuple(sorted(cases, key=lambda c: c.id))

    def inputs(self, case: EvalCase) -> dict[str, str]:
        return {name: (value if isinstance(value, str)
                       else resolve_in(self.directory, value.file).read_text(encoding="utf-8"))
                for name, value in sorted(case.inputs.items())}


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

    def metric(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


def results_path(results_root: Path, prompt_id: str, version: int, model: str) -> Path:
    return results_root / prompt_id / f"v{version}__{model}.json"
