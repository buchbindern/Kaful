"""Compare two eval results (two prompt versions, or one version on two models).

The comparison is paired on the cases both results share. With eval sets this
small it reports counts and per-case changes, never significance.
"""

from typing import Annotated

from pydantic import Field

from kaful.artifacts.base import NonEmptyStr, StrictModel, by_attr, sorted_unique
from kaful.evals.models import EvalResult


class MetricDelta(StrictModel):
    name: NonEmptyStr
    a: float | None
    b: float | None
    delta: float | None


class CaseChange(StrictModel):
    case_id: NonEmptyStr
    metric: NonEmptyStr
    a: float | None
    b: float | None


class ResultRef(StrictModel):
    version: int
    model: NonEmptyStr
    content_sha256: NonEmptyStr
    passed: bool
    cost_usd: float = Field(ge=0.0, description="Billed plus replayed cost: what one live run costs.")
    input_tokens: int
    output_tokens: int


class EvalComparison(StrictModel):
    schema_version: int = 1
    kind: str = "eval_comparison"
    prompt_id: NonEmptyStr
    a: ResultRef
    b: ResultRef
    metrics: Annotated[tuple[MetricDelta, ...], by_attr("name")]
    case_changes: Annotated[tuple[CaseChange, ...], sorted_unique(lambda c: (c.case_id, c.metric), "case_changes")]
    only_in_a: tuple[str, ...]
    only_in_b: tuple[str, ...]


def _ref(result: EvalResult) -> ResultRef:
    return ResultRef(version=result.version, model=result.model, content_sha256=result.content_sha256,
                     passed=result.passed, cost_usd=result.billed_cost_usd + result.unbilled_cost_usd,
                     input_tokens=result.input_tokens, output_tokens=result.output_tokens)


def compare(a: EvalResult, b: EvalResult) -> EvalComparison:
    if a.prompt_id != b.prompt_id:
        raise ValueError(f"cannot compare {a.prompt_id} with {b.prompt_id}")
    names = sorted({m.name for m in a.metrics} | {m.name for m in b.metrics})
    metrics = []
    for name in names:
        va, vb = a.metric(name), b.metric(name)
        metrics.append(MetricDelta(name=name, a=va, b=vb, delta=None if va is None or vb is None else vb - va))
    cases_a = {c.case_id: c for c in a.cases}
    cases_b = {c.case_id: c for c in b.cases}
    changes = []
    for case_id in sorted(cases_a.keys() & cases_b.keys()):
        ca, cb = cases_a[case_id], cases_b[case_id]
        for name in sorted({s.name for s in ca.scores} | {s.name for s in cb.scores}):
            if ca.score(name) != cb.score(name):
                changes.append(CaseChange(case_id=case_id, metric=name, a=ca.score(name), b=cb.score(name)))
    return EvalComparison(
        prompt_id=a.prompt_id, a=_ref(a), b=_ref(b), metrics=tuple(metrics), case_changes=tuple(changes),
        only_in_a=tuple(sorted(cases_a.keys() - cases_b.keys())),
        only_in_b=tuple(sorted(cases_b.keys() - cases_a.keys())),
    )


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def render_markdown(cmp: EvalComparison) -> str:
    a, b = cmp.a, cmp.b
    lines = [
        f"# {cmp.prompt_id}: v{a.version} ({a.model}) vs v{b.version} ({b.model})",
        "",
        f"- passed: {a.passed} -> {b.passed}",
        f"- cost per run: ${a.cost_usd:.4f} -> ${b.cost_usd:.4f}; "
        f"tokens in/out: {a.input_tokens}/{a.output_tokens} -> {b.input_tokens}/{b.output_tokens}",
        "",
        "| metric | A | B | delta |",
        "|---|---|---|---|",
        *(f"| {m.name} | {_fmt(m.a)} | {_fmt(m.b)} | {_fmt(m.delta)} |" for m in cmp.metrics),
    ]
    if cmp.case_changes:
        lines += ["", "| case | metric | A | B |", "|---|---|---|---|",
                  *(f"| {c.case_id} | {c.metric} | {_fmt(c.a)} | {_fmt(c.b)} |" for c in cmp.case_changes)]
    if cmp.only_in_a or cmp.only_in_b:
        lines += ["", f"Cases only in A: {list(cmp.only_in_a)}; only in B: {list(cmp.only_in_b)}"]
    return "\n".join(lines) + "\n"
