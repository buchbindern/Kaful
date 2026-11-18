"""Step b: grounding. Evidence quotes by verification class, provenance downgraded by code (rule 4), parameters by
source and role, and the ingest index's page coverage on the fixed M6 quote set.

Counts walk the loaded artifacts (typed, rule 8): every `Evidence` is a quote, every `Provenanced` an item carrying a
source. A `deployment` parameter is declared in machine.toml, not documented: listed by name.
"""

import re
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from pydantic import BaseModel

from kaful.artifacts import ARTIFACT_MODELS
from kaful.artifacts.base import SOURCE_KINDS, Evidence, EvidenceClass, Provenanced
from kaful.artifacts.evaluation import EvidenceCount, Grounding, OldOnlyReview, PageCoverage, SourceRoleCount
from kaful.artifacts.param_set import ParamSet
from kaful.config import REPO_ROOT, EvaluateSettings
from kaful.evals.grounding import GroundingReport
from kaful.paths import resolve_in
from kaful.phases.p9_evaluate.a_resolve_lineage import Lineage
from kaful.phases.p9_evaluate.helpers.files import repo_ref
from kaful.runs import load_artifact

MATCHER = re.compile(r"__matcher-v(\d+)\.json$")
OLD_ONLY_REVIEW = "old_only_review.toml"  # M6 hand review, beside the grounding report (ingest grounding tool)


def walk(node: object) -> Iterator[BaseModel]:
    """Every model inside `node` (itself included), depth first, in field order."""
    if isinstance(node, BaseModel):
        yield node
        for name in type(node).model_fields:
            yield from walk(getattr(node, name))
    elif isinstance(node, (tuple, list)):
        for item in node:
            yield from walk(item)
    elif isinstance(node, dict):
        for key in sorted(node):
            yield from walk(node[key])


def count(kind: str, artifact: BaseModel) -> EvidenceCount:
    quotes: Counter[str] = Counter()
    downgrades: Counter[str] = Counter()
    provenanced = 0
    for m in walk(artifact):
        if isinstance(m, Evidence):
            quotes[m.verification] += 1
        if isinstance(m, Provenanced):
            provenanced += 1
            if m.downgraded_from is not None:
                downgrades[f"{m.downgraded_from} -> {m.source}"] += 1
    return EvidenceCount(kind=kind, quotes={c: quotes[c] for c in get_args(EvidenceClass)}, provenanced=provenanced,
                         downgraded=dict(sorted(downgrades.items())))


def page_coverage(machine_id: str, ingest_run: str | None, s: EvaluateSettings,
                  root: Path = REPO_ROOT) -> PageCoverage | None:
    """The newest matcher's M6 grounding report for exactly this ingest run; None when it was never scored."""
    if ingest_run is None or not resolve_in(root, s.ingest_grounding).is_dir():
        return None
    found = [(int(m.group(1)), p) for p in resolve_in(root, s.ingest_grounding).glob(f"{machine_id}__{ingest_run}__*")
             if (m := MATCHER.search(p.name))]
    if not found:
        return None
    _, path = max(found)
    report = GroundingReport.model_validate_json(path.read_bytes())
    if report.new_any_layer is None or report.new_coverage is None:
        return None
    rel = f"{s.ingest_grounding}/{path.name}"
    return PageCoverage(report=repo_ref(rel, root), matcher_version=report.matcher_version, n_quotes=report.n_quotes,
                        verbatim=report.new_any_layer.verbatim, near=report.new_any_layer.near,
                        unsupported=report.new_any_layer.unsupported, page_share=report.new_coverage.share,
                        old_verbatim=report.old.verbatim, old_near=report.old.near,
                        old_unsupported=report.old.unsupported, old_page_share=report.old_page_coverage,
                        old_partial=report.old_partial,
                        old_only_review=old_only_review(f"{s.ingest_grounding}/{machine_id}__{ingest_run}__"
                                                        f"{OLD_ONLY_REVIEW}", root))


def old_only_review(rel: str, root: Path = REPO_ROOT) -> OldOnlyReview | None:
    """Counts of the M6 old-only review's verdicts; None when the review file does not exist."""
    path = resolve_in(root, rel)
    if not path.is_file():
        return None
    verdicts = Counter(str(q.get("present", "")) for q in tomllib.loads(path.read_text(encoding="utf-8"))
                       .get("quotes", []))
    return OldOnlyReview(file=repo_ref(rel, root), yes=verdicts["yes"], partial=verdicts["partial"], no=verdicts["no"],
                         unreviewed=sum(n for v, n in verdicts.items() if v not in ("yes", "partial", "no")))


@dataclass(frozen=True)
class ParamCounts:
    by_source: dict[str, int]
    by_role: tuple[SourceRoleCount, ...]
    statements: dict[str, int]
    declared: tuple[str, ...]


def params(ps: ParamSet) -> ParamCounts:
    pairs = Counter((p.source, p.role) for p in ps.parameters)
    statements = Counter(st.source for st in ps.life_statements)
    return ParamCounts(by_source={str(k): v for k, v in ps.counts_by_source().items()},
                       by_role=tuple(SourceRoleCount(source=s, role=r, count=n) for (s, r), n in sorted(pairs.items())),
                       statements={k: statements[k] for k in SOURCE_KINDS if statements[k]},
                       declared=tuple(sorted(p.param_id for p in ps.parameters if p.source == "deployment")))


def grounding(lineage: Lineage, machine_dir: Path, s: EvaluateSettings, root: Path = REPO_ROOT) -> Grounding:
    """Called only when understand was reached; later phases count when they were reached."""
    evidence = []
    for kind in sorted(s.evidence_kinds):
        for rec in lineage.phases:
            ref = lineage.ref(rec.phase, kind) if lineage.reached(rec.phase) else None
            if ref is not None:
                evidence.append(count(kind, load_artifact(machine_dir, ref, ARTIFACT_MODELS[kind])))
                break
    counts = ParamCounts(by_source={k: 0 for k in SOURCE_KINDS}, by_role=(), statements={}, declared=())
    notes = []
    ps_ref = lineage.ref("parameterize", "param_set") if lineage.reached("parameterize") else None
    if ps_ref is not None:
        counts = params(load_artifact(machine_dir, ps_ref, ParamSet))
    else:
        notes.append(f"parameters not counted: {lineage.why_not('parameterize')}")
    cov = page_coverage(lineage.machine_id, lineage.get("ingest").produced_by, s, root)
    if cov is None:
        notes.append(f"no M6 grounding report for ingest run {lineage.get('ingest').produced_by} "
                     f"(`kaful ingest grounding`): page coverage not reported")
    return Grounding(evidence=tuple(evidence), params_by_source=counts.by_source, params_by_source_role=counts.by_role,
                     life_statements_by_source=counts.statements, declared_params=counts.declared, page_coverage=cov,
                     notes=tuple(notes))
