"""`kaful ingest review | classifier-eval | grounding` (offline; no API calls).

    review           sample pages per predicted class and write the labelling page to
                     outputs/ingest_review/ (gitignored: it embeds manual page images)
    classifier-eval  score the current classifier on tests/fixtures/ingest/page_labels.toml;
                     result in evals/results/ingest_classifier/<classifier version>.json
    grounding        score the frozen legacy quote set against the old index and the new
                     index of an ingest run (--run), or raw text only (preview); results in
                     evals/results/ingest_grounding/. --freeze writes the quote set once.
"""

import argparse
import sys
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import DocumentIndex, PageClass, PageText
from kaful.evals.grounding import (
    GroundingReport,
    QuoteSet,
    Score,
    build_report,
    collect_quotes,
    old_only_review,
    read_old_index,
)
from kaful.config import REPO_ROOT, KafulSettings, load_machine_config
from kaful.paths import resolve_in
from kaful.phases.ingest.classifier_eval import LabelKey, evaluate_labels, read_labels
from kaful.phases.ingest.classify import CLASSIFIER_VERSION, classify_page
from kaful.phases.ingest.documents import check_documents
from kaful.phases.ingest.pages import page_signals, raw_text
from kaful.phases.ingest.read import read_document
from kaful.phases.ingest.review import MachineDocs, sample_labels, sample_pages, write_review
from kaful.runs import (
    code_version,
    dump_json,
    load_artifact,
    now_utc,
    read_manifest,
    runs_dir,
    sha256_bytes,
    sha256_file,
    write_text_atomic,
)

REVIEW_DIR = REPO_ROOT / "outputs" / "ingest_review"
LABELS_FILE = REPO_ROOT / "tests" / "fixtures" / "ingest" / "page_labels.toml"
CLASSIFIER_RESULTS = REPO_ROOT / "evals" / "results" / "ingest_classifier"
QUOTES_DIR = REPO_ROOT / "tests" / "fixtures" / "ingest" / "grounding_quotes"
GROUNDING_RESULTS = REPO_ROOT / "evals" / "results" / "ingest_grounding"
REVIEW_MACHINES = ("cnc_spindle", "eversys_coffee_machine")

EXIT_OK = 0
EXIT_FAILED = 1


def add_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    ingest = commands.add_parser("ingest", help="ingest tools (offline)").add_subparsers(
        dest="ingest_command", required=True)
    review = ingest.add_parser("review", help="write the classifier labelling page")
    review.add_argument("machines", nargs="*", default=list(REVIEW_MACHINES))
    review.add_argument("--per-class", type=int, default=None)
    cls = ingest.add_parser("classifier-eval", help="score the classifier on the hand labels")
    cls.add_argument("--labels", type=Path, default=LABELS_FILE)
    grounding = ingest.add_parser("grounding", help="M6 grounding and coverage report")
    grounding.add_argument("machine")
    grounding.add_argument("--run", default=None, help="ingest run id to score (default: raw-text preview)")
    grounding.add_argument("--freeze", action="store_true", help="write the quote set if it does not exist yet")


def _machine_docs(settings: KafulSettings, machine_id: str) -> MachineDocs:
    machine_dir = settings.machine_dir(machine_id)
    config = load_machine_config(machine_dir)
    records = check_documents(machine_dir, config).records
    return MachineDocs(machine_id, machine_dir, tuple(read_document(machine_dir, r, settings.ingest) for r in records))


def ingest_review(args: argparse.Namespace, settings: KafulSettings) -> int:
    per_class = args.per_class or settings.ingest.review_per_class
    machines = [_machine_docs(settings, m) for m in sorted(args.machines)]
    sampled = sample_pages(machines, per_class)
    page, blank = write_review(REVIEW_DIR, sampled, sample_labels(sampled, per_class))
    sys.stdout.write(f"{len(sampled)} pages sampled\n  review page: {page}\n  blank labels: {blank}\n")
    return EXIT_OK


def _predictions(settings: KafulSettings, keys: list[LabelKey], shas: dict[LabelKey, str]) -> dict[LabelKey, PageClass]:
    predictions: dict[LabelKey, PageClass] = {}
    for machine_id, path, page_no in keys:
        file = resolve_in(settings.machine_dir(machine_id), path)
        if sha256_file(file) != shas[(machine_id, path, page_no)]:
            raise ValueError(f"{machine_id}/{path} changed since the pages were labelled")
        with fitz.open(file) as doc:
            page = doc[page_no - 1]
            signals = page_signals(page, raw_text(page), settings.ingest)
        predictions[(machine_id, path, page_no)] = classify_page(signals, settings.ingest.classifier)
    return predictions


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def ingest_classifier_eval(args: argparse.Namespace, settings: KafulSettings) -> int:
    labels = read_labels(args.labels)
    keys = sorted(p.key for p in labels.pages)
    predictions = _predictions(settings, keys, {p.key: p.doc_sha256 for p in labels.pages})
    thresholds = {k: float(v) for k, v in settings.ingest.classifier.model_dump().items()}
    result = evaluate_labels(labels, predictions, classifier_version=CLASSIFIER_VERSION, thresholds=thresholds,
                             min_recall=settings.ingest.classifier_min_recall,
                             git_commit=code_version().git_commit, created_at=now_utc())
    path = CLASSIFIER_RESULTS / f"{CLASSIFIER_VERSION}.json"
    write_text_atomic(path, dump_json(result))
    sys.stdout.write(
        f"{CLASSIFIER_VERSION}: {'PASSED' if result.passed else 'FAILED'} "
        f"({result.n_labelled} labelled, {result.n_unlabelled} unlabelled)\n"
        f"  needs-LLM recall {_fmt(result.needs_llm_recall)} (min {result.min_recall}), "
        f"precision {_fmt(result.needs_llm_precision)}, weighted recall {_fmt(result.weighted_recall)}\n"
        f"  missed: {[f'{m.machine_id}/{m.path} p{m.page}' for m in result.missed]}\n"
        f"  saved {path.relative_to(REPO_ROOT)}\n")
    return EXIT_OK if result.passed else EXIT_FAILED


def _quote_set(settings: KafulSettings, machine_id: str, freeze: bool) -> tuple[QuoteSet, str]:
    path = QUOTES_DIR / f"{machine_id}.json"
    if freeze:
        if path.is_file():
            raise ValueError(f"{path.relative_to(REPO_ROOT)} exists; the quote set is frozen")
        quotes = collect_quotes(settings.machine_dir(machine_id), machine_id, settings.ingest.grounding_min_tokens)
        write_text_atomic(path, dump_json(quotes))
    data = path.read_bytes()
    return QuoteSet.model_validate_json(data), sha256_bytes(data)


def _run_pages(settings: KafulSettings, machine_id: str, run_id: str) -> tuple[list[PageText], DocumentIndex]:
    machine_dir = settings.machine_dir(machine_id)
    manifest = read_manifest(runs_dir(machine_dir) / run_id)
    refs = [r for r in manifest.phase("ingest").artifacts if r.kind == "document_index"]
    if len(refs) != 1:
        raise ValueError(f"run {run_id} has no document_index")
    index = load_artifact(machine_dir, refs[0], DocumentIndex)
    return list(index.pages), index


def ingest_grounding(args: argparse.Namespace, settings: KafulSettings) -> int:
    machine_dir = settings.machine_dir(args.machine)
    records = check_documents(machine_dir, load_machine_config(machine_dir)).records
    quote_set, quote_sha = _quote_set(settings, args.machine, args.freeze)
    index: DocumentIndex | None = None
    if args.run:
        pages, index = _run_pages(settings, args.machine, args.run)
    else:
        pages = [p for r in records for p in read_document(machine_dir, r, settings.ingest).pages]
    old = read_old_index(machine_dir, {Path(r.path).name: r.doc_id for r in records}, len(pages))
    chunk_pages = None if index is None else {(c.doc_id, n) for c in index.chunks
                                              for n in range(c.page_start, c.page_end + 1)}
    report = build_report(quote_set, quote_sha, old, pages, settings.ingest, run_id=args.run,
                          index_hash=None if index is None else index.index_hash, chunk_pages=chunk_pages)
    name = f"{args.machine}__{args.run or 'raw_preview'}"
    write_text_atomic(GROUNDING_RESULTS / f"{name}.json", dump_json(report))
    if args.run:
        write_text_atomic(GROUNDING_RESULTS / f"{name}__old_only_review.toml", old_only_review(report, quote_set, pages))
    sys.stdout.write(_grounding_summary(report))
    return EXIT_OK


def _score_text(score: Score | None, n: int) -> str:
    if score is None:
        return "–"
    return f"{score.supported}/{n} (verbatim {score.verbatim}, near {score.near})"


def _grounding_summary(report: GroundingReport) -> str:
    cov = report.new_coverage
    return (f"{report.machine_id}: {report.n_quotes} quotes\n"
            f"  old index{' (partial)' if report.old_partial else ''}: {_score_text(report.old, report.n_quotes)}, "
            f"page coverage {report.old_page_coverage:.3f}\n"
            f"  new, any layer: {_score_text(report.new_any_layer, report.n_quotes)}\n"
            f"  new, raw text only: {_score_text(report.raw_only, report.n_quotes)}\n"
            f"  new coverage: {'–' if cov is None else f'{cov.pages_in_chunks}/{cov.pages_total - cov.pages_blank} non-blank pages'}\n"
            f"  old-only quotes: {len(report.old_only)}\n")
