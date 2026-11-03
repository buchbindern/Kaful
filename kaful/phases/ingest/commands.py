"""`kaful ingest review | classifier-eval | grounding` (offline; no API calls).

    review           sample pages per predicted class and write the labelling page to
                     outputs/ingest_review/ (gitignored: it embeds manual page images)
    classifier-eval  score the current classifier on tests/fixtures/ingest/page_labels.toml;
                     result in evals/results/ingest_classifier/<classifier version>.json
"""

import argparse
import sys
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import PageClass
from kaful.config import REPO_ROOT, KafulSettings, load_machine_config
from kaful.paths import resolve_in
from kaful.phases.ingest.classifier_eval import LabelKey, evaluate_labels, read_labels
from kaful.phases.ingest.classify import CLASSIFIER_VERSION, classify_page
from kaful.phases.ingest.documents import check_documents
from kaful.phases.ingest.pages import page_signals, raw_text
from kaful.phases.ingest.read import read_document
from kaful.phases.ingest.review import MachineDocs, sample_labels, sample_pages, write_review
from kaful.runs import code_version, dump_json, now_utc, sha256_file, write_text_atomic

REVIEW_DIR = REPO_ROOT / "outputs" / "ingest_review"
LABELS_FILE = REPO_ROOT / "tests" / "fixtures" / "ingest" / "page_labels.toml"
CLASSIFIER_RESULTS = REPO_ROOT / "evals" / "results" / "ingest_classifier"
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
