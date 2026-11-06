"""Classifier review sample and scoring against hand labels."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kaful.artifacts.document_index import PageClass
from kaful.config import IngestSettings, MachineConfig
from kaful.phases.p1_ingest.a_check_documents import check_documents
from kaful.phases.p1_ingest.tools.classifier_eval import (
    LabelledPage,
    PageLabels,
    evaluate_labels,
    labels_toml,
    read_labels,
)
from kaful.phases.p1_ingest.tools.read_document import read_document
from kaful.phases.p1_ingest.tools.review import (
    MachineDocs,
    review_html,
    sample_labels,
    sample_pages,
)
from tests.harness.kaful_pdf import write_manual

NOW = datetime(2026, 11, 3, tzinfo=UTC)


def _machine(tmp_path: Path, machine_id: str) -> MachineDocs:
    machine_dir = tmp_path / machine_id
    write_manual(machine_dir / "manual" / "a.pdf")
    write_manual(machine_dir / "manual" / "b.pdf", kinds=("text", "text", "figure_image", "table_unruled"))
    config = MachineConfig.model_validate({"machine_id": machine_id, "display_name": "x", "time_unit": "hour",
        "documents": [{"path": "manual/a.pdf", "kind": "manual"}, {"path": "manual/b.pdf", "kind": "manual"}]})
    records = check_documents(machine_dir, config).records
    return MachineDocs(machine_id, machine_dir, tuple(read_document(machine_dir, r, IngestSettings()) for r in records))


def test_sample_is_stratified_deterministic_and_order_free(tmp_path: Path) -> None:
    machines = [_machine(tmp_path, "m_one"), _machine(tmp_path, "m_two")]
    sampled = sample_pages(machines, per_class=2)
    assert sample_pages(list(reversed(machines)), per_class=2) == sampled
    entries = [s.entry for s in sampled]
    texts = [e for e in entries if e.machine_id == "m_one" and e.predicted == "text"]
    assert len(texts) == 2 and {(e.stratum_size, e.stratum_sampled) for e in texts} == {(3, 2)}
    labels = sample_labels(sampled, per_class=2)
    html = review_html(sampled, labels)
    assert html.count('<section class="card"') == len(entries) and "data:image/png;base64," in html


def test_labels_round_trip_through_toml(tmp_path: Path) -> None:
    labels = sample_labels(sample_pages([_machine(tmp_path, "m_one")], per_class=1), per_class=1)
    first = labels.pages[0].model_copy(update={"label": "figure", "note": 'odd "quote" \\ here'})
    labels = labels.model_copy(update={"pages": (first, *labels.pages[1:])})
    path = tmp_path / "labels.toml"
    path.write_text(labels_toml(labels))
    assert read_labels(path) == labels


def _page(n: int, predicted: PageClass, label: PageClass | None, size: int = 4) -> LabelledPage:
    return LabelledPage(machine_id="m", path="manual/a.pdf", doc_sha256="a" * 64, page=n, predicted=predicted,
                        stratum_size=size, stratum_sampled=2, label=label)


def test_needs_llm_recall_precision_and_misses() -> None:
    pages = (_page(1, "figure", "figure"), _page(2, "text", "table_unruled", size=40), _page(3, "text", "text"),
             _page(4, "figure", "text"), _page(5, "table_ruled", "table_ruled"))
    labels = PageLabels(sampled_with="classifier-v1", per_class=2, pages=pages)
    predictions = {p.key: p.predicted for p in pages}
    result = evaluate_labels(labels, predictions, classifier_version="classifier-v1", thresholds={},
                             min_recall=0.9, git_commit="x", created_at=NOW)
    assert (result.needs_llm_tp, result.needs_llm_fn, result.needs_llm_fp, result.needs_llm_tn) == (1, 1, 1, 2)
    assert result.needs_llm_recall == 0.5 and result.needs_llm_precision == 0.5 and not result.passed
    assert result.weighted_recall == pytest.approx(2 / (2 + 20))
    assert [(m.page, m.label) for m in result.missed] == [(2, "table_unruled")]


def test_unlabelled_pages_block_the_gate() -> None:
    pages = (_page(1, "figure", "figure"), _page(2, "text", None))
    labels = PageLabels(sampled_with="classifier-v1", per_class=2, pages=pages)
    result = evaluate_labels(labels, {p.key: p.predicted for p in pages}, classifier_version="classifier-v1",
                             thresholds={}, min_recall=0.9, git_commit="x", created_at=NOW)
    assert result.needs_llm_recall == 1.0 and result.n_unlabelled == 1 and not result.passed
