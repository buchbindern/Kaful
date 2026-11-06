"""Read one document offline, the way ingest does (steps b, c, d), for tools and tests.

Not used during a run: phase.py calls the steps itself. Used by tools/commands (grounding
preview, classifier eval, review page), the ingest tests and the understand tests.
"""

from pathlib import Path

from kaful.artifacts.document_index import DocumentRecord
from kaful.config import IngestSettings
from kaful.phases.p1_ingest.b_read_pages import read_pages
from kaful.phases.p1_ingest.c_classify_pages import classify_pages
from kaful.phases.p1_ingest.d_add_ruled_tables import add_ruled_tables
from kaful.phases.p1_ingest.helpers.documents import ReadDocument


def read_document(machine_dir: Path, record: DocumentRecord, settings: IngestSettings) -> ReadDocument:
    document = classify_pages(read_pages(machine_dir, record, settings), settings.classifier)
    return add_ruled_tables(machine_dir, document, settings.classifier)
