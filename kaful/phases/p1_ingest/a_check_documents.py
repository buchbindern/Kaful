"""The documents machine.toml lists, validated against the files on disk.

Errors (always): a listed file is missing, is not a PDF, cannot be opened or has no
pages, two entries have the same content, or two file names map to the same doc_id.
Warning (an error with `strict`): a PDF under manual/ that machine.toml does not list.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # type: ignore[import-untyped]

from kaful.artifacts.document_index import DocumentRecord
from kaful.config import MachineConfig
from kaful.paths import resolve_in, to_rel
from kaful.runs import sha256_file

log = logging.getLogger("kaful.ingest.documents")

PDF_MAGIC = b"%PDF-"
MANUAL_SUBDIR = "manual"


class DocumentError(ValueError):
    """machine.toml's document list does not match the files on disk."""


@dataclass(frozen=True)
class DocumentCheck:
    records: tuple[DocumentRecord, ...]
    warnings: tuple[str, ...]


def doc_id_for(path: str) -> str:
    """A document id from its file name: lowercase stem, non-alphanumerics as '_'."""
    stem = Path(path).stem.lower()
    doc_id = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if not re.match(r"^[a-z][a-z0-9_]*$", doc_id):
        raise DocumentError(f"{path}: cannot derive a document id (must start with a letter)")
    return doc_id


def _record(machine_dir: Path, rel: str, kind: str, ruled_tables: str) -> DocumentRecord:
    path = resolve_in(machine_dir, rel)
    if not path.is_file():
        raise DocumentError(f"listed document {rel!r} not found")
    with path.open("rb") as handle:
        if handle.read(len(PDF_MAGIC)) != PDF_MAGIC:
            raise DocumentError(f"listed document {rel!r} is not a PDF")
    try:
        with fitz.open(path) as doc:
            n_pages = doc.page_count
            title = (doc.metadata or {}).get("title") or None
    except (RuntimeError, ValueError) as exc:
        raise DocumentError(f"listed document {rel!r} cannot be opened: {exc}") from exc
    if n_pages < 1:
        raise DocumentError(f"listed document {rel!r} has no pages")
    return DocumentRecord(doc_id=doc_id_for(rel), path=rel, kind=kind, sha256=sha256_file(path), n_pages=n_pages,
                          title=title, ruled_tables=ruled_tables)


def _unlisted(machine_dir: Path, listed: set[str]) -> list[str]:
    manual = machine_dir / MANUAL_SUBDIR
    if not manual.is_dir():
        return []
    pdfs = (p for p in manual.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    return sorted(rel for rel in (to_rel(machine_dir, p) for p in pdfs) if rel not in listed)


def check_documents(machine_dir: Path, config: MachineConfig, strict: bool = False) -> DocumentCheck:
    """DocumentRecords for every listed document, sorted by doc_id; raise DocumentError on a mismatch."""
    if not config.documents:
        raise DocumentError(f"{config.machine_id}: machine.toml lists no documents")
    records = [_record(machine_dir, d.path, d.kind, d.ruled_tables) for d in config.documents]
    for attr in ("doc_id", "sha256"):
        values = [getattr(r, attr) for r in records]
        dupes = sorted({v for v in values if values.count(v) > 1})
        if dupes:
            paths = sorted(r.path for r in records if getattr(r, attr) in dupes)
            raise DocumentError(f"documents share a {attr}: {paths}")
    unlisted = _unlisted(machine_dir, {d.path for d in config.documents})
    warnings = tuple(f"{rel} is under {MANUAL_SUBDIR}/ but not listed in machine.toml" for rel in unlisted)
    if warnings and strict:
        raise DocumentError("; ".join(warnings))
    for warning in warnings:
        log.warning("unlisted document", extra={"fields": {"detail": warning}})
    return DocumentCheck(tuple(sorted(records, key=lambda r: r.doc_id)), warnings)
