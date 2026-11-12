"""Loading the document index for later phases, with its hash check.

`load_index(ctx)` loads the DocumentIndex this run (or its base run) produced and compares
its hash with the manifest's `index_hash`. A mismatch is a warning, or an error with
`--strict`. `load_vectors` reads the embeddings file and checks its sha256 and shape.
"""

import io
import logging
from pathlib import Path

import numpy as np

from kaful.artifacts.document_index import DocumentIndex, VectorsRef
from kaful.llm.embeddings import Vectors
from kaful.paths import resolve_in
from kaful.phases import PhaseName
from kaful.runs import ArtifactMismatch, RunContext, load_artifact, sha256_bytes

log = logging.getLogger("kaful.retrieval")

INGEST_PHASE: PhaseName = "ingest"
INDEX_KIND = "document_index"


class IndexMismatch(ArtifactMismatch):
    """The document index does not match the run manifest's index_hash."""


def check_index_hash(index_hash: str | None, expected: str | None, strict: bool) -> None:
    if index_hash == expected:
        return
    message = f"document index hash {(index_hash or 'none')[:12]} != manifest index_hash {(expected or 'none')[:12]}"
    if strict:
        raise IndexMismatch(message)
    log.warning("index hash mismatch", extra={"fields": {"detail": message}})


def load_index(ctx: RunContext) -> DocumentIndex:
    """The DocumentIndex recorded for this run's ingest phase, checked against the manifest."""
    refs = [r for r in ctx.manifest.phase(INGEST_PHASE).artifacts if r.kind == INDEX_KIND]
    if len(refs) != 1:
        raise FileNotFoundError(f"run {ctx.manifest.run_id} has no {INDEX_KIND} from {INGEST_PHASE}")
    index = load_artifact(ctx.machine_dir, refs[0], DocumentIndex)
    check_index_hash(index.index_hash, ctx.manifest.index_hash, ctx.manifest.strict)
    return index


def load_vectors(machine_dir: Path, ref: VectorsRef) -> Vectors:
    data = resolve_in(machine_dir, ref.path).read_bytes()
    if sha256_bytes(data) != ref.sha256:
        raise ArtifactMismatch(f"{ref.path}: content does not match the recorded sha256")
    vectors: Vectors = np.load(io.BytesIO(data)).astype(np.float32)
    if vectors.shape != (ref.n_rows, ref.dimensions):
        raise ArtifactMismatch(f"{ref.path}: shape {vectors.shape} != ({ref.n_rows}, {ref.dimensions})")
    return vectors
