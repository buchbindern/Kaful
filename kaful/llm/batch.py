"""Message Batches bookkeeping: custom ids, resumable batch state, polling.

A submitted batch is recorded in `<state_dir>/<key>.json`, where the key hashes the
sorted custom ids. A rerun that needs the same requests (e.g. `kaful run --from P`
after the previous run stopped while waiting) finds the record and collects that
batch instead of submitting and paying again. The record is deleted once collected.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from kaful.artifacts.base import NonEmptyStr, StrictModel
from kaful.llm.transport import TransientTransportError, Transport

log = logging.getLogger("kaful.llm.batch")

CUSTOM_ID_HASH_CHARS = 40
SAMPLE_DIGITS = 4


class BatchPending(RuntimeError):
    """A batch did not finish within the wait limit; rerun to collect it."""

    def __init__(self, batch_id: str, waited_s: float) -> None:
        super().__init__(f"batch {batch_id} still running after {waited_s:.0f}s; rerun to collect it")
        self.batch_id = batch_id


def custom_id(phash: str, sample: int) -> str:
    """Matches the API's ^[a-zA-Z0-9_-]{1,64}$ and sorts by (hash, sample)."""
    return f"{phash[:CUSTOM_ID_HASH_CHARS]}_{sample:0{SAMPLE_DIGITS}d}"


class BatchState(StrictModel):
    batch_id: NonEmptyStr
    custom_ids: tuple[str, ...]
    submitted_at: NonEmptyStr


def state_key(custom_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(custom_ids)).encode()).hexdigest()


def find_state(state_dir: Path, custom_ids: Sequence[str]) -> BatchState | None:
    path = state_dir / f"{state_key(custom_ids)}.json"
    return BatchState.model_validate_json(path.read_bytes()) if path.is_file() else None


def saved_states(state_dir: Path) -> list[BatchState]:
    """Every batch submitted but not yet collected, in key order."""
    if not state_dir.is_dir():
        return []
    return [BatchState.model_validate_json(p.read_bytes()) for p in sorted(state_dir.glob("*.json"))]


def save_state(state_dir: Path, state: BatchState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{state_key(state.custom_ids)}.json"
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clear_state(state_dir: Path, custom_ids: Sequence[str]) -> None:
    (state_dir / f"{state_key(custom_ids)}.json").unlink(missing_ok=True)


def wait_for_batch(
    transport: Transport, batch_id: str, poll_s: float, max_wait_s: float,
    sleep: Callable[[float], None], clock: Callable[[], float], max_errors: int,
) -> None:
    """Poll until the batch has ended; raise BatchPending after `max_wait_s`, or after
    more than `max_errors` transient errors in a row (the batch stays saved, so a rerun
    collects it without paying twice). A transient error waits one poll interval."""
    started, errors = clock(), 0
    while True:
        try:
            if transport.batch_status(batch_id) == "ended":
                return
            errors = 0
        except TransientTransportError as exc:
            errors += 1
            log.warning("batch status poll failed; retrying", extra={"fields": {
                "batch_id": batch_id, "error": str(exc), "in_a_row": errors, "max": max_errors}})
            if errors > max_errors:
                raise BatchPending(batch_id, clock() - started) from exc
        waited = clock() - started
        if waited >= max_wait_s:
            raise BatchPending(batch_id, waited)
        sleep(min(poll_s, max_wait_s - waited))
