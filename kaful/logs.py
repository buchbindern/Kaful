"""Structured logging: JSON lines with the current run id and phase.

Pipeline code logs through `logging.getLogger("kaful...")` and never prints.
Pass structured data with `extra={"fields": {...}}`.
"""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

ROOT_LOGGER = "kaful"

_run_id: ContextVar[str | None] = ContextVar("kaful_run_id", default=None)
_phase: ContextVar[str | None] = ContextVar("kaful_phase", default=None)


class JsonFormatter(logging.Formatter):
    """One JSON object per line, keys sorted."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "run_id": _run_id.get(),
            "phase": _phase.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            entry["fields"] = fields
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, sort_keys=True, default=str)


def configure_logging(level: str, log_file: Path | None = None) -> None:
    """Send kaful logs as JSON to stderr and, if given, to `log_file`. Idempotent."""
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def log_context(run_id: str | None = None, phase: str | None = None) -> Iterator[None]:
    """Attach run_id and/or phase to every log record inside the block."""
    tokens = []
    if run_id is not None:
        tokens.append((_run_id, _run_id.set(run_id)))
    if phase is not None:
        tokens.append((_phase, _phase.set(phase)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
