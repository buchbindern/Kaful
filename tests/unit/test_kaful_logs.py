"""Structured logging: JSON lines carry run_id and phase."""

import json
import logging
from pathlib import Path

from kaful.logs import configure_logging, log_context


def test_log_lines_are_json_with_run_and_phase(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log.jsonl"
    configure_logging("INFO", log_file)
    logger = logging.getLogger("kaful.test")
    with log_context(run_id="r1"):
        with log_context(phase="scope"):
            logger.info("decided", extra={"fields": {"in_scope": 2}})
        logger.warning("after phase")
    logger.debug("hidden")
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [(e["msg"], e["run_id"], e["phase"]) for e in lines] == [
        ("decided", "r1", "scope"), ("after phase", "r1", None)]
    assert lines[0]["fields"] == {"in_scope": 2} and lines[0]["level"] == "INFO"


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    configure_logging("INFO", tmp_path / "a.jsonl")
    configure_logging("INFO", tmp_path / "b.jsonl")
    assert len(logging.getLogger("kaful").handlers) == 2
