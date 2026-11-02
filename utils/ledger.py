"""
utils/ledger.py
---------------
Minimal run ledger for the legacy pipeline (M1). One JSON line per event in
machines/<id>/outputs/ledger.jsonl.

Events:
    llm_call        one call_claude attempt
    pdf_extract     one phase-1 PDF extraction attempt
    embedding       one OpenAI embedding request (tokens estimated from chars)
    checkpoint_hit  an LLM step skipped because its output already exists

Phase and step are inferred from the call stack: the nearest frame whose module
is phases.phaseN_*[.step_*]. Per-component steps set the component explicitly
with `ledger.set_component(name)`.

The ledger must never break a run: write failures are logged and swallowed.
M5 replaces this with the kaful/llm gateway ledger.

Usage:
    from utils import ledger
    ledger.start_run(cfg)
"""

import contextvars
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.pricing import cost_usd, pricing_as_of

SCHEMA_VERSION = 1
# OpenAI's rule of thumb for English text; embeddings are logged as estimates.
EMBEDDING_CHARS_PER_TOKEN = 4.0
LEDGER_FILENAME = "ledger.jsonl"

logger = logging.getLogger(__name__)

_PHASE_MODULE = re.compile(r"^phases\.(phase\d+)_\w+?(?:\.(step_\w+))?$")


@dataclass
class _Run:
    run_id: str
    machine_id: str
    path: Path


_run: _Run | None = None
_component: contextvars.ContextVar[str | None] = contextvars.ContextVar("component", default=None)


# ── Run scope ─────────────────────────────────────────────────────────────────

def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


def start_run(cfg: dict, run_id: str | None = None) -> str:
    """Point the ledger at this machine. Returns the run id."""
    global _run
    path = Path(cfg["machine_dir"]) / "outputs" / LEDGER_FILENAME
    _run = _Run(run_id or new_run_id(), cfg["machine_id"], path)
    logger.info("ledger run %s → %s", _run.run_id, path)
    return _run.run_id


def ensure_run(cfg: dict) -> str:
    """Start a run unless one is already active (for standalone phase scripts)."""
    if _run is None or _run.machine_id != cfg["machine_id"]:
        return start_run(cfg)
    return _run.run_id


def end_run() -> None:
    global _run
    _run = None


def set_component(name: str | None) -> None:
    """Attribute the following events to one component (None clears it).

    Set at the top of a per-component loop body and cleared after the loop,
    so the legacy loops need no re-indentation.
    """
    _component.set(name)


def current_component() -> str | None:
    return _component.get()


# ── Attribution ───────────────────────────────────────────────────────────────

def caller_scope() -> dict[str, Any]:
    """Phase, step and call site from the nearest phases.* frame."""
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        m = _PHASE_MODULE.match(module)
        if m:
            phase, step = m.group(1), m.group(2) or module.rsplit("_", 1)[-1]
            return {
                "phase": phase,
                "step": step,
                "call_site": f"{module}:{frame.f_code.co_name}:{frame.f_lineno}",
            }
        frame = frame.f_back
    return {"phase": "unknown", "step": "unknown", "call_site": None}


# ── Recording ─────────────────────────────────────────────────────────────────

def _write(record: dict[str, Any]) -> None:
    if _run is None:
        logger.warning("ledger: no active run; event not written: %s", record)
        return
    try:
        _run.path.parent.mkdir(parents=True, exist_ok=True)
        with open(_run.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # the ledger must never break a run
        logger.exception("ledger write failed")


def _base(event: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": _run.run_id if _run else None,
        "machine_id": _run.machine_id if _run else None,
        "event": event,
        **caller_scope(),
        "component": current_component(),
    }


def usage_tokens(usage: Any) -> dict[str, int]:
    """Token fields from an Anthropic `response.usage` object."""
    def get(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    cache_write = get("cache_creation_input_tokens")
    split = getattr(usage, "cache_creation", None)
    write_1h = int(getattr(split, "ephemeral_1h_input_tokens", 0) or 0) if split else 0
    return {
        "input_tokens": get("input_tokens"),
        "output_tokens": get("output_tokens"),
        "cache_creation_input_tokens": cache_write,
        "cache_write_1h_tokens": write_1h,
        "cache_read_input_tokens": get("cache_read_input_tokens"),
    }


def record_llm_call(
    *,
    event: str,
    request: dict[str, Any],
    prompt_sha256: str,
    started: float,
    status: str,
    attempt: int,
    usage: Any = None,
    response_chars: int | None = None,
    error: str | None = None,
) -> None:
    """Record one Anthropic Messages attempt (call_claude or PDF extraction)."""
    try:
        tokens = usage_tokens(usage) if usage is not None else usage_tokens(None)
        model = request["model"]
        write_1h = tokens.pop("cache_write_1h_tokens")
        cost = cost_usd(
            model,
            input_tokens=tokens["input_tokens"],
            output_tokens=tokens["output_tokens"],
            cache_write_5m_tokens=tokens["cache_creation_input_tokens"] - write_1h,
            cache_write_1h_tokens=write_1h,
            cache_read_tokens=tokens["cache_read_input_tokens"],
        )
        _write({
            **_base(event),
            "provider": "anthropic",
            "model": model,
            "attempt": attempt,
            "status": status,
            **tokens,
            "tokens_estimated": False,
            "cache_hit": tokens["cache_read_input_tokens"] > 0,
            "cost_usd": cost,
            "pricing_as_of": pricing_as_of(),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "max_tokens": request.get("max_tokens"),
            "temperature": request.get("temperature"),
            "prompt_sha256": prompt_sha256,
            "request_chars": _request_chars(request),
            "response_chars": response_chars,
            "error": error,
        })
    except Exception:
        logger.exception("ledger: failed to record llm call")


def record_embedding(*, model: str, texts: list[str], started: float, purpose: str) -> None:
    """Record one embedding request. Chroma drops OpenAI's usage, so tokens are estimated."""
    try:
        chars = sum(len(t) for t in texts)
        tokens = round(chars / EMBEDDING_CHARS_PER_TOKEN)
        _write({
            **_base("embedding"),
            "provider": "openai",
            "model": model,
            "purpose": purpose,
            "n_texts": len(texts),
            "request_chars": chars,
            "input_tokens": tokens,
            "output_tokens": 0,
            "tokens_estimated": True,
            "cost_usd": cost_usd(model, input_tokens=tokens),
            "pricing_as_of": pricing_as_of(),
            "latency_ms": round((time.monotonic() - started) * 1000),
        })
    except Exception:
        logger.exception("ledger: failed to record embedding")


def checkpoint_hit(path: Path | str) -> None:
    """Record that an LLM step was skipped because its output exists on disk."""
    try:
        _write({**_base("checkpoint_hit"), "path": Path(path).name, "cost_usd": 0.0})
    except Exception:
        logger.exception("ledger: failed to record checkpoint hit")


def _request_chars(request: dict[str, Any]) -> int:
    """Characters of text in a request; base64 document payloads are excluded."""
    total = len(request.get("system") or "")
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
        else:
            total += sum(len(b.get("text", "")) for b in content if b.get("type") == "text")
    return total
