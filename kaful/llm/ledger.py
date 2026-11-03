"""The run ledger: one JSON line per API event, in one schema shared with M1.

`LedgerRow` reads both versions:
- v1 rows, written by the legacy `utils/ledger.py` (M1) to
  machines/<id>/outputs/ledger.jsonl;
- v2 rows, written by the kaful gateway to outputs/runs/<run_id>/ledger.jsonl.

v2 keeps every v1 field name and meaning and only adds optional fields. Notably:
`cache_hit` still means "the call read from the prompt cache" (as in M1), and the
new `response_cache` says whether the gateway's own response cache answered the call.
For kaful rows, `step` holds the subphase name, and `attempt` counts validation
attempts (transport retries happen inside the SDK).

`cost_usd` is what was billed: 0 for a response-cache hit or a fixture replay, in
which case `unbilled_cost_usd` holds what the original response cost.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from kaful.artifacts.base import StrictModel

LEDGER_FILE = "ledger.jsonl"
LEDGER_SCHEMA_VERSION = 2

LedgerEvent = Literal["llm_call", "pdf_extract", "embedding", "checkpoint_hit", "llm_batch_submit"]
LedgerStatus = Literal["ok", "retry", "error", "refused_budget", "submitted"]
ResponseCacheUse = Literal["hit", "miss", "bypass", "replay"]
Validation = Literal["valid", "invalid", "not_checked"]


class LedgerRow(StrictModel):
    """One ledger line (v1 or v2)."""

    schema_version: Literal[1, 2]
    ts: str
    run_id: str | None
    machine_id: str | None
    event: LedgerEvent
    phase: str
    step: str
    call_site: str | None = None
    component: str | None = None
    provider: str | None = None
    model: str | None = None
    attempt: int | None = Field(default=None, ge=1)
    status: LedgerStatus | None = None
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    tokens_estimated: bool = False
    cache_hit: bool = False
    cost_usd: float | None = Field(default=None, ge=0.0)
    pricing_as_of: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    max_tokens: int | None = None
    temperature: float | None = None
    prompt_sha256: str | None = None
    request_chars: int | None = None
    response_chars: int | None = None
    error: str | None = None
    # v1, event-specific
    purpose: str | None = None
    n_texts: int | None = None
    path: str | None = None
    # v2 additions
    prompt_id: str | None = None
    prompt_version: int | None = None
    sample: int | None = Field(default=None, ge=0)
    cache_creation_1h_tokens: int = Field(default=0, ge=0)
    response_cache: ResponseCacheUse | None = None
    validation: Validation | None = None
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    unbilled_cost_usd: float | None = Field(default=None, ge=0.0)
    batch_id: str | None = None
    custom_id: str | None = None
    output_artifact: str | None = None

    @model_validator(mode="after")
    def _v2_calls_name_their_prompt(self) -> "LedgerRow":
        if self.schema_version == 2 and self.event == "llm_call":
            missing = [f for f in ("prompt_id", "prompt_version", "response_cache", "status")
                       if getattr(self, f) is None]
            if missing:
                raise ValueError(f"v2 llm_call row is missing {missing}")
        return self


class Ledger:
    """Append-only JSONL ledger at `path`. Write failures raise: the budget depends on it."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, row: LedgerRow) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row.model_dump(mode="json", exclude_none=True), sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def rows(self) -> list[LedgerRow]:
        return read_ledger(self.path)


def read_ledger(path: Path) -> list[LedgerRow]:
    """Every row of a ledger file (v1 or v2); a missing file is an empty ledger."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LedgerRow.model_validate_json(line) for line in lines if line.strip()]


def total_cost(rows: Iterable[LedgerRow]) -> float:
    return sum(r.cost_usd or 0.0 for r in rows)


def cost_by_phase(rows: Iterable[LedgerRow]) -> dict[str, float]:
    """Billed cost per phase, keys sorted."""
    totals: dict[str, float] = {}
    for row in rows:
        totals[row.phase] = totals.get(row.phase, 0.0) + (row.cost_usd or 0.0)
    return dict(sorted(totals.items()))
