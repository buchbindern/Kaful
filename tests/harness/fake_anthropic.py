"""
tests/harness/fake_anthropic.py
-------------------------------
Stand-ins for `anthropic.Anthropic` used by the legacy pipeline
(`utils.llm.call_claude` and phase 1's `ManualRAG._extract_rich_content`).

    ReplayAnthropic     answers from a FixtureStore; never touches the network.
    RecordingAnthropic  forwards to the real client, saves each response verbatim,
                        and refuses any call whose worst-case cost would take the
                        ledger total over the budget.

Replayed responses report zero usage, so a replayed run's ledger costs $0.
"""

import base64
import io
import json
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tests.harness.fixtures import FixtureStore
from utils import ledger
from utils.pricing import cost_usd

# Worst-case input estimate for the budget check (real tokenizers average ~4 chars/token).
CHARS_PER_TOKEN_WORST = 2.5
# Anthropic documents 1,500–3,000 tokens per PDF page; M1 measured ~2,000.
PDF_TOKENS_PER_PAGE_WORST = 3_000


class BudgetExceededError(RuntimeError):
    """The next call could take the recording run over its budget."""


def _scope() -> dict[str, Any]:
    return {**ledger.caller_scope(), "component": ledger.current_component()}


def _usage(tokens: dict[str, int] | None = None) -> Any:
    t = tokens or {}
    return types.SimpleNamespace(
        input_tokens=t.get("input_tokens", 0), output_tokens=t.get("output_tokens", 0),
        cache_creation_input_tokens=t.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=t.get("cache_read_input_tokens", 0), cache_creation=None,
    )


def _message(response: dict[str, Any]) -> Any:
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text=response["text"])],
        model=response.get("model"), stop_reason=response.get("stop_reason"), usage=_usage(),
    )


class _Messages:
    def __init__(self, create: Any) -> None:
        self.create = create


class ReplayAnthropic:
    """Drop-in for `anthropic.Anthropic(...)` that replays fixtures."""

    def __init__(self, store: FixtureStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self._create)

    def __call__(self, *_args: Any, **_kwargs: Any) -> "ReplayAnthropic":
        return self  # used as the patched class: Anthropic(api_key=...) → this client

    def _create(self, **request: Any) -> Any:
        scope = _scope()
        self.calls.append({**scope, "model": request.get("model")})
        return _message(self.store.next_response(request, scope))


class RecordingAnthropic:
    """Drop-in for `anthropic.Anthropic(...)` that records real responses."""

    def __init__(self, real: Any, store: FixtureStore, budget_usd: float,
                 ledger_path: Path) -> None:
        self.real = real
        self.store = store
        self.budget_usd = budget_usd
        self.ledger_path = ledger_path
        self.messages = _Messages(self._create)

    def __call__(self, *_args: Any, **_kwargs: Any) -> "RecordingAnthropic":
        return self

    def _create(self, **request: Any) -> Any:
        spent = ledger_spent_usd(self.ledger_path)
        worst = worst_case_cost_usd(request)
        if spent + worst > self.budget_usd:
            raise BudgetExceededError(
                f"stopping: ledger total ${spent:.4f} + worst case ${worst:.4f} for "
                f"{_scope()['step']} exceeds the ${self.budget_usd:.2f} budget")
        response = self.real.messages.create(**request)
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        self.store.append(request, _scope(), {
            "text": text,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "usage": ledger.usage_tokens(response.usage),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, source="recorded")
        return response


def ledger_spent_usd(path: Path) -> float:
    """Sum of `cost_usd` over every ledger row so far."""
    if not path.exists():
        return 0.0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sum(r.get("cost_usd") or 0.0 for r in rows)


def worst_case_cost_usd(request: dict[str, Any]) -> float:
    """Upper bound on one call's cost: generous input estimate plus full max_tokens."""
    chars = len(request.get("system") or "")
    pages = 0
    for msg in request.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content:
            if block.get("type") == "text":
                chars += len(block["text"])
            elif block.get("type") == "document":
                pages += _pdf_pages(block["source"]["data"])
    input_tokens = int(chars / CHARS_PER_TOKEN_WORST) + pages * PDF_TOKENS_PER_PAGE_WORST
    cost = cost_usd(request["model"], input_tokens=input_tokens,
                    output_tokens=int(request.get("max_tokens") or 0))
    if cost is None:
        raise BudgetExceededError(f"no price for model {request['model']!r}; refusing to call")
    return cost


def _pdf_pages(data_b64: str) -> int:
    from pypdf import PdfReader
    return len(PdfReader(io.BytesIO(base64.b64decode(data_b64))).pages)
