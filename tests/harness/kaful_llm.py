"""Offline transports and helpers for kaful.llm tests. Nothing here touches the network.

ScriptedTransport answers each request with `responder(request)`: a reply text (or a
Completion, or an exception to raise). Its batch methods hold submitted requests and
return their results in a seeded random order, so tests prove results are matched by
custom_id, never by position.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kaful.config import LLMSettings
from kaful.llm.budget import RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.ledger import Ledger
from kaful.llm.pricing import TokenUsage
from kaful.llm.transport import BatchItem, BatchStatus, Completion

Reply = str | Completion | Exception
FIXED_NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
DEFAULT_USAGE = TokenUsage(input_tokens=1200, output_tokens=300)


def last_user_text(request: dict[str, Any]) -> str:
    content = request["messages"][-1]["content"]
    return content if isinstance(content, str) else content[-1]["text"]


@dataclass
class ScriptedTransport:
    responder: Callable[[dict[str, Any]], Reply]
    usage: TokenUsage = DEFAULT_USAGE
    polls_before_end: int = 0
    seed: int = 0
    batch_errors: set[str] = field(default_factory=set)  # each custom_id errors once
    sent: list[dict[str, Any]] = field(default_factory=list)
    timeouts: list[float] = field(default_factory=list)
    batches: dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    polls: int = 0

    @property
    def can_count_tokens(self) -> bool:
        return False

    def count_tokens(self, request: dict[str, Any]) -> int:
        raise AssertionError("ScriptedTransport cannot count tokens")

    def _complete(self, request: dict[str, Any]) -> Completion:
        reply = self.responder(request)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, Completion):
            return reply
        return Completion(text=reply, model=request["model"], stop_reason="end_turn", usage=self.usage)

    def send(self, request: dict[str, Any], timeout_s: float) -> Completion:
        self.sent.append(request)
        self.timeouts.append(timeout_s)
        return self._complete(request)

    def submit_batch(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> str:
        batch_id = f"msgbatch_{len(self.batches) + 1}"
        self.batches[batch_id] = list(requests)
        return batch_id

    def batch_status(self, batch_id: str) -> BatchStatus:
        self.polls += 1
        return "ended" if self.polls > self.polls_before_end else "in_progress"

    def batch_results(self, batch_id: str) -> list[BatchItem]:
        items = []
        for custom_id, request in self.batches[batch_id]:
            if custom_id in self.batch_errors:
                self.batch_errors.discard(custom_id)
                items.append(BatchItem(custom_id=custom_id, outcome="errored", error="overloaded_error"))
            else:
                self.sent.append(request)
                items.append(BatchItem(custom_id=custom_id, outcome="succeeded", completion=self._complete(request)))
        random.Random(self.seed).shuffle(items)
        return items


def make_gateway(tmp: Path, transport: Any, budget_usd: float = 1.0, *, cache: bool = True,
                 settings: LLMSettings | None = None) -> Gateway:
    """A gateway writing its ledger, cache and batch state under `tmp`."""
    clock = iter(range(10**9))
    return Gateway(
        settings=settings or LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5"),
        ledger=Ledger(tmp / "run" / "ledger.jsonl"),
        budget=RunBudget(budget_usd),
        transport=transport,
        cache=ResponseStore(tmp / "llm_cache") if cache else None,
        batch_state_dir=tmp / "llm_batches",
        run_id="20261103T120000Z-abcdef",
        machine_id="toy_machine",
        sleep=lambda _s: None,
        clock=lambda: float(next(clock)),
        now=lambda: FIXED_NOW,
    )
