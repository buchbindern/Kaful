"""The boundary between the gateway and a model provider.

The gateway talks only to a `Transport`. `kaful.llm.anthropic_api.AnthropicTransport`
is the real one (and the only module that imports `anthropic`); tests use replaying
and recording transports from tests/harness, so no test touches the network.
"""

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from kaful.artifacts.base import StrictModel
from kaful.llm.pricing import TokenUsage

BatchStatus = Literal["in_progress", "canceling", "ended"]
BatchOutcome = Literal["succeeded", "errored", "canceled", "expired"]


class Completion(StrictModel):
    """One model response, provider-neutral.

    `billed` is False for responses that cost nothing now (a replayed fixture); their
    `usage` is what the original call used.
    """

    text: str
    model: str
    stop_reason: str | None
    usage: TokenUsage
    billed: bool = True
    request_id: str | None = None


class BatchItem(StrictModel):
    """One result of a batch, matched to its request by `custom_id` only."""

    custom_id: str
    outcome: BatchOutcome
    completion: Completion | None = None
    error: str | None = None


class TransportError(RuntimeError):
    """The provider rejected or failed a request (after the SDK's own retries)."""


class Transport(Protocol):
    """What the gateway needs from a provider."""

    @property
    def can_count_tokens(self) -> bool:
        """True if `count_tokens` works (it needs the network)."""
        ...

    def send(self, request: dict[str, Any]) -> Completion: ...

    def count_tokens(self, request: dict[str, Any]) -> int: ...

    def submit_batch(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> str:
        """Submit (custom_id, request) pairs; return the batch id."""
        ...

    def batch_status(self, batch_id: str) -> BatchStatus: ...

    def batch_results(self, batch_id: str) -> list[BatchItem]: ...
