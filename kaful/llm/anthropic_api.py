"""The Anthropic transport. This is the only kaful module that imports `anthropic`.

The client is created on first use (reading ANTHROPIC_API_KEY from .env then, without
overriding a variable already set), so building a gateway never needs a key or the
network. The SDK retries 408/409/429/5xx and connection errors itself
(`sdk_max_retries`); anything left is raised as TransportError.
"""

from collections.abc import Sequence
from typing import Any, cast

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from kaful.config import REPO_ROOT, LLMSettings
from kaful.llm.pricing import TokenUsage
from kaful.llm.transport import BatchItem, BatchStatus, Completion, TransientTransportError, TransportError

# Request fields count_tokens accepts (it has no max_tokens).
COUNT_TOKENS_FIELDS = ("model", "system", "messages", "output_config", "thinking", "tools", "tool_choice")


def output_format(model: type[BaseModel]) -> dict[str, Any]:
    """`output_config.format` for a Pydantic output model.

    The SDK's transform keeps what the API enforces and moves other constraints
    (minimum, maximum, pattern, ...) into descriptions; the gateway still validates
    the reply against the full model.
    """
    return {"type": "json_schema", "schema": anthropic.transform_schema(model)}


def usage_from_api(usage: Any) -> TokenUsage:
    """TokenUsage from an SDK `Usage` object (cache writes split by TTL)."""
    write_total = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    split = getattr(usage, "cache_creation", None)
    write_1h = int(getattr(split, "ephemeral_1h_input_tokens", 0) or 0) if split is not None else 0
    return TokenUsage(
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        cache_write_5m_tokens=write_total - write_1h,
        cache_write_1h_tokens=write_1h,
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


def completion_from_message(message: Any) -> Completion:
    text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
    return Completion(
        text=text,
        model=message.model,
        stop_reason=message.stop_reason,
        usage=usage_from_api(message.usage),
        request_id=getattr(message, "_request_id", None),
    )


# Errors worth retrying: the request may well succeed a little later.
TRANSIENT = (anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError,
             anthropic.RateLimitError)

class AnthropicTransport:
    """Messages, token counting and Message Batches through the official SDK."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._client: anthropic.Anthropic | None = None

    @property
    def can_count_tokens(self) -> bool:
        return True

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            load_dotenv(REPO_ROOT / ".env", override=False)
            self._client = anthropic.Anthropic(
                timeout=self._settings.request_timeout_s, max_retries=self._settings.sdk_max_retries
            )
        return self._client

    def send(self, request: dict[str, Any], timeout_s: float) -> Completion:
        try:
            message = self.client.messages.create(**request, timeout=timeout_s)
        except anthropic.APIError as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc
        return completion_from_message(message)

    def count_tokens(self, request: dict[str, Any]) -> int:
        fields = {k: request[k] for k in COUNT_TOKENS_FIELDS if request.get(k) is not None}
        try:
            return int(self.client.messages.count_tokens(**fields).input_tokens)
        except anthropic.APIError as exc:
            raise TransportError(f"count_tokens: {type(exc).__name__}: {exc}") from exc

    def submit_batch(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> str:
        payload = [{"custom_id": custom_id, "params": params} for custom_id, params in requests]
        try:
            batch = self.client.messages.batches.create(requests=cast(Any, payload))
        except anthropic.APIError as exc:
            raise TransportError(f"batch submit: {type(exc).__name__}: {exc}") from exc
        return str(batch.id)

    def batch_status(self, batch_id: str) -> BatchStatus:
        try:
            status: BatchStatus = self.client.messages.batches.retrieve(batch_id).processing_status
        except TRANSIENT as exc:
            raise TransientTransportError(f"batch status: {type(exc).__name__}: {exc}") from exc
        except anthropic.APIError as exc:
            raise TransportError(f"batch status: {type(exc).__name__}: {exc}") from exc
        return status

    def batch_results(self, batch_id: str) -> list[BatchItem]:
        items = []
        for result in self.client.messages.batches.results(batch_id):
            outcome = result.result
            if outcome.type == "succeeded":
                items.append(BatchItem(custom_id=result.custom_id, outcome="succeeded",
                                       completion=completion_from_message(outcome.message)))
            else:
                error = str(getattr(outcome, "error", "")) or None
                items.append(BatchItem(custom_id=result.custom_id, outcome=outcome.type, error=error))
        return sorted(items, key=lambda item: item.custom_id)
