"""The Anthropic transport with a fake SDK client (no network): long generations are streamed,
and every call has a wall-clock limit of its own (seen live 2026-11-07: a call hung for over
two hours although the SDK timeout was 120 s)."""

import threading
import types
from typing import Any

import pytest

from kaful.config import LLMSettings
from kaful.llm.anthropic_api import AnthropicTransport
from kaful.llm.transport import TransportError

MESSAGE = types.SimpleNamespace(
    content=[types.SimpleNamespace(type="text", text="{}")], model="claude-haiku-5-5", stop_reason="end_turn",
    usage=types.SimpleNamespace(input_tokens=10, output_tokens=2, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0))


class _Stream:
    def __enter__(self) -> "_Stream":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def get_final_message(self) -> Any:
        return MESSAGE


class _Messages:
    def __init__(self, block: threading.Event | None = None) -> None:
        self.used: list[str] = []
        self.block = block

    def create(self, **request: Any) -> Any:
        self.used.append("create")
        if self.block is not None:
            self.block.wait(5)
        return MESSAGE

    def stream(self, **request: Any) -> _Stream:
        self.used.append("stream")
        return _Stream()


def _transport(messages: _Messages, **settings: Any) -> AnthropicTransport:
    transport = AnthropicTransport(LLMSettings(**settings))
    transport._client = types.SimpleNamespace(messages=messages, close=lambda: None)  # type: ignore[assignment]
    return transport


def test_long_generations_are_streamed_and_short_ones_are_not() -> None:
    messages = _Messages()
    transport = _transport(messages, stream_min_max_tokens=8000)
    assert transport.send({"model": "m", "max_tokens": 16000, "messages": []}, 60.0).text == "{}"
    transport.send({"model": "m", "max_tokens": 4000, "messages": []}, 60.0)
    assert messages.used == ["stream", "create"]


def test_a_call_past_its_wall_clock_limit_raises() -> None:
    release = threading.Event()
    transport = _transport(_Messages(block=release), call_deadline_s=0.2)
    with pytest.raises(TransportError, match="wall-clock limit of 0.2 s"):
        transport.send({"model": "m", "max_tokens": 100, "messages": []}, 60.0)
    release.set()
