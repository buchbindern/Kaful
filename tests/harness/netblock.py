"""
tests/harness/netblock.py
-------------------------
Blocks outbound network for the test process (CLAUDE.md hard rule 1).

Only IP sockets are refused; AF_UNIX stays available because local tools
(multiprocessing, some sqlite/chroma internals) use it. Subprocesses are not
covered; tests strip real API keys from the environment for those.
"""

import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class NetworkBlockedError(RuntimeError):
    """Raised when test code tries to open a network connection."""


def _refuse(*args: Any, **_kwargs: Any) -> Any:
    target = args[1] if len(args) > 1 else args[0] if args else "?"
    raise NetworkBlockedError(f"network access is blocked in tests (target: {target!r})")


def _guarded_connect(original: Any) -> Any:
    def connect(self: socket.socket, address: Any) -> Any:
        if self.family == socket.AF_UNIX:
            return original(self, address)
        return _refuse(self, address)
    return connect


@contextmanager
def blocked_network() -> Iterator[None]:
    """Refuse IP connections and DNS lookups for the duration of the block."""
    saved = (socket.socket.connect, socket.socket.connect_ex,
             socket.create_connection, socket.getaddrinfo)
    socket.socket.connect = _guarded_connect(saved[0])  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect(saved[1])  # type: ignore[method-assign]
    socket.create_connection = _refuse  # type: ignore[assignment]
    socket.getaddrinfo = _refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        (socket.socket.connect, socket.socket.connect_ex,  # type: ignore[method-assign]
         socket.create_connection, socket.getaddrinfo) = saved
