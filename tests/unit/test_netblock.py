"""The autouse network block refuses IP connections and keeps real keys out. Offline."""

import os
import socket

import pytest

from tests.harness.netblock import NetworkBlockedError


def test_tcp_connect_is_refused() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(NetworkBlockedError):
            s.connect(("93.184.216.34", 443))


def test_create_connection_and_dns_are_refused() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("api.anthropic.com", 443))
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("api.anthropic.com", 443)


def test_unix_sockets_still_work() -> None:
    a, b = socket.socketpair(socket.AF_UNIX)
    with a, b:
        a.sendall(b"ok")
        assert b.recv(2) == b"ok"


def test_api_keys_are_placeholders() -> None:
    assert os.environ["ANTHROPIC_API_KEY"] == "offline-test-key"
    assert os.environ["OPENAI_API_KEY"] == "offline-test-key"


def test_anthropic_client_cannot_reach_the_api() -> None:
    import anthropic

    client = anthropic.Anthropic(max_retries=0)
    with pytest.raises(anthropic.APIConnectionError):
        client.messages.create(model="claude-sonnet-4-6", max_tokens=1,
                               messages=[{"role": "user", "content": "hi"}])
