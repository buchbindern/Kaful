"""
Shared test setup: every test runs with the network blocked and no real API keys.

`load_dotenv()` in run.py and phase 1 does not override variables that are
already set, so placeholder keys keep the real ones in .env out of the process.
Tests marked `live` are exempt (and never run without approval).
"""

from collections.abc import Iterator

import pytest

from tests.harness.netblock import blocked_network

OFFLINE_ENV = {
    "ANTHROPIC_API_KEY": "offline-test-key",
    "OPENAI_API_KEY": "offline-test-key",
    "ANONYMIZED_TELEMETRY": "False",
}


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker("live"):
        yield
        return
    for name, value in OFFLINE_ENV.items():
        monkeypatch.setenv(name, value)
    with blocked_network():
        yield
