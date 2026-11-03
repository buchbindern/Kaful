"""The identity of a Messages API request: `prompt_hash`.

Ported from legacy `utils/prompt_hash.py` (M1), which the legacy ledger records as
`prompt_sha256` and M2's recorded fixtures are keyed by. The canonical form is
unchanged for every request the legacy code sent: the fields in `HASHED_FIELDS`,
as compact JSON with sorted keys, UTF-8, then SHA-256.

Two extensions, neither of which changes a legacy hash:
- fields in `OPTIONAL_HASHED_FIELDS` determine output too, but legacy never sent
  them, so they are hashed only when present;
- `cache_control` markers are removed first: caching changes cost, not output, so
  changing a cache TTL must not invalidate cached responses or fixtures.
"""

import hashlib
import json
from typing import Any

HASHED_FIELDS = ("model", "system", "messages", "max_tokens", "temperature")
OPTIONAL_HASHED_FIELDS = ("output_config", "stop_sequences", "thinking", "tool_choice", "tools")
CACHE_CONTROL_KEY = "cache_control"


def strip_cache_control(value: Any) -> Any:
    """A copy of `value` with every `cache_control` key removed, at any depth."""
    if isinstance(value, dict):
        return {k: strip_cache_control(v) for k, v in value.items() if k != CACHE_CONTROL_KEY}
    if isinstance(value, list):
        return [strip_cache_control(v) for v in value]
    return value


def hashed_subset(request: dict[str, Any]) -> dict[str, Any]:
    """The request fields that determine the model's output."""
    subset = {k: request.get(k) for k in HASHED_FIELDS}
    subset.update({k: request[k] for k in OPTIONAL_HASHED_FIELDS if request.get(k) is not None})
    stripped: dict[str, Any] = strip_cache_control(subset)
    return stripped


def canonical_request(request: dict[str, Any]) -> str:
    """The canonical JSON string for a Messages API request."""
    return json.dumps(hashed_subset(request), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_hash(request: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical request."""
    return hashlib.sha256(canonical_request(request).encode("utf-8")).hexdigest()
