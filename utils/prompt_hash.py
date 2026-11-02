"""
utils/prompt_hash.py
--------------------
The single definition of a prompt's identity.

The run ledger records it as `prompt_sha256`, and M2's fake LLM gateway keys
recorded fixtures by the same value. Both must call `prompt_hash()` from here;
do not hash requests anywhere else, or ledger rows and fixtures stop lining up.
(M5 ports this function into kaful/llm/ unchanged, since kaful/ does not import
legacy code. Any change to the canonical form invalidates recorded fixtures.)

Canonical form: the request fields that determine the model's output, as
compact JSON with sorted keys, UTF-8 encoded, then SHA-256.
"""

import hashlib
import json
from typing import Any

# Request fields that determine output. Anything else (timeouts, retries,
# client options) is excluded so it never changes a prompt's identity.
HASHED_FIELDS = ("model", "system", "messages", "max_tokens", "temperature")


def canonical_request(request: dict[str, Any]) -> str:
    """Return the canonical JSON string for a Messages API request."""
    subset = {k: request.get(k) for k in HASHED_FIELDS}
    return json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_hash(request: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical request."""
    return hashlib.sha256(canonical_request(request).encode("utf-8")).hexdigest()
