"""Stored count_tokens results (M15), so a cache-only run decides exactly as the online run did.

A decision that depends on a text's API token count (understand: does the manual go in whole?) must not
change when no API client exists. The gateway stores every count_tokens result here, and an offline
gateway reads it back; a count that was never stored is an error offline, never an estimate.

One file per counted request, keyed by the sha256 of the fields count_tokens accepts (canonical JSON):

    <root>/<model>/<request_sha256>.json
    {"schema_version": 1, "model": ..., "request_sha256": ..., "input_tokens": ...}

A count is a pure function of the request and the model's tokenizer, so nothing expires.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from kaful.artifacts.base import NonEmptyStr, StrictModel

# Request fields count_tokens accepts (no max_tokens, temperature or metadata).
COUNTED_FIELDS = ("model", "system", "messages", "output_config", "thinking", "tools", "tool_choice")


class StoredCount(StrictModel):
    schema_version: Literal[1] = 1
    model: NonEmptyStr
    request_sha256: NonEmptyStr
    input_tokens: int = Field(ge=0)


def count_request_sha256(request: dict[str, Any]) -> str:
    """The key of a count request: its count_tokens fields as canonical JSON."""
    fields = {k: request[k] for k in COUNTED_FIELDS if request.get(k) is not None}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class TokenCountStore:
    """Read/write access to one directory of stored token counts."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, model: str, request_sha256: str) -> Path:
        return self.root / model / f"{request_sha256}.json"

    def get(self, request: dict[str, Any]) -> int | None:
        path = self.path(str(request["model"]), count_request_sha256(request))
        if not path.is_file():
            return None
        return StoredCount.model_validate_json(path.read_bytes()).input_tokens

    def put(self, request: dict[str, Any], input_tokens: int) -> Path:
        key = count_request_sha256(request)
        stored = StoredCount(model=str(request["model"]), request_sha256=key, input_tokens=input_tokens)
        path = self.path(stored.model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(stored.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path
