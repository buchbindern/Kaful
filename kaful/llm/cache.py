"""Content-addressed response store: the gateway's response cache and recorded fixtures.

One file per distinct request, in M2's fixture format plus `prompt_id` and
`prompt_version`:

    <root>/<prompt_id>/v<version>/<prompt_sha256>.json
    {"prompt_sha256": ..., "prompt_id": ..., "prompt_version": ..., "phase": ...,
     "step": ..., "component": ..., "source": "recorded" | "batch" | "hand_authored",
     "request": {...hashed fields, base64 payloads replaced by their sha256...},
     "responses": [{"text", "model", "stop_reason", "usage", "recorded_at"}, ...]}

`responses[i]` is sample i of the request (repeated identical requests, e.g. N
self-consistency extractions, are samples 0..N-1). A validation retry's request carries its
sample's own reply, so it is unique to that sample and is stored as response 0 of its own
file. A response served from an older prompt version (a ReuseRule) is stored under the new
request too, with `source = "reused"` and the older version and hash on the response. The key is the request's content,
so any change to template text, inputs, model, max_tokens, effort or output schema
is a miss; nothing expires.

The same class serves two roles: a run's response cache (read and written by the
gateway) and a directory of recorded fixtures (read by the replay transport in tests).
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from kaful.artifacts.base import NonEmptyStr, StrictModel
from kaful.llm.hashing import hashed_subset
from kaful.llm.pricing import TokenUsage
from kaful.llm.transport import Completion

RecordSource = Literal["recorded", "batch", "hand_authored", "reused"]


class StoredResponse(StrictModel):
    text: str
    model: str
    stop_reason: str | None
    usage: dict[str, int] = Field(description="Token counts, in the M1/M2 ledger field names.")
    recorded_at: str
    source: RecordSource | None = Field(default=None, description=(
        "Where this response came from when it differs from the record's source: 'reused' for an older "
        "version's response served under a ReuseRule (added 2026-11-08)."))
    reused_version: int | None = Field(default=None, ge=1, description="The older prompt version, when reused.")
    reused_sha256: str | None = Field(default=None, description="The older request's prompt_sha256, when reused.")

    def completion(self) -> Completion:
        """The stored response as a (not billed) completion."""
        u = self.usage
        write_1h = u.get("cache_write_1h_tokens", 0)
        usage = TokenUsage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_write_5m_tokens=u.get("cache_creation_input_tokens", 0) - write_1h,
            cache_write_1h_tokens=write_1h,
            cache_read_tokens=u.get("cache_read_input_tokens", 0),
        )
        return Completion(text=self.text, model=self.model, stop_reason=self.stop_reason, usage=usage, billed=False)


class StoredRecord(StrictModel):
    prompt_sha256: NonEmptyStr
    prompt_id: NonEmptyStr
    prompt_version: int = Field(ge=1)
    phase: str
    step: str
    component: str | None = None
    source: RecordSource
    request: dict[str, Any]
    responses: tuple[StoredResponse, ...] = ()


def usage_fields(usage: TokenUsage) -> dict[str, int]:
    """TokenUsage in the M1/M2 field names."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_write_5m_tokens + usage.cache_write_1h_tokens,
        "cache_write_1h_tokens": usage.cache_write_1h_tokens,
        "cache_read_input_tokens": usage.cache_read_tokens,
    }


def redact_request(request: dict[str, Any]) -> dict[str, Any]:
    """Hashed request fields, with base64 payloads replaced by sha256 + size (as in M2)."""
    subset = copy.deepcopy(hashed_subset(request))
    for message in subset.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            source = block.get("source") if isinstance(block, dict) else None
            if source and source.get("type") == "base64":
                data = source.pop("data").encode()
                source["data_sha256"] = hashlib.sha256(data).hexdigest()
                source["data_base64_chars"] = len(data)
    return subset


class ResponseStore:
    """Read/write access to one directory of stored responses."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, prompt_id: str, version: int, phash: str) -> Path:
        return self.root / prompt_id / f"v{version}" / f"{phash}.json"

    def record(self, prompt_id: str, version: int, phash: str) -> StoredRecord | None:
        path = self.path(prompt_id, version, phash)
        if not path.is_file():
            return None
        return StoredRecord.model_validate_json(path.read_bytes())

    def get(self, prompt_id: str, version: int, phash: str, sample: int) -> StoredResponse | None:
        record = self.record(prompt_id, version, phash)
        if record is None or sample >= len(record.responses):
            return None
        return record.responses[sample]

    def put(self, record: StoredRecord, sample: int, response: StoredResponse) -> Path:
        """Store `response` as sample `sample`; samples are written in order, 0 first."""
        existing = self.record(record.prompt_id, record.prompt_version, record.prompt_sha256)
        responses = existing.responses if existing is not None else ()
        if sample != len(responses):
            raise ValueError(
                f"{record.prompt_id} {record.prompt_sha256[:12]}: cannot store sample {sample}; "
                f"{len(responses)} stored (samples are written in order)"
            )
        updated = record.model_copy(update={"responses": (*responses, response)})
        path = self.path(record.prompt_id, record.prompt_version, record.prompt_sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        text = json.dumps(updated.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)
        return path
