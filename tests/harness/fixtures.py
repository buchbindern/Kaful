"""
tests/harness/fixtures.py
-------------------------
Recorded LLM responses, keyed by `utils.prompt_hash.prompt_hash` (the same value
the ledger logs as `prompt_sha256`). One JSON file per distinct request:

    {
      "prompt_sha256": "...",
      "phase": "phase2", "step": "step_c_extract", "component": null,
      "source": "recorded" | "hand_authored",
      "request": {...canonical request, base64 payloads replaced by their sha256...},
      "responses": [{"text": "...", "model": "...", "stop_reason": "...",
                     "usage": {...}, "recorded_at": "2026-11-02T..."}]
    }

A request sent N times (2c sends the same prompt 5 times) replays the N-th
response. Running past the end, or sending an unknown request, is an error
that names the step and shows the nearest fixture's diff.
"""

import copy
import difflib
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from utils.prompt_hash import HASHED_FIELDS, prompt_hash

MAX_DIFF_LINES = 40


class MissingFixtureError(LookupError):
    """No recorded response for this request (or all of them already used)."""


def redact_request(request: dict[str, Any]) -> dict[str, Any]:
    """Hashed request fields, with base64 payloads replaced by sha256 + size."""
    subset = copy.deepcopy({k: request.get(k) for k in HASHED_FIELDS})
    for msg in subset.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                source = block.get("source") if isinstance(block, dict) else None
                if source and source.get("type") == "base64":
                    data = source.pop("data").encode()
                    source["data_sha256"] = hashlib.sha256(data).hexdigest()
                    source["data_base64_chars"] = len(data)
    return subset


def fixture_name(scope: dict[str, Any], phash: str) -> str:
    component = scope.get("component") or "-"
    return f"{scope['phase']}__{scope['step']}__{component}__{phash[:12]}.json"


class FixtureStore:
    """Read/write access to one directory of fixture files."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.records: dict[str, dict[str, Any]] = {}
        self.paths: dict[str, Path] = {}
        self.used: Counter[str] = Counter()
        for path in sorted(self.directory.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            self.records[record["prompt_sha256"]] = record
            self.paths[record["prompt_sha256"]] = path

    def next_response(self, request: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        """The next unused response for this request."""
        phash = prompt_hash(request)
        record = self.records.get(phash)
        n = self.used[phash]
        if record is None or n >= len(record["responses"]):
            raise MissingFixtureError(self._explain(request, scope, phash, record, n))
        self.used[phash] += 1
        return record["responses"][n]

    def append(self, request: dict[str, Any], scope: dict[str, Any],
               response: dict[str, Any], source: str) -> Path:
        """Add one response to the request's fixture file (created if new)."""
        phash = prompt_hash(request)
        record = self.records.get(phash)
        if record is None:
            record = {"prompt_sha256": phash, "phase": scope["phase"], "step": scope["step"],
                      "component": scope.get("component"), "source": source,
                      "request": redact_request(request), "responses": []}
            self.records[phash] = record
            self.paths[phash] = self.directory / fixture_name(scope, phash)
        record["responses"].append(response)
        self.used[phash] += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths[phash].write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        return self.paths[phash]

    def _explain(self, request: dict[str, Any], scope: dict[str, Any], phash: str,
                 record: dict[str, Any] | None, n: int) -> str:
        where = (f"{scope.get('phase')}/{scope.get('step')}"
                 f" component={scope.get('component')} model={request.get('model')}")
        if record is not None:
            return (f"fixture exhausted for {where} prompt_sha256={phash}: call #{n + 1}, "
                    f"{len(record['responses'])} recorded in {self.paths[phash].name}")
        msg = f"no fixture for {where} prompt_sha256={phash} in {self.directory}"
        nearest = self._nearest(scope)
        if nearest is None:
            return msg + " (no fixture recorded for this step)"
        old = json.dumps(nearest["request"], indent=1, sort_keys=True).splitlines()
        new = json.dumps(redact_request(request), indent=1, sort_keys=True).splitlines()
        diff = list(difflib.unified_diff(old, new, "recorded", "sent", lineterm="", n=1))
        shown = "\n".join(diff[:MAX_DIFF_LINES])
        more = f"\n... {len(diff) - MAX_DIFF_LINES} more diff lines" if len(diff) > MAX_DIFF_LINES else ""
        return f"{msg}\nnearest: {self.paths[nearest['prompt_sha256']].name}\n{shown}{more}"

    def _nearest(self, scope: dict[str, Any]) -> dict[str, Any] | None:
        same_step = [r for r in self.records.values()
                     if (r["phase"], r["step"]) == (scope.get("phase"), scope.get("step"))]
        same_comp = [r for r in same_step if r.get("component") == scope.get("component")]
        candidates = same_comp or same_step
        return candidates[0] if candidates else None
