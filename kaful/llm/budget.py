"""Per-run API budget, checked before every call is sent.

A call is allowed only if `spent + reserved + worst_case <= limit`, where the worst
case prices the estimated input as a full cache write (when the prompt caches) and
assumes the whole `max_tokens` is generated. `reserved` holds the worst case of
batch requests that are submitted but not yet collected.
"""

import base64
from dataclasses import dataclass
from typing import Any

import fitz  # type: ignore[import-untyped]

from kaful.config import LLMSettings


class BudgetExceeded(RuntimeError):
    """The next call could take the run over its budget; nothing was sent."""


@dataclass
class RunBudget:
    limit_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.spent_usd - self.reserved_usd

    def check(self, worst_usd: float, what: str) -> None:
        """Raise BudgetExceeded unless `worst_usd` still fits."""
        if worst_usd > self.remaining_usd:
            raise BudgetExceeded(
                f"{what}: worst case ${worst_usd:.4f} exceeds the remaining ${self.remaining_usd:.4f} "
                f"of the ${self.limit_usd:.2f} budget (spent ${self.spent_usd:.4f}, "
                f"reserved ${self.reserved_usd:.4f})"
            )

    def charge(self, usd: float) -> None:
        self.spent_usd += usd

    def reserve(self, usd: float, what: str) -> None:
        self.check(usd, what)
        self.reserved_usd += usd

    def release(self, usd: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - usd)


def estimate_input_tokens(request: dict[str, Any], settings: LLMSettings) -> int:
    """Conservative input-token estimate from characters and PDF page counts (no network)."""
    chars = _text_chars(request.get("system"))
    pages = 0
    for message in request.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content:
            if block.get("type") == "text":
                chars += len(block["text"])
            elif block.get("type") == "document":
                pages += _pdf_pages(block)
            else:
                raise ValueError(f"no token estimate for content block type {block.get('type')!r}")
    if request.get("output_config"):
        chars += len(str(request["output_config"]))
    return int(chars / settings.chars_per_token_worst) + pages * settings.pdf_tokens_per_page_worst


def _text_chars(system: Any) -> int:
    if system is None:
        return 0
    if isinstance(system, str):
        return len(system)
    return sum(len(block.get("text", "")) for block in system)


def _pdf_pages(block: dict[str, Any]) -> int:
    source = block["source"]
    if source.get("type") != "base64":
        raise ValueError("only base64 PDF documents can be estimated offline")
    with fitz.open(stream=base64.b64decode(source["data"]), filetype="pdf") as doc:
        return int(doc.page_count)
