"""Offline ingest helpers: a synthetic machine, a fake embedding transport, a scripted extractor."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kaful.config import KafulSettings
from kaful.prompts.registry import LockEntry, PromptLock, PromptRegistry
from tests.harness.kaful_pdf import write_manual

MACHINE_ID = "pdf_machine"
DIMENSIONS = 8
MACHINE_TOML = """\
machine_id = "pdf_machine"
display_name = "Synthetic PDF machine"
time_unit = "hour"
time_unit_provisional = true

[[documents]]
path = "manual/tp_manual.pdf"
kind = "manual"
"""
EXTRACTION = {
    "tables": [{"title": "Settings", "source": "text_layer", "header": ["Parameter", "Value", "-"],
                "rows": [{"cells": ["Cake Thickness [mm]", "15,0", "-"]}]}],
    "figures": [{"kind": "screenshot", "caption": None, "description": "A touch screen with a countdown.",
                 "text_in_image": ["Clean the screen now"],
                 "values": [{"label": "Countdown", "value": "10", "unit": "s", "how": "printed"}]}],
    "ocr_text": None,
}


def make_pdf_machine(machines_dir: Path) -> Path:
    machine_dir = machines_dir / MACHINE_ID
    write_manual(machine_dir / "manual" / "tp_manual.pdf")
    (machine_dir / "machine.toml").write_text(MACHINE_TOML)
    return machine_dir


def ingest_settings(machines_dir: Path) -> KafulSettings:
    settings = KafulSettings(machines_dir=machines_dir)
    # No per-prompt entries from kaful.toml: every prompt runs on Haiku, without escalation.
    llm = settings.llm.model_copy(update={"token_estimate": "chars", "default_model": "claude-haiku-5-5", "prompts": {}})
    ingest = settings.ingest.model_copy(update={"embedding_dimensions": DIMENSIONS, "label_min_pages": 3})
    return settings.model_copy(update={"llm": llm, "ingest": ingest})


def extraction_reply(_request: dict[str, Any]) -> str:
    return json.dumps(EXTRACTION)


def locked_extract_page() -> PromptLock:
    """A lock in which the current ingest.extract_page is locked (tests only)."""
    prompt = PromptRegistry().load("ingest.extract_page")
    return PromptLock(prompts={prompt.id: LockEntry(version=prompt.version, content_sha256=prompt.content_sha256)})


@dataclass
class FakeEmbeddings:
    """Deterministic vectors from the text's hash; counts calls and texts."""

    calls: list[int] = field(default_factory=list)

    def embed(self, texts: Sequence[str], model: str, dimensions: int) -> tuple[list[list[float]], int]:
        self.calls.append(len(texts))
        vectors = [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:dimensions]] for t in texts]
        return vectors, sum(len(t) // 4 + 1 for t in texts)
