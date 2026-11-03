"""Evals on document pages: page-sourced inputs, numbers/contains scorers, batch prefetch, pending locks."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kaful.config import KafulSettings, LLMSettings
from kaful.evals.checks import resolve, run_check, score_golden
from kaful.evals.models import CheckSpec, EvalCase, EvalSet, GoldenSpec, SourceUnavailable
from kaful.evals.runner import run_eval
from kaful.prompts.registry import (
    PendingEntry,
    PromptLock,
    PromptNotLocked,
    PromptRegistry,
    load_prompt_dir,
    lock_problems,
    require_locked,
)
from tests.harness.kaful_llm import ScriptedTransport, make_gateway
from tests.harness.kaful_pdf import write_manual

REAL = PromptRegistry().load("ingest.extract_page").directory
REPLY = {
    "tables": [{"title": None, "source": "text_layer", "header": ["Parameter", "Value", "Unit"],
                "rows": [{"cells": ["Rated flow", "6,0", "m3/h"]}, {"cells": ["Inlet pressure", "1,5", "bar"]}]}],
    "figures": [{"kind": "screenshot", "caption": None, "description": "A touch screen.", "text_in_image": ["Clean"],
                 "values": [{"label": "Countdown", "value": "10", "unit": "s", "how": "printed"}]}],
    "ocr_text": None,
}
CASE = """id = "toy_table"
description = "Ruled table page of the synthetic manual."
page = {{ machine = "toy", path = "manual/m.pdf", sha256 = "{sha}", page = 2 }}

[inputs]
raw_text = {{ from_page = "raw_text" }}

[images]
page_image = {{ from_page = "png" }}

[[golden]]
metric = "table_cells"
field = "tables[].rows[].cells"
compare = "contains_all"
expected = ["Rated flow | 6,0 | m3/h", "Inlet pressure | 1.5"]
"""


@pytest.fixture
def toy(tmp_path: Path) -> tuple[Path, KafulSettings]:
    """The real extract_page prompt with a one-case eval set on a synthetic machine page."""
    pdf = write_manual(tmp_path / "machines" / "toy" / "manual" / "m.pdf")
    directory = tmp_path / "prompt"
    shutil.copytree(REAL, directory, ignore=shutil.ignore_patterns("cases", "__pycache__"))
    (directory / "evals" / "cases").mkdir()
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    (directory / "evals" / "cases" / "toy_table.toml").write_text(CASE.format(sha=sha))
    return directory, KafulSettings(machines_dir=tmp_path / "machines")


def test_page_inputs_come_from_the_pinned_document(toy: tuple[Path, KafulSettings]) -> None:
    directory, settings = toy
    evalset = EvalSet(load_prompt_dir(directory), settings)
    case = evalset.cases[0]
    assert "Rated flow" in evalset.inputs(case)["raw_text"]
    assert evalset.images(case)["page_image"].startswith(b"\x89PNG")
    (settings.machine_dir("toy") / "manual" / "m.pdf").unlink()
    assert not evalset.page_available(case)
    with pytest.raises(SourceUnavailable):
        evalset.inputs(case)


def test_batch_eval_scores_tables_and_numbers(tmp_path: Path, toy: tuple[Path, KafulSettings]) -> None:
    directory, settings = toy
    prompt = load_prompt_dir(directory)
    transport = ScriptedTransport(lambda _r: json.dumps(REPLY))
    gateway = make_gateway(tmp_path, transport, settings=LLMSettings(token_estimate="chars",
                                                                     default_model="claude-haiku-5-5"))
    result = run_eval(prompt, gateway, model="claude-haiku-5-5", effort=None, judge_prompt=None, judge_model=None,
                      source="live", git_commit="x", now=lambda: datetime(2026, 11, 3, tzinfo=UTC), batch=True,
                      evalset=EvalSet(prompt, settings))
    assert len(transport.batches) == 1 and transport.sent[0]["messages"][0]["content"][0]["type"] == "image"
    assert result.metric("table_cells") == 0.5  # '1.5' is not '1,5': decimal separators are kept apart
    assert result.metric("table_numbers_in_text_layer") == 1.0 and result.metric("schema_valid") == 1.0


def test_numbers_check_flags_numbers_missing_from_the_source() -> None:
    spec = CheckSpec(name="nums", kind="numbers_in_source", field="tables[source=text_layer].rows[].cells[]",
                     source="raw_text")
    output: dict[str, Any] = {"tables": [{"source": "text_layer", "rows": [{"cells": ["a", "6,0", "7.5"]}]},
                                         {"source": "image", "rows": [{"cells": ["99"]}]}]}
    outcome, scores = run_check(spec, output, {"raw_text": "flow 6,0 m3/h"}, None)
    assert scores == {"nums": 0.5} and outcome.details == ("not in source: 7.5",)
    _, none = run_check(spec, {"tables": []}, {"raw_text": "x"}, None)
    assert none == {}


def test_path_filters_and_contains_all_flattening() -> None:
    data = {"items": [{"k": "a", "v": 1}, {"k": "b", "v": 2}]}
    assert resolve(data, "items[k=b].v") == [2] and resolve(data, "items[].v") == [1, 2]
    figures = {"figures": [{"label": "Foam Texture", "value": "50", "unit": None}]}
    spec = GoldenSpec(metric="f", field="figures[]", compare="contains_all", expected=["Foam Texture | 50", "x"])
    assert score_golden(spec, figures) == 0.5


def test_page_inputs_need_a_page() -> None:
    with pytest.raises(ValueError, match="need a `page`"):
        EvalCase.model_validate({"id": "x", "description": "x", "inputs": {"t": {"from_page": "raw_text"}}})


def test_pending_prompts_are_listed_not_locked() -> None:
    registry = PromptRegistry()
    prompt = registry.load("ingest.extract_page")
    pending = PendingEntry(version=prompt.version, content_sha256=prompt.content_sha256, reason="first eval")
    lock = PromptLock(prompts={}, pending={prompt.id: pending})
    assert not [p for p in lock_problems(registry, lock) if p.startswith("ingest.")]
    stale = lock.model_copy(update={"pending": {prompt.id: pending.model_copy(update={"content_sha256": "0" * 64})}})
    assert any("changed since it was listed as pending" in p for p in lock_problems(registry, stale))
    with pytest.raises(PromptNotLocked, match="not locked"):
        require_locked(prompt, lock)
