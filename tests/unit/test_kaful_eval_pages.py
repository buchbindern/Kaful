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


def _routed_settings() -> LLMSettings:
    from kaful.config import Escalation, ModelChoice

    choice = ModelChoice(model="claude-haiku-5-5", escalate=Escalation(
        model="claude-sonnet-5-5", when=("first_pass_failed", "output_has_tables")))
    return LLMSettings(token_estimate="chars", default_model="claude-haiku-5-5",
                       prompts={"ingest.extract_page": choice})


def _routed_reply(haiku_tables: bool) -> Any:
    def reply(request: dict[str, Any]) -> str:
        if request["model"] == "claude-sonnet-5-5":  # the escalation reads the merged cell right
            rows = [{"cells": ["Rated flow", "6,0", "m3/h"]}, {"cells": ["Inlet pressure", "1.5", "bar"]}]
            return json.dumps({**REPLY, "tables": [{**REPLY["tables"][0], "rows": rows}]})
        return json.dumps(REPLY if haiku_tables else {**REPLY, "tables": []})
    return reply


def _routed_run(tmp_path: Path, toy: tuple[Path, KafulSettings], haiku_tables: bool, sample: int = 0,
                batch: bool = False) -> Any:
    directory, settings = toy
    prompt = load_prompt_dir(directory)
    transport = ScriptedTransport(_routed_reply(haiku_tables))
    gateway = make_gateway(tmp_path / f"gw{sample}", transport, settings=_routed_settings())
    result = run_eval(prompt, gateway, model="routed", effort=None, judge_prompt=None, judge_model=None,
                      source="live", git_commit="x", now=lambda: datetime(2026, 11, 6, tzinfo=UTC), batch=batch,
                      evalset=EvalSet(prompt, settings), sample=sample, routed=True)
    return result, transport


def test_routed_eval_scores_what_the_pipeline_keeps(tmp_path: Path, toy: tuple[Path, KafulSettings]) -> None:
    result, transport = _routed_run(tmp_path, toy, haiku_tables=True)
    assert [r["model"] for r in transport.sent] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert result.metric("table_cells") == 1.0 and result.metric("table_detection") == 1.0  # Sonnet's table
    routing = next(c for c in result.cases[0].checks if c.check == "routing")
    assert routing.details == ("first pass claude-haiku-5-5: escalated (output_has_tables) to claude-sonnet-5-5",)
    assert result.routing == "claude-haiku-5-5 -> claude-sonnet-5-5 when first_pass_failed or output_has_tables"
    batched, transport = _routed_run(tmp_path / "b", toy, haiku_tables=True, batch=True)
    assert [[req["model"] for _, req in b] for _, b in sorted(transport.batches.items())] == [
        ["claude-haiku-5-5"], ["claude-sonnet-5-5"]]  # first passes, then the escalations
    assert len(transport.sent) == 2  # each request sent once (the harness logs batched requests too)
    assert batched.metric("table_cells") == 1.0


def test_a_missed_table_fails_detection(tmp_path: Path, toy: tuple[Path, KafulSettings]) -> None:
    result, transport = _routed_run(tmp_path, toy, haiku_tables=False)
    assert [r["model"] for r in transport.sent] == ["claude-haiku-5-5"]  # never escalated
    assert result.metric("table_detection") == 0.0 and result.metric("table_cells") == 0.0


def test_a_repeat_run_makes_fresh_calls(tmp_path: Path, toy: tuple[Path, KafulSettings]) -> None:
    from kaful.evals.models import results_path

    result, transport = _routed_run(tmp_path, toy, haiku_tables=True, sample=1)
    assert result.sample == 1 and len(transport.sent) == 2
    assert results_path(tmp_path, "p", 3, "m", None, 1).name == "v3__m__run-1.json"
    assert results_path(tmp_path, "p", 3, "m").name == "v3__m.json"


def test_a_lock_needs_two_passing_runs_and_records_their_scores(tmp_path: Path, toy: tuple[Path, KafulSettings],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse

    from kaful.evals import commands
    from kaful.evals.runner import save_result

    directory, _ = toy
    prompt = load_prompt_dir(directory)
    monkeypatch.setattr(commands, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(commands, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(commands.PromptRegistry, "load", lambda _self, _pid: prompt)
    written: list[PromptLock] = []
    monkeypatch.setattr(commands, "read_lock", lambda: PromptLock())
    monkeypatch.setattr(commands, "write_lock", written.append)
    args = argparse.Namespace(prompt_id=prompt.id)

    def run(sample: int, passed: bool) -> None:
        result, _ = _routed_run(tmp_path / f"r{sample}", toy, haiku_tables=True, sample=sample)
        save_result(result.model_copy(update={"model": "routed", "passed": passed}), tmp_path / "results")

    run(0, True)
    with pytest.raises(ValueError, match="2\\+ runs"):
        commands.prompts_lock(args, KafulSettings())
    run(1, False)
    with pytest.raises(ValueError, match="all passing"):
        commands.prompts_lock(args, KafulSettings())
    run(1, True)
    assert commands.prompts_lock(args, KafulSettings()) == 0
    entry = written[-1].prompts[prompt.id]
    assert [(r.sample, r.metrics["table_cells"]) for r in entry.run_scores] == [(0, 1.0), (1, 1.0)]
    v = f"results/ingest.extract_page/v{prompt.version}__routed"
    assert entry.eval_results == (f"{v}.json", f"{v}__run-1.json")



def test_a_lock_records_every_model_that_is_ready(tmp_path: Path, toy: tuple[Path, KafulSettings],
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """Both models' per-run scores go into the lock (decided 2026-11-07); a model with a failing run is left out."""
    import argparse

    from kaful.evals import commands
    from kaful.evals.runner import save_result

    directory, _ = toy
    prompt = load_prompt_dir(directory)
    monkeypatch.setattr(commands, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(commands, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(commands.PromptRegistry, "load", lambda _self, _pid: prompt)
    written: list[PromptLock] = []
    monkeypatch.setattr(commands, "read_lock", lambda: PromptLock())
    monkeypatch.setattr(commands, "write_lock", written.append)
    for model, outcomes in (("a-model", (True, True)), ("b-model", (True, True)), ("c-model", (True, False))):
        for sample, passed in enumerate(outcomes):
            result, _ = _routed_run(tmp_path / f"{model}{sample}", toy, haiku_tables=True, sample=sample)
            save_result(result.model_copy(update={"model": model, "passed": passed}), tmp_path / "results")
    assert commands.prompts_lock(argparse.Namespace(prompt_id=prompt.id), KafulSettings()) == 0
    entry = written[-1].prompts[prompt.id]
    v = f"results/ingest.extract_page/v{prompt.version}"
    assert [(r.result, r.sample) for r in entry.run_scores] == [
        (f"{v}__a-model.json", 0), (f"{v}__a-model__run-1.json", 1),
        (f"{v}__b-model.json", 0), (f"{v}__b-model__run-1.json", 1)]


def test_golden_review_shows_the_printed_label(tmp_path: Path, toy: tuple[Path, KafulSettings]) -> None:
    from kaful.evals.golden_review import review_html

    directory, settings = toy
    evalset = EvalSet(load_prompt_dir(directory), settings)
    page = review_html(evalset, list(evalset.cases))
    assert "manual/m.pdf · page 2 (label 2)" in page
