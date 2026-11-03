"""Prompt registry: loading, strict rendering, content hash and the version lock."""

import shutil
from pathlib import Path

import pytest

from kaful.prompts.registry import (
    LockEntry,
    PromptError,
    PromptLock,
    PromptRegistry,
    load_prompt_dir,
    lock_problems,
)

TOY = Path(__file__).resolve().parents[1] / "fixtures" / "prompts" / "toy" / "echo_items"


@pytest.fixture
def registry(tmp_path: Path) -> PromptRegistry:
    shutil.copytree(TOY, tmp_path / "toy" / "echo_items")
    return PromptRegistry(tmp_path)


def _edit(registry: PromptRegistry, name: str, old: str, new: str) -> None:
    path = registry.root / "toy" / "echo_items" / name
    path.write_text(path.read_text().replace(old, new))


def test_render_is_strict(registry: PromptRegistry) -> None:
    prompt = registry.load("toy.echo_items")
    rendered = prompt.render({"doc": "D", "question": "Q?"})
    assert (rendered.system, rendered.context, rendered.task) == (
        "You list items. Costs $5 to ask.\n", "Document:\nD\n", "Question: Q?\n")
    with pytest.raises(PromptError, match="inputs"):
        prompt.render({"doc": "D"})
    with pytest.raises(PromptError, match="inputs"):
        prompt.render({"doc": "D", "question": "Q", "extra": "x"})


def test_placeholders_must_match_declared_inputs(registry: PromptRegistry) -> None:
    _edit(registry, "task.md", "${question}", "${question} $unknown")
    with pytest.raises(PromptError, match="placeholders"):
        registry.load("toy.echo_items")


def test_task_segment_is_never_cached(registry: PromptRegistry) -> None:
    _edit(registry, "prompt.toml", 'task = { file = "task.md" }', 'task = { file = "task.md", cache = "5m" }')
    with pytest.raises(ValueError, match="never cached"):
        registry.load("toy.echo_items")


def _locked(registry: PromptRegistry) -> PromptLock:
    prompt = registry.load("toy.echo_items")
    return PromptLock(prompts={prompt.id: LockEntry(version=prompt.version, content_sha256=prompt.content_sha256)})


@pytest.mark.parametrize(("name", "old", "new"), [
    ("system.md", "You list items.", "You list all items."),
    ("models.py", "min_length=1", "min_length=2"),
    ("prompt.toml", "max_tokens = 500", "max_tokens = 600"),
    ("prompt.toml", "schema_valid = { min = 1.0 }", "schema_valid = { min = 0.5 }"),
])
def test_changes_without_a_version_bump_are_caught(registry: PromptRegistry, name: str, old: str, new: str) -> None:
    lock = _locked(registry)
    assert lock_problems(registry, lock) == []
    _edit(registry, name, old, new)
    assert lock_problems(registry, lock) == ["toy.echo_items: content changed but version is still 1; bump it"]
    _edit(registry, "prompt.toml", "version = 1", "version = 2")
    assert "v2 is not locked" in lock_problems(registry, lock)[0]


def test_version_alone_does_not_change_content_hash(registry: PromptRegistry) -> None:
    before = registry.load("toy.echo_items").content_sha256
    _edit(registry, "prompt.toml", "version = 1", "version = 2")
    assert registry.load("toy.echo_items").content_sha256 == before


def test_unlocked_and_removed_prompts_are_reported(registry: PromptRegistry) -> None:
    assert "not in versions.lock.json" in lock_problems(registry, PromptLock())[0]
    stale = PromptLock(prompts={**_locked(registry).prompts,
                                "toy.gone": LockEntry(version=1, content_sha256="0" * 64)})
    assert lock_problems(registry, stale) == ["toy.gone: locked but no longer exists"]


def test_load_outside_package_does_not_register_module() -> None:
    import sys

    load_prompt_dir(TOY)
    assert not [m for m in sys.modules if m.startswith("_kaful_prompt")]
