"""Versioned prompts: one directory per prompt, paired with its output model and eval set.

Layout (the prompt id's dots are directories under kaful/prompts/):

    kaful/prompts/<area>/<name>/
        prompt.toml   id, version, output model, max_tokens, mode, segments, inputs, thresholds
        system.md     fixed rules            (cached when `cache` is set)
        context.md    large shared material  (optional; cached when `cache` is set)
        task.md       the per-item request   (never cached; any declared images go just before it)
        models.py     the output model(s)
        evals/        the prompt's eval set (see kaful.evals)

Templates use `string.Template` placeholders (`$name` / `${name}`). Rendering is strict:
the declared inputs must be exactly the placeholders used, and every call must pass
exactly the declared inputs. Model choice is not in the prompt; it comes from
`[llm.prompts."<id>"]` in kaful.toml.

`content_sha256` covers everything that changes the prompt's behaviour or its pass
bar: prompt.toml except `version`, the segment texts, and the output model's JSON
schema. `versions.lock.json` pins (version, content_sha256) per prompt, so a change
without a version bump fails the test suite (see `lock_problems`).
"""

import hashlib
import importlib
import importlib.util
import json
import string
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, model_validator

from kaful.artifacts.base import SORTED_STRS, NonEmptyStr, Sha256, StrictModel
from kaful.config import REPO_ROOT

PROMPTS_DIR = Path(__file__).resolve().parent
PROMPT_FILE = "prompt.toml"
LOCK_FILE = PROMPTS_DIR / "versions.lock.json"

PromptId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")]
CacheSetting = Literal["5m", "1h", "none"]
SegmentName = Literal["system", "context", "task"]
SEGMENT_ORDER: tuple[SegmentName, ...] = get_args(SegmentName)


class PromptError(ValueError):
    """A prompt directory is invalid, or a render got the wrong inputs."""


class SegmentSpec(StrictModel):
    file: NonEmptyStr
    cache: CacheSetting = "none"


class Threshold(StrictModel):
    """Pass bar for one eval metric: value >= min and/or value <= max."""

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _one_bound(self) -> "Threshold":
        if self.min is None and self.max is None:
            raise ValueError("a threshold needs min and/or max")
        return self

    def passes(self, value: float) -> bool:
        return (self.min is None or value >= self.min) and (self.max is None or value <= self.max)


class PromptSpec(StrictModel):
    """Contents of prompt.toml."""

    id: PromptId
    version: int = Field(ge=1)
    description: NonEmptyStr
    output_model: str = Field(pattern=r"^[a-z_][a-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*$")
    max_tokens: int = Field(gt=0)
    mode: Literal["sync", "batch"] = "sync"
    segments: dict[SegmentName, SegmentSpec]
    inputs: Annotated[tuple[str, ...], SORTED_STRS] = ()
    images: Annotated[tuple[str, ...], SORTED_STRS] = Field(
        default=(), description="PNG attachments each call passes by name; sent just before the task segment."
    )
    thresholds: dict[str, Threshold] = {}

    @model_validator(mode="after")
    def _check_segments(self) -> "PromptSpec":
        missing = {"system", "task"} - self.segments.keys()
        if missing:
            raise ValueError(f"segments {sorted(missing)} are required")
        if self.segments["task"].cache != "none":
            raise ValueError("the task segment varies per call and is never cached")
        return self


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    context: str | None
    task: str


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt: spec, segment texts, output model and content hash."""

    spec: PromptSpec
    directory: Path
    texts: Mapping[SegmentName, str]
    output_model: type[BaseModel]
    content_sha256: str

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def version(self) -> int:
        return self.spec.version

    def render(self, inputs: Mapping[str, str]) -> RenderedPrompt:
        """Fill the templates; the inputs must be exactly the declared ones."""
        given = sorted(inputs)
        if given != list(self.spec.inputs):
            raise PromptError(f"{self.id}: inputs {given} != declared {list(self.spec.inputs)}")
        filled = {name: string.Template(text).substitute(inputs) for name, text in self.texts.items()}
        return RenderedPrompt(system=filled["system"], context=filled.get("context"), task=filled["task"])


def content_sha256(spec: PromptSpec, texts: Mapping[SegmentName, str], output_model: type[BaseModel]) -> str:
    payload = {
        # `images` is left out when empty so prompts without images keep their M5 hashes.
        "spec": spec.model_dump(mode="json", exclude={"version"} | (set() if spec.images else {"images"})),
        "texts": dict(sorted(texts.items())),
        "output_schema": output_model.model_json_schema(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_prompt_dir(directory: Path) -> Prompt:
    """Load and check one prompt directory."""
    with (directory / PROMPT_FILE).open("rb") as handle:
        spec = PromptSpec.model_validate(tomllib.load(handle))
    texts = {name: (directory / seg.file).read_text(encoding="utf-8")
             for name, seg in sorted(spec.segments.items(), key=lambda kv: SEGMENT_ORDER.index(kv[0]))}
    used: set[str] = set()
    for name, text in texts.items():
        template = string.Template(text)
        if not template.is_valid():
            raise PromptError(f"{spec.id}: {name} has an invalid $placeholder")
        used.update(template.get_identifiers())
    if sorted(used) != list(spec.inputs):
        raise PromptError(f"{spec.id}: placeholders {sorted(used)} != declared inputs {list(spec.inputs)}")
    model = _output_model(directory, spec.output_model)
    return Prompt(spec, directory, texts, model, content_sha256(spec, texts, model))


def _output_model(directory: Path, ref: str) -> type[BaseModel]:
    module_name, class_name = ref.split(":")
    if directory.is_relative_to(PROMPTS_DIR):
        dotted = ".".join(("kaful", "prompts", *directory.relative_to(PROMPTS_DIR).parts, module_name))
        module = importlib.import_module(dotted)
    else:  # a prompt outside the package (tests): load the file without registering it
        file_spec = importlib.util.spec_from_file_location(f"_kaful_prompt_{module_name}", directory / f"{module_name}.py")
        if file_spec is None or file_spec.loader is None:
            raise PromptError(f"cannot load {directory / module_name}.py")
        module = importlib.util.module_from_spec(file_spec)
        file_spec.loader.exec_module(module)
    model = getattr(module, class_name, None)
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise PromptError(f"{ref} is not a Pydantic model")
    return model


class PromptRegistry:
    """Every prompt under one root directory, by id."""

    def __init__(self, root: Path = PROMPTS_DIR) -> None:
        self.root = root

    def ids(self) -> list[str]:
        return sorted(".".join(p.parent.relative_to(self.root).parts) for p in self.root.rglob(PROMPT_FILE))

    def load(self, prompt_id: str) -> Prompt:
        directory = self.root.joinpath(*prompt_id.split("."))
        if not (directory / PROMPT_FILE).is_file():
            raise PromptError(f"no prompt {prompt_id!r} under {self.root.name}/")
        prompt = load_prompt_dir(directory)
        if prompt.id != prompt_id:
            raise PromptError(f"{directory}: id {prompt.id!r} does not match its directory")
        return prompt


class RunScores(StrictModel):
    """One passing eval run a lock rests on, with its metric values."""

    result: NonEmptyStr = Field(description="Repo-relative path of the eval result.")
    sample: int = Field(ge=0)
    metrics: dict[str, float]


class LockEntry(StrictModel):
    version: int = Field(ge=1)
    content_sha256: Sha256
    eval_results: Annotated[tuple[str, ...], SORTED_STRS] = Field(
        default=(), description="Repo-relative paths of the passing eval results this lock relies on."
    )
    run_scores: tuple[RunScores, ...] = Field(
        default=(), description="Per-run scores (locks from 2026-11-06 on rest on >= 2 runs of one model/route).")


class PendingEntry(StrictModel):
    """A new prompt awaiting its first eval run: listed so it is never silently unlocked."""

    version: int = Field(ge=1)
    content_sha256: Sha256
    reason: NonEmptyStr


class PromptLock(StrictModel):
    schema_version: Literal[1] = 1
    prompts: dict[str, LockEntry] = {}
    pending: dict[str, PendingEntry] = Field(
        default={}, description="New prompts not yet evaluated. The pipeline refuses to call them."
    )


class PromptNotLocked(RuntimeError):
    """A pipeline phase tried to use a prompt version that is not locked."""


def require_locked(prompt: Prompt, lock: PromptLock) -> None:
    """Raise unless this exact prompt version and content is locked (evaluated and passing)."""
    entry = lock.prompts.get(prompt.id)
    if entry is None or (entry.version, entry.content_sha256) != (prompt.version, prompt.content_sha256):
        raise PromptNotLocked(f"{prompt.id} v{prompt.version} is not locked; run its evals, then "
                              f"`kaful prompts lock {prompt.id}`")


def read_lock(path: Path = LOCK_FILE) -> PromptLock:
    return PromptLock.model_validate_json(path.read_bytes()) if path.is_file() else PromptLock()


def write_lock(lock: PromptLock, path: Path = LOCK_FILE) -> None:
    data = lock.model_dump(mode="json")
    data["prompts"] = dict(sorted(data["prompts"].items()))
    data["pending"] = dict(sorted(data["pending"].items()))
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lock_problems(registry: PromptRegistry, lock: PromptLock) -> list[str]:
    """Why the lock does not match the prompts on disk (empty when it does)."""
    problems = []
    for prompt_id in registry.ids():
        prompt = registry.load(prompt_id)
        entry = lock.prompts.get(prompt_id)
        pending = lock.pending.get(prompt_id)
        if entry is None and pending is not None:
            if (pending.version, pending.content_sha256) != (prompt.version, prompt.content_sha256):
                problems.append(f"{prompt_id}: changed since it was listed as pending; update its pending entry")
            continue
        if entry is None:
            problems.append(f"{prompt_id}: not in versions.lock.json; run its evals, then `kaful prompts lock`")
        elif prompt.content_sha256 != entry.content_sha256 and prompt.version == entry.version:
            problems.append(f"{prompt_id}: content changed but version is still {prompt.version}; bump it")
        elif prompt.version != entry.version or prompt.content_sha256 != entry.content_sha256:
            problems.append(f"{prompt_id}: v{prompt.version} is not locked (lock has v{entry.version}); "
                            f"run its evals, then `kaful prompts lock {prompt_id}`")
        problems += [f"{prompt_id}: locked eval result {p} is missing"
                     for p in entry.eval_results if not (REPO_ROOT / p).is_file()] if entry else []
    problems += [f"{p}: locked but no longer exists" for p in sorted(lock.prompts) if p not in registry.ids()]
    problems += [f"{p}: pending but no longer exists" for p in sorted(lock.pending) if p not in registry.ids()]
    problems += [f"{p}: both locked and pending" for p in sorted(lock.pending) if p in lock.prompts]
    return problems
