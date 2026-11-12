"""Run-scoped output directories, artifact save/load, manifests and base-run selection.

Layout: machines/<id>/outputs/runs/<run_id>/
    manifest.json
    run.log.jsonl
    <phase>/<kind>.json

Every stored path is relative to the machine directory (hard rule 12). JSON is
written with sorted keys so bytes never depend on dict order (hard rule 14).
"""

import hashlib
import json
import re
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

import kaful
from kaful.artifacts.base import ArtifactBase, ArtifactRef
from kaful.artifacts.manifest import (
    BaseRun,
    CodeVersion,
    InputFile,
    PhaseRecord,
    PhaseStatus,
    RunManifest,
)
from kaful.config import MACHINE_CONFIG_FILE, REPO_ROOT, KafulSettings, MachineConfig
from kaful.llm.embeddings import Embedder
from kaful.llm.gateway import Gateway
from kaful.paths import resolve_in, to_rel
from kaful.phases import PhaseName

RUNS_SUBDIR = Path("outputs") / "runs"
# Machine-level, kept across runs (gitignored): response cache, pending batches, vector cache.
LLM_CACHE_SUBDIR = Path("outputs") / "llm_cache"
LLM_BATCHES_SUBDIR = Path("outputs") / "llm_batches"
EMBEDDING_CACHE_SUBDIR = Path("outputs") / "embedding_cache"
MANIFEST_FILE = "manifest.json"
LOG_FILE = "run.log.jsonl"
RUN_ID_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{6}$")
RUN_TOKEN_BYTES = 3
REUSABLE_STATUSES: frozenset[PhaseStatus] = frozenset({"passed", "reused"})

A = TypeVar("A", bound=ArtifactBase)


def new_run_id(now: datetime, token: str | None = None) -> str:
    """Sortable run id, e.g. '20261102T153012Z-a1b2c3'."""
    suffix = token if token is not None else secrets.token_hex(RUN_TOKEN_BYTES)
    run_id = f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{suffix}"
    if not RUN_ID_PATTERN.match(run_id):
        raise ValueError(f"bad run id {run_id!r}")
    return run_id


def runs_dir(machine_dir: Path) -> Path:
    return machine_dir / RUNS_SUBDIR


def dump_json(model: BaseModel) -> str:
    """Canonical JSON for a model: sorted keys, 2-space indent, trailing newline."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class ArtifactMismatch(ValueError):
    """A saved artifact is not what its reference recorded (changed bytes, wrong kind). The phase that loads it
    fails with this message; the run records it (M10b)."""


def load_artifact(machine_dir: Path, ref: ArtifactRef, model: type[A]) -> A:
    """Load an artifact by reference, checking its hash, kind and schema version."""
    path = resolve_in(machine_dir, ref.path)
    data = path.read_bytes()
    if sha256_bytes(data) != ref.sha256:
        raise ArtifactMismatch(f"{ref.path}: content does not match the recorded sha256")
    artifact = model.model_validate_json(data)
    if getattr(artifact, "kind", None) != ref.kind:
        raise ArtifactMismatch(f"{ref.path}: expected kind {ref.kind!r}")
    return artifact


def read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / MANIFEST_FILE).read_bytes())


def list_manifests(machine_dir: Path) -> list[RunManifest]:
    """Every run's manifest for this machine, oldest first.

    Ordered by creation time (microseconds), then run id: run ids alone only
    resolve to the second, so two runs started in the same second could misorder.
    """
    root = runs_dir(machine_dir)
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if RUN_ID_PATTERN.match(d.name) and (d / MANIFEST_FILE).is_file()]
    return sorted((read_manifest(d) for d in dirs), key=lambda m: (m.created_at, m.run_id))


def machine_inputs(machine_dir: Path, config: MachineConfig) -> tuple[InputFile, ...]:
    """machine.toml plus every listed document, hashed. A missing document is an error."""
    paths = [MACHINE_CONFIG_FILE, *(doc.path for doc in config.documents)]
    files = []
    for rel in sorted(paths):
        path = resolve_in(machine_dir, rel)
        if not path.is_file():
            raise FileNotFoundError(f"{config.machine_id}: listed input {rel!r} not found")
        files.append(InputFile(path=rel, sha256=sha256_file(path)))
    return tuple(files)


def code_version() -> CodeVersion:
    """Git commit and dirty flag of the Kaful checkout ('unknown' outside git)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return CodeVersion(
        git_commit=commit or "unknown",
        git_dirty=bool(status) if status is not None else True,
        kaful_version=kaful.__version__,
    )


class BaseRunError(ValueError):
    """No usable base run for the phases before --from / --only."""


def _flat(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested settings as dotted keys (understand.n_samples)."""
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Mapping):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def config_differences(before: Mapping[str, Any], now: Mapping[str, Any]) -> list[str]:
    """Settings that differ between two config snapshots, one line each, sorted by key."""
    a, b = _flat(before), _flat(now)
    return [f"config {key}: {json.dumps(a.get(key), sort_keys=True)} -> {json.dumps(b.get(key), sort_keys=True)}"
            for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)]


def input_differences(
    manifest: RunManifest, inputs: tuple[InputFile, ...], config: Mapping[str, Any]
) -> list[str]:
    """Human-readable differences between a run's inputs and settings and the current ones (sorted).

    The document index is not compared: no run but ingest knows an index hash, and a reused index is
    checked against its recorded hash whenever it is loaded (kaful.retrieval.load_index; M10b)."""
    before = {f.path: f.sha256 for f in manifest.inputs}
    now = {f.path: f.sha256 for f in inputs}
    diffs = [
        f"{path}: {'added' if path not in before else 'removed' if path not in now else 'changed'}"
        for path in sorted(before.keys() | now.keys())
        if before.get(path) != now.get(path)
    ]
    return diffs + config_differences(manifest.config, config)


def _upstream_passed(manifest: RunManifest, needed: tuple[PhaseName, ...]) -> bool:
    return all(manifest.phase(p).status in REUSABLE_STATUSES for p in needed)


def select_base_run(
    machine_dir: Path,
    needed: tuple[PhaseName, ...],
    inputs: tuple[InputFile, ...],
    config: Mapping[str, Any],
    explicit: str | None,
) -> tuple[RunManifest, BaseRun, list[str]]:
    """Pick the run whose artifacts are reused for `needed` phases.

    Without `explicit`, use the latest run whose `needed` phases passed, but only if
    its inputs (machine.toml and document sha256s) and its settings (the config
    snapshot) match the current ones; otherwise raise. An explicit run may differ;
    the differences are returned so the caller records them as an override.
    """
    names = ", ".join(needed)
    if explicit is not None:
        run_dir = runs_dir(machine_dir) / explicit
        if not (RUN_ID_PATTERN.match(explicit) and (run_dir / MANIFEST_FILE).is_file()):
            raise BaseRunError(f"base run {explicit!r} not found")
        manifest = read_manifest(run_dir)
        if not _upstream_passed(manifest, needed):
            raise BaseRunError(f"base run {explicit} did not pass all of: {names}")
        diffs = input_differences(manifest, inputs, config)
        return manifest, BaseRun(run_id=explicit, selected_by="explicit", inputs_match=not diffs), diffs
    passing = [m for m in list_manifests(machine_dir) if _upstream_passed(m, needed)]
    if not passing:
        raise BaseRunError(f"no earlier run passed {names}; run those phases first")
    latest = passing[-1]
    diffs = input_differences(latest, inputs, config)
    if diffs:
        raise BaseRunError(
            f"latest run that passed {names} ({latest.run_id}) has different inputs "
            f"({'; '.join(diffs)}); pass --base-run explicitly to reuse it"
        )
    return latest, BaseRun(run_id=latest.run_id, selected_by="auto", inputs_match=True), []


@dataclass
class RunContext:
    """Everything a phase needs: settings, machine, run directory and the live manifest."""

    settings: KafulSettings
    machine: MachineConfig
    machine_dir: Path
    run_dir: Path
    manifest: RunManifest
    gateway: Gateway | None = None
    embedder: Embedder | None = None

    def write_manifest(self) -> None:
        write_text_atomic(self.run_dir / MANIFEST_FILE, dump_json(self.manifest))

    def set_phase(self, record: PhaseRecord) -> None:
        """Replace one phase's record and rewrite the manifest."""
        phases = tuple(record if r.phase == record.phase else r for r in self.manifest.phases)
        total = sum(r.cost_usd for r in phases if r.status != "reused")
        self.manifest = self.manifest.model_copy(update={"phases": phases, "total_cost_usd": total})
        self.write_manifest()

    def save_artifact(self, phase: PhaseName, artifact: ArtifactBase) -> ArtifactRef:
        """Write `artifact` to <run>/<phase>/<kind>.json and return its reference."""
        kind = str(getattr(artifact, "kind"))
        if artifact.run_id != self.manifest.run_id:
            raise ValueError(f"artifact run_id {artifact.run_id!r} != run {self.manifest.run_id!r}")
        path = self.run_dir / phase / f"{kind}.json"
        text = dump_json(artifact)
        write_text_atomic(path, text)
        return ArtifactRef(
            kind=kind,
            run_id=self.manifest.run_id,
            path=to_rel(self.machine_dir, path),
            sha256=sha256_bytes(text.encode("utf-8")),
        )


def now_utc() -> datetime:
    return datetime.now(UTC)
