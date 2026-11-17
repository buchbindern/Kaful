"""Citing the files a number came from (used by b_grounding, c_prompt_quality, d_twin_quality, e_cost).

A repository file is written `repo:<path>`; any other path is relative to the machine directory. Every reference
carries the sha256 of the bytes that were read, so a report shows exactly what it rests on (rule 12: no absolute
paths).
"""

from pathlib import Path

from kaful.artifacts.evaluation import FileRef
from kaful.config import REPO_ROOT
from kaful.paths import resolve_in
from kaful.runs import sha256_file

REPO_PREFIX = "repo:"


def repo_ref(rel: str, root: Path = REPO_ROOT) -> FileRef:
    return FileRef(path=f"{REPO_PREFIX}{rel}", sha256=sha256_file(resolve_in(root, rel)))


def machine_ref(machine_dir: Path, rel: str) -> FileRef:
    return FileRef(path=rel, sha256=sha256_file(resolve_in(machine_dir, rel)))
