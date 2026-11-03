"""Machine-relative paths (hard rule 12).

Artifacts and generated code never store absolute paths. Every stored path is a
`RelPath`: POSIX-style, relative, and without `..` segments. It is resolved
against the machine directory only at load time, with `resolve_in`.
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated

from pydantic import AfterValidator


def check_rel_path(value: str) -> str:
    """Validate and normalise a machine-relative path; raise ValueError otherwise."""
    if not value:
        raise ValueError("path is empty")
    if "\\" in value:
        raise ValueError(f"path must use '/' separators: {value!r}")
    posix = PurePosixPath(value)
    if posix.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"path must be relative to the machine directory: {value!r}")
    if ".." in posix.parts:
        raise ValueError(f"path must not contain '..': {value!r}")
    normalised = posix.as_posix()
    if normalised == ".":
        raise ValueError("path must name a file or directory, not '.'")
    return normalised


RelPath = Annotated[str, AfterValidator(check_rel_path)]


def resolve_in(base: Path, rel: str) -> Path:
    """Resolve `rel` under `base`, refusing anything that escapes it (including via symlinks)."""
    root = base.resolve()
    target = (root / check_rel_path(rel)).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"path escapes {root.name}/: {rel!r}")
    return target


def to_rel(base: Path, path: Path) -> str:
    """Express `path` relative to `base` as a RelPath string."""
    return check_rel_path(path.resolve().relative_to(base.resolve()).as_posix())
