"""Machine-relative paths (hard rule 12)."""

from pathlib import Path

import pytest

from kaful.paths import check_rel_path, resolve_in, to_rel


@pytest.mark.parametrize(("raw", "clean"), [("a/b.pdf", "a/b.pdf"), ("a/./b.pdf", "a/b.pdf"), ("x", "x")])
def test_relative_paths_are_normalised(raw: str, clean: str) -> None:
    assert check_rel_path(raw) == clean


@pytest.mark.parametrize("raw", ["", ".", "/abs", "../up", "a/../../b", "a\\b", "C:/x"])
def test_bad_paths_are_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        check_rel_path(raw)


def test_resolve_and_to_rel_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "f.json"
    target.parent.mkdir()
    target.write_text("{}")
    assert resolve_in(tmp_path, "outputs/f.json") == target.resolve()
    assert to_rel(tmp_path, target) == "outputs/f.json"


def test_symlinks_cannot_escape(tmp_path: Path) -> None:
    (tmp_path / "machine").mkdir()
    (tmp_path / "machine" / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        resolve_in(tmp_path / "machine", "escape/secret.txt")
