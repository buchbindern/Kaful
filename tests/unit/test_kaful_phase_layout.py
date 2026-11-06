"""The phase layout rule (CLAUDE.md, Naming): phase folders follow PHASE_ORDER; only pipeline
steps are lettered (a, b, c, ... in execution order) and phase.py calls each of them directly,
in order, after a map that names them; helpers/ say which steps use them; tools/ is unprefixed."""

import ast
import re
from pathlib import Path

from kaful.phases import PHASE_ORDER

PHASES_DIR = Path(__file__).resolve().parents[2] / "kaful" / "phases"
STEP = re.compile(r"^([a-z])_[a-z0-9_]+$")
ALLOWED_DIRS = {"helpers", "tools", "__pycache__"}


def _phase_dirs() -> list[Path]:
    return sorted(p for p in PHASES_DIR.iterdir() if p.is_dir() and p.name != "__pycache__")


def _steps(folder: Path) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.py") if p.name not in {"__init__.py", "phase.py"})


def _first_calls(phase: Path, folder: str) -> list[str]:
    """Step modules phase.py calls (via names imported from them), in order of first call."""
    tree = ast.parse(phase.read_text(encoding="utf-8"))
    imported: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(f"kaful.phases.{folder}."):
            module = node.module.rsplit(".", 1)[1]
            if STEP.match(module):
                imported.update({alias.asname or alias.name: module for alias in node.names})
    first: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported:
            module = imported[node.func.id]
            first[module] = min(first.get(module, node.lineno), node.lineno)
    return sorted(first, key=lambda m: first[m])


def test_phase_folders_match_phase_order() -> None:
    assert [p.name for p in _phase_dirs()] == [f"p{i}_{name}" for i, name in enumerate(PHASE_ORDER, 1)]


def test_only_steps_helpers_and_tools_live_in_a_phase() -> None:
    for folder in _phase_dirs():
        dirs = {p.name for p in folder.iterdir() if p.is_dir()}
        assert dirs <= ALLOWED_DIRS, f"{folder.name}: unexpected folders {sorted(dirs - ALLOWED_DIRS)}"
        steps = _steps(folder)
        assert all(STEP.match(s) for s in steps), f"{folder.name}: unlettered modules {steps} (helpers/ or tools/)"
        assert [s[0] for s in steps] == [chr(ord("a") + i) for i in range(len(steps))], f"{folder.name}: {steps}"


def test_phase_py_maps_and_calls_every_step_in_order() -> None:
    for folder in _phase_dirs():
        steps, phase = _steps(folder), folder / "phase.py"
        if not steps:
            continue
        assert phase.is_file(), f"{folder.name} has steps but no phase.py"
        assert _first_calls(phase, folder.name) == steps, f"{folder.name}/phase.py must call {steps} directly, in order"
        doc = ast.get_docstring(ast.parse(phase.read_text(encoding="utf-8"))) or ""
        positions = [doc.find(s) for s in steps]
        assert -1 not in positions and positions == sorted(positions), f"{folder.name}/phase.py: map must list {steps}"


def test_helpers_say_which_steps_use_them_and_nothing_outside_steps_is_lettered() -> None:
    for folder in _phase_dirs():
        steps = set(_steps(folder))
        for helper in sorted((folder / "helpers").glob("*.py")) if (folder / "helpers").is_dir() else []:
            assert not STEP.match(helper.stem), f"{helper}: helpers are not lettered"
            if helper.name == "__init__.py":
                continue
            doc = ast.get_docstring(ast.parse(helper.read_text(encoding="utf-8"))) or ""
            assert any(step in doc for step in steps), f"{helper}: its docstring must name the steps that use it"
        for tool in sorted((folder / "tools").glob("*.py")) if (folder / "tools").is_dir() else []:
            assert not STEP.match(tool.stem), f"{tool}: tools are not lettered"
