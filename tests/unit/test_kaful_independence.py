"""kaful/ stays independent of ProgPy (outside its adapter) and of the legacy code."""

import ast
import os
import subprocess
import sys
from pathlib import Path

from kaful.config import REPO_ROOT

KAFUL_DIR = REPO_ROOT / "kaful"
PROGPY_ADAPTER = KAFUL_DIR / "backends" / "progpy"
PROGPY_MODULES = {"progpy", "prog_models", "prog_algs"}
ANTHROPIC_ADAPTER = KAFUL_DIR / "llm" / "anthropic_api.py"
LEGACY_MODULES = {"phases", "utils", "config", "run", "rag_frameworks", "demo", "prompts",
                  "scripts", "tests", "data_pipeline"}


def _sources() -> list[Path]:
    files = sorted(KAFUL_DIR.rglob("*.py"))
    assert len(files) > 20
    return files


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_progpy_is_imported_only_in_its_adapter() -> None:
    offenders = [str(p.relative_to(REPO_ROOT)) for p in _sources()
                 if not p.is_relative_to(PROGPY_ADAPTER) and _imported_top_levels(p) & PROGPY_MODULES]
    assert offenders == []


def test_anthropic_is_imported_only_in_its_transport() -> None:
    offenders = [str(p.relative_to(REPO_ROOT)) for p in _sources()
                 if p != ANTHROPIC_ADAPTER and "anthropic" in _imported_top_levels(p)]
    assert offenders == []


def test_no_legacy_imports() -> None:
    offenders = {str(p.relative_to(REPO_ROOT)): sorted(_imported_top_levels(p) & LEGACY_MODULES)
                 for p in _sources()}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_no_print_calls_in_pipeline_code() -> None:
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


def test_no_sys_path_or_sys_modules_access() -> None:
    offenders = []
    for path in _sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Attribute) and node.attr in {"path", "modules"}
                    and isinstance(node.value, ast.Name) and node.value.id == "sys"):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


def test_importing_every_kaful_module_loads_no_progpy_or_legacy_code() -> None:
    script = (
        "import importlib, pkgutil, sys, kaful\n"
        "mods = [m.name for m in pkgutil.walk_packages(kaful.__path__, 'kaful.')\n"
        "        if m.name != 'kaful.__main__'  # importing __main__ runs the CLI\n"
        "        and not m.name.startswith('kaful.backends.progpy')]  # the adapter (M10); loaded by name\n"
        "[importlib.import_module(m) for m in mods]\n"
        "bad = {'progpy', 'prog_models', 'prog_algs', 'phases', 'utils', 'config', 'run'}\n"
        "loaded = sorted({n.split('.')[0] for n in sys.modules} & bad)\n"
        "print(len(mods), loaded)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"}}
    done = subprocess.run([sys.executable, "-B", "-c", script], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, check=True)
    count, loaded = done.stdout.split(" ", 1)
    assert int(count) > 20
    assert loaded.strip() == "[]"
