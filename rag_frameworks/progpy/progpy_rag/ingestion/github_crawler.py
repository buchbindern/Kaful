"""
ingestion/github_crawler.py

Clones the ProgPy GitHub repo and extracts:
  - Source .py files (chunked per class and per method)
  - Example scripts (chunked as complete examples)
  - RST doc source files (chunked per section)

Outputs raw text files to data/raw/github/ for the chunker to process.
"""

import os
import ast
import subprocess
from pathlib import Path
from dataclasses import dataclass


REPO_URL = "https://github.com/nasa/progpy.git"
CLONE_DIR = Path("./data/raw/github/progpy")

# Directories within the repo that are valuable for RAG
VALUABLE_DIRS = [
    "src/progpy",       # core source — models, state estimators, predictors
    "examples",         # complete runnable examples (gold for RAG)
    "tests",            # shows correct API usage patterns
    "docs/source",      # RST source (cleaner than rendered HTML)
]

# Domain keywords used to auto-tag chunks during extraction
DOMAIN_KEYWORDS = {
    "battery":    ["battery", "batt", "soc", "eol"],
    "thermal":    ["thermal", "temperature", "heat", "temp"],
    "mechanical": ["thrown", "vibration", "fatigue", "crack", "stress"],
    "pneumatic":  ["pump", "pneumatic", "pressure", "valve"],
    "electrical": ["electrical", "power", "voltage", "current"],
    "general":    [],  # fallback
}


@dataclass
class RawChunk:
    """Intermediate representation before embedding."""
    text: str
    source: str          # relative path within repo
    chunk_type: str      # "class" | "method" | "example" | "guide"
    name: str            # class/method/file name
    domain: str          # auto-detected domain
    pattern: str         # "component" | "composite" | "degradation" | "general"


def clone_or_update() -> None:
    """Clone the repo if not present, pull latest if it is."""
    if CLONE_DIR.exists():
        print(f"Repo exists at {CLONE_DIR}, pulling latest...")
        subprocess.run(["git", "-C", str(CLONE_DIR), "pull"], check=True)
    else:
        print(f"Cloning {REPO_URL} → {CLONE_DIR}")
        CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_URL, str(CLONE_DIR)], check=True)


def detect_domain(text: str) -> str:
    """Infer domain from content keywords."""
    text_lower = text.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain == "general":
            continue
        if any(k in text_lower for k in keywords):
            return domain
    return "general"


def detect_pattern(text: str, name: str) -> str:
    """Infer ProgPy pattern from content."""
    combined = (text + name).lower()
    if "compositemodel" in combined:
        return "composite"
    if any(k in combined for k in ["threshold_met", "event_state", "simulate_to_threshold"]):
        return "degradation"
    if "prognosticsmodel" in combined or "state_equation" in combined:
        return "component"
    return "general"


# ------------------------------------------------------------------
# Python source file extraction
# ------------------------------------------------------------------

def extract_classes_from_file(filepath: Path, repo_root: Path) -> list[RawChunk]:
    """Parse a .py file with AST and extract one chunk per class."""
    chunks = []
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    rel_path = str(filepath.relative_to(repo_root))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Extract the full class source text
            lines = source.splitlines()
            start = node.lineno - 1
            end = node.end_lineno
            class_text = "\n".join(lines[start:end])

            domain = detect_domain(class_text)
            pattern = detect_pattern(class_text, node.name)

            chunks.append(RawChunk(
                text=class_text,
                source=f"github:{rel_path}",
                chunk_type="class",
                name=node.name,
                domain=domain,
                pattern=pattern,
            ))

            # Extract each method as its own chunk, with class header for context
            class_header = "\n".join(lines[start:min(start+15, end)])

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_start = item.lineno - 1
                    method_end = item.end_lineno
                    method_text = "\n".join(lines[method_start:method_end])

                    # Prepend class name and header so chunk is self-contained
                    full_method_text = (
                        f"# Class: {node.name}\n"
                        f"{class_header}\n"
                        f"    ...\n"
                        f"{method_text}"
                    )

                    chunks.append(RawChunk(
                        text=full_method_text,
                        source=f"github:{rel_path}",
                        chunk_type="method",
                        name=f"{node.name}.{item.name}",
                        domain=domain,
                        pattern=detect_pattern(method_text, item.name),
                    ))

    return chunks


def extract_example_file(filepath: Path, repo_root: Path) -> RawChunk:
    """Treat an entire example script as one chunk."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    rel_path = str(filepath.relative_to(repo_root))
    return RawChunk(
        text=text,
        source=f"github:{rel_path}",
        chunk_type="example",
        name=filepath.stem,
        domain=detect_domain(text),
        pattern=detect_pattern(text, filepath.stem),
    )


def extract_rst_sections(filepath: Path, repo_root: Path) -> list[RawChunk]:
    """Split an RST file into one chunk per top-level section."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    rel_path = str(filepath.relative_to(repo_root))
    chunks = []

    # Split on RST section headers (underline style: ===, ---, ~~~)
    import re
    sections = re.split(r'\n(?=[^\n]+\n[=\-~]{3,}\n)', text)

    for i, section in enumerate(sections):
        if len(section.strip()) < 50:
            continue
        chunks.append(RawChunk(
            text=section.strip(),
            source=f"github:{rel_path}",
            chunk_type="guide",
            name=f"{filepath.stem}_section_{i}",
            domain=detect_domain(section),
            pattern=detect_pattern(section, filepath.stem),
        ))

    return chunks


# ------------------------------------------------------------------
# Main crawler entry point
# ------------------------------------------------------------------

def crawl() -> list[RawChunk]:
    """
    Clone/update the repo and extract all valuable chunks.
    Returns a flat list of RawChunks ready for the chunker.
    """
    clone_or_update()

    all_chunks: list[RawChunk] = []

    for dir_path in VALUABLE_DIRS:
        target = CLONE_DIR / dir_path
        if not target.exists():
            print(f"  Skipping {dir_path} (not found)")
            continue

        print(f"  Scanning {dir_path}...")

        for filepath in target.rglob("*"):
            if not filepath.is_file():
                continue

            # Skip compiled/cache files
            if any(p in filepath.parts for p in ["__pycache__", ".git"]):
                continue

            if filepath.suffix == ".py":
                if "examples" in filepath.parts or "tests" in filepath.parts:
                    all_chunks.append(extract_example_file(filepath, CLONE_DIR))
                else:
                    all_chunks.extend(extract_classes_from_file(filepath, CLONE_DIR))

            elif filepath.suffix == ".rst":
                all_chunks.extend(extract_rst_sections(filepath, CLONE_DIR))

    print(f"  Extracted {len(all_chunks)} raw chunks from GitHub")
    return all_chunks